"""
Memory systems for the cognition layer.

Implements 4 types of memory:
1. Short-term: Recent conversation context (for current reasoning)
2. Long-term: Semantic embeddings (for knowledge retrieval)
3. Episodic: Mission logs (for learning from past experiences)
4. Spatial: Map annotations (for environmental understanding)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # For semantic search


@dataclass
class MissionLog:
    """Log of a completed mission."""
    mission_id: str
    goal: str
    start_time: datetime
    end_time: datetime
    success: bool
    observations: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    outcome: str = ""  # Description of result
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class SpatialFeature:
    """A spatial landmark or obstacle in memory."""
    name: str
    position: Tuple[float, float]  # (x, y) in meters
    type: str  # "landmark", "obstacle", "goal", "danger"
    confidence: float = 1.0  # How sure we are about this
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """
    Short-term memory: Recent conversation context.
    
    Stores the N most recent messages for context window.
    Used by the LLM to understand current conversation.
    """
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize short-term memory.
        
        Args:
            max_messages: Maximum messages to keep in context
        """
        self.max_messages = max_messages
        self.messages: List[MemoryEntry] = []
    
    def add(self, content: str, role: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to short-term memory."""
        entry = MemoryEntry(
            content=content,
            metadata={"role": role, **(metadata or {})}
        )
        self.messages.append(entry)
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> str:
        """Get formatted context from recent messages."""
        context = ""
        for entry in self.messages:
            role = entry.metadata.get("role", "unknown")
            context += f"{role}: {entry.content}\n"
        return context
    
    def clear(self) -> None:
        """Clear all short-term memory."""
        self.messages = []


class LongTermMemory:
    """
    Long-term memory: Semantic knowledge base.
    
    Stores important facts and learns from experiences.
    Uses embeddings for semantic search.
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        """
        Initialize long-term memory.
        
        Args:
            persistence_path: Path to save memory to disk
        """
        self.persistence_path = persistence_path
        self.entries: List[MemoryEntry] = []
        self._load_from_disk()
    
    def add(self, content: str, embedding: Optional[List[float]] = None,
            metadata: Optional[Dict] = None) -> None:
        """Add a memory entry."""
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.entries.append(entry)
        self._save_to_disk()
    
    def search(self, query: str, embedding: Optional[List[float]] = None,
               top_k: int = 5) -> List[MemoryEntry]:
        """
        Search memory using semantic similarity.
        
        Args:
            query: Text query
            embedding: Embedding of query (for semantic search)
            top_k: Number of results to return
            
        Returns:
            Top K most relevant memories
        """
        if not embedding or not self.entries:
            return []
        
        # Compute similarity scores
        similarities = []
        for entry in self.entries:
            if entry.embedding:
                similarity = self._cosine_similarity(embedding, entry.embedding)
                similarities.append((entry, similarity))
        
        # Return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in similarities[:top_k]]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _save_to_disk(self) -> None:
        """Persist memory to disk."""
        if not self.persistence_path:
            return
        
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "content": e.content,
                "timestamp": e.timestamp.isoformat(),
                "embedding": e.embedding,
                "metadata": e.metadata,
            }
            for e in self.entries
        ]
        
        with open(self.persistence_path, "w") as f:
            json.dump(data, f)
    
    def _load_from_disk(self) -> None:
        """Load memory from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        with open(self.persistence_path, "r") as f:
            data = json.load(f)
        
        self.entries = [
            MemoryEntry(
                content=e["content"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                embedding=e.get("embedding"),
                metadata=e.get("metadata", {}),
            )
            for e in data
        ]


class EpisodicMemory:
    """
    Episodic memory: Mission logs and outcomes.
    
    Records what happened during missions for learning and replay.
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        """Initialize episodic memory."""
        self.persistence_path = persistence_path
        self.missions: List[MissionLog] = []
        self._load_from_disk()
    
    def start_mission(self, mission_id: str, goal: str) -> MissionLog:
        """Start logging a new mission."""
        mission = MissionLog(
            mission_id=mission_id,
            goal=goal,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=False,
        )
        self.missions.append(mission)
        return mission
    
    def end_mission(self, mission_id: str, success: bool, outcome: str = "") -> None:
        """Mark a mission as complete."""
        for mission in self.missions:
            if mission.mission_id == mission_id:
                mission.end_time = datetime.now()
                mission.success = success
                mission.outcome = outcome
                self._save_to_disk()
                return
    
    def add_observation(self, mission_id: str, observation: str) -> None:
        """Log an observation during a mission."""
        for mission in self.missions:
            if mission.mission_id == mission_id:
                mission.observations.append(observation)
                return
    
    def add_decision(self, mission_id: str, decision: str) -> None:
        """Log a decision during a mission."""
        for mission in self.missions:
            if mission.mission_id == mission_id:
                mission.decisions.append(decision)
                return
    
    def get_successful_missions(self, goal_filter: Optional[str] = None) -> List[MissionLog]:
        """Get successful missions, optionally filtered by goal."""
        missions = [m for m in self.missions if m.success]
        if goal_filter:
            missions = [m for m in missions if goal_filter in m.goal]
        return missions
    
    def _save_to_disk(self) -> None:
        """Persist missions to disk."""
        if not self.persistence_path:
            return
        
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "mission_id": m.mission_id,
                "goal": m.goal,
                "start_time": m.start_time.isoformat(),
                "end_time": m.end_time.isoformat(),
                "success": m.success,
                "outcome": m.outcome,
                "observations": m.observations,
                "decisions": m.decisions,
                "lessons_learned": m.lessons_learned,
            }
            for m in self.missions
        ]
        
        with open(self.persistence_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_from_disk(self) -> None:
        """Load missions from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        with open(self.persistence_path, "r") as f:
            data = json.load(f)
        
        self.missions = [
            MissionLog(
                mission_id=m["mission_id"],
                goal=m["goal"],
                start_time=datetime.fromisoformat(m["start_time"]),
                end_time=datetime.fromisoformat(m["end_time"]),
                success=m["success"],
                outcome=m.get("outcome", ""),
                observations=m.get("observations", []),
                decisions=m.get("decisions", []),
                lessons_learned=m.get("lessons_learned", []),
            )
            for m in data
        ]


class SpatialMemory:
    """
    Spatial memory: Environmental landmarks and obstacles.
    
    Maintains a map-like representation of known locations,
    obstacles, goals, and danger zones.
    """
    
    def __init__(self):
        """Initialize spatial memory."""
        self.features: Dict[str, SpatialFeature] = {}
    
    def add_feature(self, name: str, position: Tuple[float, float],
                   feature_type: str, confidence: float = 1.0,
                   metadata: Optional[Dict] = None) -> None:
        """Add a spatial feature (landmark, obstacle, etc.)."""
        feature = SpatialFeature(
            name=name,
            position=position,
            type=feature_type,
            confidence=confidence,
            metadata=metadata or {}
        )
        self.features[name] = feature
    
    def get_nearby_features(self, position: Tuple[float, float],
                           radius: float, feature_type: Optional[str] = None
                           ) -> List[SpatialFeature]:
        """Get features near a position."""
        nearby = []
        for feature in self.features.values():
            # Euclidean distance
            dx = feature.position[0] - position[0]
            dy = feature.position[1] - position[1]
            distance = (dx * dx + dy * dy) ** 0.5
            
            if distance <= radius:
                if feature_type is None or feature.type == feature_type:
                    nearby.append(feature)
        
        return sorted(nearby, key=lambda f: (
            (f.position[0] - position[0])**2 + (f.position[1] - position[1])**2
        ) ** 0.5)
    
    def get_obstacles(self, position: Tuple[float, float],
                     radius: float) -> List[SpatialFeature]:
        """Get all obstacles near a position."""
        return self.get_nearby_features(position, radius, "obstacle")
    
    def get_landmarks(self, position: Tuple[float, float],
                     radius: float) -> List[SpatialFeature]:
        """Get all landmarks near a position."""
        return self.get_nearby_features(position, radius, "landmark")
    
    def decay_confidence(self, decay_factor: float = 0.95) -> None:
        """Decay confidence of old memories (older = less confident)."""
        now = datetime.now()
        for feature in self.features.values():
            age_seconds = (now - feature.timestamp).total_seconds()
            # Confidence decays exponentially with age
            feature.confidence *= decay_factor ** (age_seconds / 3600)  # per hour


class MemoryStack:
    """
    Complete memory system combining all 4 memory types.
    
    Provides unified interface for the cognition layer to access all memories.
    """
    
    def __init__(self, persistence_dir: Optional[Path] = None):
        """
        Initialize memory stack.
        
        Args:
            persistence_dir: Directory to save persistent memories
        """
        self.short_term = ShortTermMemory(max_messages=10)
        self.long_term = LongTermMemory(
            persistence_path=persistence_dir / "long_term.json" if persistence_dir else None
        )
        self.episodic = EpisodicMemory(
            persistence_path=persistence_dir / "episodic.json" if persistence_dir else None
        )
        self.spatial = SpatialMemory()
    
    def get_full_context(self, robot_position: Optional[Tuple[float, float]] = None,
                        context_radius: float = 5.0) -> str:
        """Get comprehensive context for LLM decision making."""
        context = ""
        
        # Recent messages
        context += "=== Recent Context ===\n"
        context += self.short_term.get_context()
        
        # Spatial context
        if robot_position:
            context += "\n=== Nearby Environment ===\n"
            obstacles = self.spatial.get_obstacles(robot_position, context_radius)
            for obs in obstacles:
                context += f"- Obstacle '{obs.name}' at {obs.position}\n"
            
            landmarks = self.spatial.get_landmarks(robot_position, context_radius)
            for landmark in landmarks:
                context += f"- Landmark '{landmark.name}' at {landmark.position}\n"
        
        return context
