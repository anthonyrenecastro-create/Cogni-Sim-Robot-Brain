"""
Comprehensive tests for LLM cognition layer.

Tests all components:
- LLM provider interface
- Memory systems (short/long/episodic/spatial)
- Decision router (fast vs slow brain)
- Planning engine
- Language interface
- Full cognition layer integration
"""

import unittest
import asyncio
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

from llm_cognition import CognitionLayer
from llm_cognition.providers.base import LLMProvider, Message, LLMResponse, ModelInfo, ModelType
from llm_cognition.memory_stack import (
    MemoryStack, ShortTermMemory, LongTermMemory,
    EpisodicMemory, SpatialMemory, MemoryEntry
)
from llm_cognition.decision_router import (
    DecisionRouter, DecisionType, Urgency, Decision
)
from llm_cognition.planning import PlanningEngine, Plan, RecoveryStrategy, ActionType
from llm_cognition.language_interface import LanguageInterface


class TestShortTermMemory(unittest.TestCase):
    """Test short-term memory (recent context)."""
    
    def setUp(self):
        self.memory = ShortTermMemory(max_messages=5)
    
    def test_add_and_retrieve(self):
        """Test adding and retrieving messages."""
        self.memory.add("Hello", "user")
        self.memory.add("Hi there", "assistant")
        
        context = self.memory.get_context()
        self.assertIn("Hello", context)
        self.assertIn("Hi there", context)
    
    def test_max_messages_limit(self):
        """Test that max messages limit is enforced."""
        for i in range(10):
            self.memory.add(f"Message {i}", "user")
        
        self.assertEqual(len(self.memory.messages), 5)
        context = self.memory.get_context()
        self.assertIn("Message 9", context)
        self.assertNotIn("Message 0", context)
    
    def test_clear(self):
        """Test clearing memory."""
        self.memory.add("Test", "user")
        self.memory.clear()
        self.assertEqual(len(self.memory.messages), 0)


class TestLongTermMemory(unittest.TestCase):
    """Test long-term memory (semantic knowledge base)."""
    
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.temp_path = Path(tmpdir) / "long_term.json"
            self.memory = LongTermMemory(persistence_path=self.temp_path)
    
    def test_add_entry(self):
        """Test adding entries."""
        self.memory.add("Robot can navigate obstacles", metadata={"type": "fact"})
        self.assertEqual(len(self.memory.entries), 1)
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        a = [1, 0, 0]
        b = [1, 0, 0]
        similarity = self.memory._cosine_similarity(a, b)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Orthogonal vectors
        c = [0, 1, 0]
        similarity = self.memory._cosine_similarity(a, c)
        self.assertAlmostEqual(similarity, 0.0, places=5)
    
    def test_search(self):
        """Test searching with embeddings."""
        # Add entries with embeddings
        self.memory.add("Obstacle at 1m", embedding=[1.0, 0.0, 0.0])
        self.memory.add("Clear path ahead", embedding=[0.9, 0.1, 0.0])
        self.memory.add("Turn right", embedding=[0.0, 1.0, 0.0])
        
        # Search for similar to first embedding
        results = self.memory.search("", embedding=[1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("Obstacle at 1m", results[0].content)


class TestEpisodicMemory(unittest.TestCase):
    """Test episodic memory (mission logs)."""
    
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.temp_path = Path(tmpdir) / "episodic.json"
            self.memory = EpisodicMemory(persistence_path=self.temp_path)
    
    def test_start_and_end_mission(self):
        """Test mission lifecycle."""
        mission = self.memory.start_mission("m1", "navigate_to_goal")
        self.assertFalse(mission.success)
        
        self.memory.end_mission("m1", success=True, outcome="Reached goal")
        
        completed = self.memory.get_successful_missions()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].mission_id, "m1")
    
    def test_add_observations_and_decisions(self):
        """Test adding observations and decisions."""
        mission = self.memory.start_mission("m2", "explore")
        
        self.memory.add_observation("m2", "Found obstacle")
        self.memory.add_decision("m2", "Turned right")
        
        self.memory.end_mission("m2", success=False)
        
        missions = [m for m in self.memory.missions if m.mission_id == "m2"]
        self.assertEqual(len(missions[0].observations), 1)
        self.assertEqual(len(missions[0].decisions), 1)


class TestSpatialMemory(unittest.TestCase):
    """Test spatial memory (map and landmarks)."""
    
    def setUp(self):
        self.memory = SpatialMemory()
    
    def test_add_feature(self):
        """Test adding spatial features."""
        self.memory.add_feature("obstacle_1", (5.0, 3.0), "obstacle")
        self.assertIn("obstacle_1", self.memory.features)
    
    def test_get_nearby_features(self):
        """Test finding nearby features."""
        self.memory.add_feature("obs_1", (0.0, 0.0), "obstacle")
        self.memory.add_feature("obs_2", (1.0, 0.0), "obstacle")
        self.memory.add_feature("obs_3", (10.0, 0.0), "obstacle")
        
        nearby = self.memory.get_nearby_features((0.0, 0.0), radius=2.0)
        self.assertEqual(len(nearby), 2)
    
    def test_confidence_decay(self):
        """Test confidence decay over time."""
        feature = self.memory.add_feature("old_obs", (0.0, 0.0), "obstacle", confidence=1.0)
        
        # Manually age the feature
        self.memory.features["old_obs"].timestamp = datetime.now() - timedelta(hours=2)
        
        self.memory.decay_confidence(decay_factor=0.95)
        
        decayed = self.memory.features["old_obs"].confidence
        self.assertLess(decayed, 1.0)


class TestMemoryStack(unittest.TestCase):
    """Test complete memory stack."""
    
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.memory = MemoryStack(persistence_dir=Path(tmpdir))
    
    def test_get_full_context(self):
        """Test getting comprehensive context."""
        self.memory.short_term.add("Moving forward", "user")
        self.memory.spatial.add_feature("obstacle", (1.0, 0.0), "obstacle")
        
        context = self.memory.get_full_context(robot_position=(0.0, 0.0), context_radius=5.0)
        
        self.assertIn("Moving forward", context)
        self.assertIn("obstacle", context)


class TestDecisionRouter(unittest.TestCase):
    """Test hybrid decision routing."""
    
    def setUp(self):
        self.router = DecisionRouter()
    
    def test_should_use_slow_brain_critical_urgency(self):
        """Fast brain for critical decisions."""
        use_slow = self.router._should_use_slow_brain(
            DecisionType.OBSTACLE_AVOIDANCE,
            Urgency.CRITICAL,
            slow_brain_ready=True
        )
        self.assertFalse(use_slow)
    
    def test_should_use_slow_brain_planning(self):
        """Slow brain for planning."""
        use_slow = self.router._should_use_slow_brain(
            DecisionType.PLANNING,
            Urgency.LOW,
            slow_brain_ready=True
        )
        self.assertTrue(use_slow)
    
    def test_route_decision_fast(self):
        """Test routing to fast brain."""
        decision = self.router._route_to_fast_brain(
            observation={"sensors": [0.1, 0.2]},
            decision_type=DecisionType.VELOCITY_CONTROL,
            urgency=Urgency.HIGH,
            fast_brain_output={"action": "slow_down", "confidence": 0.8}
        )
        
        self.assertEqual(decision.action, "slow_down")
        self.assertFalse(decision.use_llm)
        self.assertEqual(decision.confidence, 0.8)
    
    def test_statistics(self):
        """Test decision statistics."""
        self.router.route(
            observation={},
            decision_type=DecisionType.VELOCITY_CONTROL,
            urgency=Urgency.HIGH,
            fast_brain_output={"action": "move", "confidence": 0.9}
        )
        
        stats = self.router.get_statistics()
        self.assertEqual(stats["total_decisions"], 1)
        self.assertEqual(stats["fast_brain_decisions"], 1)


class TestPlanningEngine(unittest.TestCase):
    """Test planning engine."""
    
    def setUp(self):
        self.planner = PlanningEngine()
    
    def test_create_plan(self):
        """Test creating a plan."""
        plan = self.planner.create_plan(
            goal="navigate_to_goal",
            context="Area is clear"
        )
        
        self.assertEqual(plan.goal, "navigate_to_goal")
        self.assertIn(plan, self.planner.plan_history)
    
    def test_add_recovery_strategy(self):
        """Test adding recovery strategy."""
        strategy = RecoveryStrategy(failure_type="stuck")
        self.planner.add_recovery_strategy("stuck", strategy)
        
        retrieved = self.planner.get_recovery_strategy("stuck")
        self.assertEqual(retrieved.failure_type, "stuck")


class TestLanguageInterface(unittest.TestCase):
    """Test language generation."""
    
    def setUp(self):
        self.language = LanguageInterface()
    
    def test_explain_decision(self):
        """Test decision explanation."""
        explanation = self.language.explain_decision(
            decision_type="obstacle_avoidance",
            action="turn_right",
            confidence=0.8,
            reason="Obstacle detected ahead"
        )
        
        self.assertIn("turn_right", explanation)
        self.assertIn("confident", explanation)
    
    def test_summarize_mission(self):
        """Test mission summary."""
        summary = self.language.summarize_mission(
            goal="explore_area",
            success=True,
            duration=120.0,
            observations=["Found obstacle", "Path clear"],
            outcome="Area explored"
        )
        
        self.assertIn("explore_area", summary)
        self.assertIn("successfully", summary)
        self.assertIn("120", summary)
    
    def test_confidence_to_language(self):
        """Test confidence level formatting."""
        high = self.language._confidence_to_language(0.95)
        low = self.language._confidence_to_language(0.3)
        
        self.assertIn("confident", high)
        self.assertIn("not", low)


class TestCognitionLayer(unittest.TestCase):
    """Test complete cognition layer."""
    
    def setUp(self):
        # Use Ollama if available, otherwise skip LLM tests
        try:
            self.cognition = CognitionLayer(provider="ollama", model="llama2")
            self.llm_available = True
        except Exception:
            self.cognition = None
            self.llm_available = False
    
    def test_initialization(self):
        """Test cognition layer initialization."""
        if not self.llm_available:
            self.skipTest("Ollama not available")
        
        self.assertIsNotNone(self.cognition)
        self.assertIsNotNone(self.cognition.memory)
        self.assertIsNotNone(self.cognition.router)
    
    def test_decide_method(self):
        """Test decision making."""
        if not self.llm_available:
            self.skipTest("Ollama not available")
        
        decision = self.cognition.decide(
            observation={"sensors": [0.1, 0.2, 0.3] * 10 + [0.5]},  # 32D
            decision_type=DecisionType.OBSTACLE_AVOIDANCE,
            urgency=Urgency.HIGH,
        )
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision.type, DecisionType.OBSTACLE_AVOIDANCE)
    
    def test_mission_lifecycle(self):
        """Test starting and ending a mission."""
        if not self.llm_available:
            self.skipTest("Ollama not available")
        
        mission_id = self.cognition.start_mission("test_mission")
        self.assertEqual(self.cognition.current_mission_id, mission_id)
        
        self.cognition.end_mission(success=True, outcome="Test passed")
        self.assertIsNone(self.cognition.current_mission_id)
    
    def test_add_observation(self):
        """Test adding observations."""
        if not self.llm_available:
            self.skipTest("Ollama not available")
        
        self.cognition.add_observation("Test observation")
        
        # Observation should be in short-term memory
        context = self.cognition.memory.short_term.get_context()
        self.assertIn("Test observation", context)


def run_all_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_all_tests()
