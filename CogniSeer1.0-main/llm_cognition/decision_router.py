"""
Hybrid decision router: Fast brain vs Slow brain.

Routes decisions based on urgency and complexity:
- Fast decisions: Uses heuristics and neural network (PyTorch) - 20Hz
- Slow decisions: Uses LLM for reasoning and planning - 0.1-1 Hz
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions the robot needs to make."""
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"  # Immediate physical response
    EMERGENCY_STOP = "emergency_stop"  # Safety override
    VELOCITY_CONTROL = "velocity_control"  # Speed/direction
    NAVIGATION = "navigation"  # Waypoint following
    PLANNING = "planning"  # Complex planning
    RECOVERY = "recovery"  # Handling failures
    COMMUNICATION = "communication"  # Language/explanation


class Urgency(Enum):
    """Urgency levels for decisions."""
    CRITICAL = "critical"  # < 100ms (emergency stop, collision)
    HIGH = "high"  # < 500ms (obstacle avoidance)
    MEDIUM = "medium"  # < 5s (navigation adjustments)
    LOW = "low"  # < 30s (planning, long-term decisions)


@dataclass
class Decision:
    """A decision made by the robot brain."""
    type: DecisionType
    urgency: Urgency
    action: str  # What to do
    confidence: float  # [0, 1] how confident
    reason: str  # Why this action
    use_llm: bool  # Whether LLM was used
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DecisionRouter:
    """
    Routes decisions to fast or slow brain based on urgency and complexity.
    
    Fast Brain (PyTorch):
    - Obstacle avoidance
    - Emergency stops
    - Velocity control
    - Reactive responses
    - Runs at 20 Hz
    
    Slow Brain (LLM):
    - Planning and navigation
    - Recovery strategy selection
    - Language generation
    - Explanations
    - Runs at 0.1-1 Hz, async
    """
    
    def __init__(self):
        """Initialize decision router."""
        # Decision statistics
        self.decisions_made = 0
        self.fast_decisions = 0
        self.slow_decisions = 0
        self.avg_confidence = 0.0
    
    def route(
        self,
        observation: Dict[str, Any],
        decision_type: DecisionType,
        urgency: Urgency,
        goal: Optional[str] = None,
        fast_brain_output: Optional[Dict[str, Any]] = None,
        slow_brain_ready: bool = False,
    ) -> Decision:
        """
        Route a decision to fast or slow brain.
        
        Args:
            observation: Current sensor observation (32D vector)
            decision_type: Type of decision needed
            urgency: How urgent is this decision?
            goal: Current mission goal
            fast_brain_output: Output from PyTorch fast brain
            slow_brain_ready: Whether slow brain has new output
            
        Returns:
            Decision object with action and reasoning
        """
        # Decide whether to use slow brain
        use_slow = self._should_use_slow_brain(
            decision_type, urgency, slow_brain_ready
        )
        
        if use_slow and slow_brain_ready:
            decision = self._route_to_slow_brain(
                observation, decision_type, urgency, goal
            )
            self.slow_decisions += 1
        else:
            decision = self._route_to_fast_brain(
                observation, decision_type, urgency, fast_brain_output
            )
            self.fast_decisions += 1
        
        # Update statistics
        self.decisions_made += 1
        self.avg_confidence = (
            (self.avg_confidence * (self.decisions_made - 1) + decision.confidence)
            / self.decisions_made
        )
        
        return decision
    
    def _should_use_slow_brain(
        self,
        decision_type: DecisionType,
        urgency: Urgency,
        slow_brain_ready: bool,
    ) -> bool:
        """Determine if slow brain should be consulted."""
        # Never use slow brain for critical/emergency decisions
        if urgency in [Urgency.CRITICAL, Urgency.HIGH]:
            return False
        
        # Don't use if slow brain isn't ready yet
        if not slow_brain_ready:
            return False
        
        # Use slow brain for planning, recovery, communication
        if decision_type in [
            DecisionType.PLANNING,
            DecisionType.RECOVERY,
            DecisionType.COMMUNICATION,
        ]:
            return True
        
        # Use slow brain for navigation if not urgent
        if decision_type == DecisionType.NAVIGATION and urgency == Urgency.LOW:
            return True
        
        return False
    
    def _route_to_fast_brain(
        self,
        observation: Dict[str, Any],
        decision_type: DecisionType,
        urgency: Urgency,
        fast_brain_output: Optional[Dict[str, Any]],
    ) -> Decision:
        """Route to fast brain (PyTorch neural network)."""
        if not fast_brain_output:
            # Fallback: safe default behavior
            return Decision(
                type=decision_type,
                urgency=urgency,
                action="stop",
                confidence=0.5,
                reason="No fast brain output available - safe fallback",
                use_llm=False,
            )
        
        # Use fast brain output
        return Decision(
            type=decision_type,
            urgency=urgency,
            action=fast_brain_output.get("action", "stop"),
            confidence=fast_brain_output.get("confidence", 0.5),
            reason=f"Fast brain ({decision_type.value})",
            use_llm=False,
            metadata=fast_brain_output,
        )
    
    def _route_to_slow_brain(
        self,
        observation: Dict[str, Any],
        decision_type: DecisionType,
        urgency: Urgency,
        goal: Optional[str],
    ) -> Decision:
        """Route to slow brain (LLM-based planning)."""
        # In real implementation, this would call the LLM
        # For now, return placeholder
        return Decision(
            type=decision_type,
            urgency=urgency,
            action="awaiting_llm_decision",
            confidence=0.0,
            reason="Waiting for LLM planning module",
            use_llm=True,
            metadata={"goal": goal},
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics."""
        return {
            "total_decisions": self.decisions_made,
            "fast_brain_decisions": self.fast_decisions,
            "slow_brain_decisions": self.slow_decisions,
            "fast_brain_percentage": (
                100 * self.fast_decisions / self.decisions_made
                if self.decisions_made > 0
                else 0
            ),
            "average_confidence": self.avg_confidence,
        }


# Fast brain heuristics and defaults
FAST_BRAIN_BEHAVIORS = {
    # Obstacle avoidance
    "obstacle_detected": {
        "action": "turn_right",
        "confidence": 0.8,
        "urgency": Urgency.HIGH,
    },
    # Emergency stop
    "collision_imminent": {
        "action": "emergency_stop",
        "confidence": 0.95,
        "urgency": Urgency.CRITICAL,
    },
    # Stale sensor
    "sensor_stale": {
        "action": "stop_and_assess",
        "confidence": 0.7,
        "urgency": Urgency.HIGH,
    },
    # Safe motion
    "path_clear": {
        "action": "continue_navigation",
        "confidence": 0.9,
        "urgency": Urgency.MEDIUM,
    },
}


def get_fast_brain_action(condition: str) -> Optional[Dict[str, Any]]:
    """Get fast brain's recommended action for a condition."""
    return FAST_BRAIN_BEHAVIORS.get(condition)
