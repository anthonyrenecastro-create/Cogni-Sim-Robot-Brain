"""
Planning engine for complex mission planning and recovery strategies.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ActionType(Enum):
    """Types of actions in a plan."""
    MOVE = "move"
    TURN = "turn"
    STOP = "stop"
    WAIT = "wait"
    RECOVER = "recover"
    COMMUNICATE = "communicate"


@dataclass
class PlanStep:
    """A single step in a plan."""
    action: ActionType
    target: Optional[str] = None  # Where to move, what to do
    duration: Optional[float] = None  # How long to wait
    success_condition: str = ""  # How to know if succeeded
    fallback_step: Optional["PlanStep"] = None  # What to do if fails


@dataclass
class Plan:
    """A complete plan for achieving a goal."""
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    estimated_duration: float = 0.0  # seconds
    success_probability: float = 0.5  # [0, 1]
    reasoning: str = ""  # Why this plan


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from a failure."""
    failure_type: str  # Type of failure detected
    recovery_steps: List[PlanStep] = field(default_factory=list)
    max_attempts: int = 3
    escalation: Optional["RecoveryStrategy"] = None  # Next strategy if this fails


class PlanningEngine:
    """
    LLM-based planning engine for complex decisions.
    
    Generates plans for:
    - Navigation and waypoint following
    - Failure recovery
    - Multi-step missions
    - Adaptation to environment
    """
    
    def __init__(self):
        """Initialize planning engine."""
        self.current_plan: Optional[Plan] = None
        self.plan_history: List[Plan] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
    
    def create_plan(
        self,
        goal: str,
        context: str,
        constraints: Optional[List[str]] = None,
    ) -> Plan:
        """
        Create a plan for achieving a goal.
        
        Args:
            goal: Goal to achieve
            context: Current context (from memory stack)
            constraints: Constraints on actions
            
        Returns:
            Plan object with steps
        """
        # This would be called by LLM-based decision making
        plan = Plan(
            goal=goal,
            reasoning=f"Planning for goal: {goal}",
            success_probability=0.7,
        )
        
        self.current_plan = plan
        self.plan_history.append(plan)
        
        return plan
    
    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Execute a plan step-by-step.
        
        Args:
            plan: Plan to execute
            
        Returns:
            Status of execution
        """
        return {
            "status": "executing",
            "current_step": 0,
            "progress": 0.0,
        }
    
    def add_recovery_strategy(
        self,
        failure_type: str,
        strategy: RecoveryStrategy,
    ) -> None:
        """Register a recovery strategy for a failure type."""
        self.recovery_strategies[failure_type] = strategy
    
    def get_recovery_strategy(self, failure_type: str) -> Optional[RecoveryStrategy]:
        """Get recovery strategy for a failure type."""
        return self.recovery_strategies.get(failure_type)
    
    def plan_recovery(
        self,
        failure: str,
        context: str,
    ) -> Optional[RecoveryStrategy]:
        """
        Plan recovery from a failure.
        
        Args:
            failure: Description of failure
            context: Current context
            
        Returns:
            Recovery strategy
        """
        # This would be called by LLM to plan recovery
        strategy = RecoveryStrategy(
            failure_type=failure,
            max_attempts=3,
        )
        
        return strategy


# Common recovery patterns
COMMON_RECOVERY_STRATEGIES = {
    "stuck_obstacle": RecoveryStrategy(
        failure_type="stuck_obstacle",
        recovery_steps=[
            PlanStep(ActionType.STOP, duration=1.0),
            PlanStep(ActionType.TURN, target="right", duration=2.0),
            PlanStep(ActionType.MOVE, target="forward", duration=5.0),
        ],
        max_attempts=3,
    ),
    "sensor_lost": RecoveryStrategy(
        failure_type="sensor_lost",
        recovery_steps=[
            PlanStep(ActionType.STOP, duration=2.0),
            PlanStep(ActionType.WAIT, success_condition="sensor_online"),
        ],
        max_attempts=5,
    ),
    "navigation_failure": RecoveryStrategy(
        failure_type="navigation_failure",
        recovery_steps=[
            PlanStep(ActionType.STOP),
            PlanStep(ActionType.WAIT, duration=5.0),
            PlanStep(ActionType.MOVE, target="home"),
        ],
    ),
}


def get_common_recovery(failure_type: str) -> Optional[RecoveryStrategy]:
    """Get a common recovery strategy for a failure type."""
    return COMMON_RECOVERY_STRATEGIES.get(failure_type)
