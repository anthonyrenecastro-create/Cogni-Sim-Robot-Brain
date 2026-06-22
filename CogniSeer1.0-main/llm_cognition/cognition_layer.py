"""
Main cognition layer combining fast and slow brains.

Fast Brain: PyTorch neural network for reactive control
- 20 Hz control loop
- Obstacle avoidance, emergency stop, velocity control
- Low latency (< 100ms)

Slow Brain: LLM for planning and reasoning
- 0.1-1 Hz planning loop
- Complex planning, recovery, explanations
- Higher latency acceptable (< 5 seconds)
- Async execution (doesn't block fast brain)
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import asyncio
from datetime import datetime

from .providers.base import LLMProvider
from .memory_stack import MemoryStack
from .decision_router import DecisionRouter, Decision, DecisionType, Urgency
from .planning import PlanningEngine, Plan, RecoveryStrategy
from .language_interface import LanguageInterface

logger = logging.getLogger(__name__)


class CognitionLayer:
    """
    Unified cognition layer for the robot brain.
    
    Coordinates fast and slow brains, memory, planning, and language.
    
    Example:
        cognition = CognitionLayer(provider="ollama", model="llama2")
        
        # Make a routine decision (fast brain)
        decision = cognition.decide(
            observation=sensor_data,
            decision_type=DecisionType.OBSTACLE_AVOIDANCE,
            urgency=Urgency.HIGH,
        )
        
        # Make a complex decision (async slow brain)
        asyncio.create_task(cognition.plan_mission(goal="navigate_to_target"))
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama2",
        memory_persistence_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize cognition layer.
        
        Args:
            provider: LLM provider ("ollama", "openai", "claude", etc.)
            model: Model name for the provider
            memory_persistence_dir: Directory for persistent memory
            config: Provider-specific configuration
        """
        self.provider_name = provider
        self.model_name = model
        self.config = config or {}
        
        # Initialize components
        self.llm: Optional[LLMProvider] = None
        self._init_llm(provider, model)
        
        self.memory = MemoryStack(persistence_dir=memory_persistence_dir)
        self.router = DecisionRouter()
        self.planner = PlanningEngine()
        self.language = LanguageInterface()
        
        # State
        self.is_initialized = True
        self.current_goal: Optional[str] = None
        self.current_mission_id: Optional[str] = None
        self.decisions_log: List[Decision] = []
        
        logger.info(f"Cognition layer initialized with {provider}/{model}")
    
    def _init_llm(self, provider: str, model: str) -> None:
        """Initialize the LLM provider."""
        provider_map = {
            "ollama": "ollm_cognition.providers.ollama_provider.OllamaProvider",
            "openai": "llm_cognition.providers.openai_provider.OpenAIProvider",
            "claude": "llm_cognition.providers.claude_provider.ClaudeProvider",
            "huggingface": "llm_cognition.providers.huggingface_provider.HuggingFaceProvider",
            "litellm": "llm_cognition.providers.litellm_provider.LiteLLMProvider",
        }
        
        if provider not in provider_map:
            logger.warning(f"Unknown provider {provider}, falling back to Ollama")
            provider = "ollama"
        
        try:
            provider_path = provider_map[provider]
            parts = provider_path.rsplit(".", 1)
            module_path, class_name = parts[0], parts[1]
            
            # Lazy import to avoid hard dependencies
            import importlib
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
            
            self.llm = provider_class(model, **self.config)
            logger.info(f"Initialized {provider} provider with {model}")
        except Exception as e:
            logger.error(f"Failed to initialize {provider}/{model}: {e}")
            self.llm = None
    
    def decide(
        self,
        observation: Dict[str, Any],
        decision_type: DecisionType,
        urgency: Urgency = Urgency.MEDIUM,
        goal: Optional[str] = None,
        fast_brain_output: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
    ) -> Decision:
        """
        Make a decision based on observation and context.
        
        This is the main decision-making interface. It routes to fast or slow
        brain depending on urgency and decision type.
        
        Args:
            observation: Current sensor observation (32D vector, etc.)
            decision_type: Type of decision (obstacle_avoidance, planning, etc.)
            urgency: How urgent is this decision?
            goal: Current mission goal
            fast_brain_output: Pre-computed output from PyTorch brain
            context: Optional context string
            
        Returns:
            Decision with action and reasoning
        """
        # Check if slow brain has new output ready
        slow_brain_ready = self.llm is not None  # Simplified check
        
        # Route decision
        decision = self.router.route(
            observation=observation,
            decision_type=decision_type,
            urgency=urgency,
            goal=goal,
            fast_brain_output=fast_brain_output,
            slow_brain_ready=slow_brain_ready,
        )
        
        # Log decision
        self.decisions_log.append(decision)
        
        # Update memory
        self.memory.short_term.add(
            content=f"{decision_type.value}: {decision.action}",
            role="decision",
            metadata={
                "urgency": urgency.value,
                "confidence": decision.confidence,
            }
        )
        
        return decision
    
    async def plan_mission(
        self,
        goal: str,
        context: Optional[str] = None,
    ) -> Optional[Plan]:
        """
        Asynchronously plan a mission using the slow brain.
        
        This runs at low frequency without blocking the fast brain.
        
        Args:
            goal: Goal to achieve
            context: Optional context
            
        Returns:
            Plan for achieving goal
        """
        if not self.llm:
            logger.warning("LLM not available, cannot plan mission")
            return None
        
        self.current_goal = goal
        
        # Get context from memory
        if context is None:
            context = self.memory.get_full_context()
        
        # Create plan
        plan = self.planner.create_plan(goal, context)
        
        # Log to episodic memory
        if self.current_mission_id:
            self.memory.episodic.add_decision(
                self.current_mission_id,
                f"Planned: {goal}"
            )
        
        logger.info(f"Planned mission: {goal}")
        return plan
    
    async def handle_failure(
        self,
        failure: str,
        context: Optional[str] = None,
    ) -> Optional[RecoveryStrategy]:
        """
        Asynchronously plan recovery from a failure.
        
        Args:
            failure: Description of failure
            context: Optional context
            
        Returns:
            Recovery strategy
        """
        if context is None:
            context = self.memory.get_full_context()
        
        # Try to get recovery strategy
        strategy = self.planner.plan_recovery(failure, context)
        
        # Log to memory
        if self.current_mission_id:
            self.memory.episodic.add_observation(
                self.current_mission_id,
                f"Failure: {failure}"
            )
        
        # Generate explanation
        explanation = self.language.apologize_and_recover(failure.lower())
        logger.info(f"Handling failure: {failure}\n{explanation}")
        
        return strategy
    
    async def explain_decision(
        self,
        decision: Decision,
    ) -> str:
        """Generate a natural language explanation of a decision."""
        return self.language.explain_decision(
            decision_type=decision.type.value,
            action=decision.action,
            confidence=decision.confidence,
            reason=decision.reason,
        )
    
    async def generate_status_report(self) -> str:
        """Generate a status report of the robot's state."""
        # Gather information
        router_stats = self.router.get_statistics()
        
        report = "=== Cognition Layer Status ===\n"
        report += f"Decisions made: {router_stats['total_decisions']}\n"
        report += f"Fast brain: {router_stats['fast_brain_decisions']} "
        report += f"({router_stats['fast_brain_percentage']:.1f}%)\n"
        report += f"Slow brain: {router_stats['slow_brain_decisions']}\n"
        report += f"Average confidence: {router_stats['average_confidence']:.2f}\n"
        report += f"Current goal: {self.current_goal or 'None'}\n"
        
        return report
    
    def start_mission(self, goal: str) -> str:
        """
        Start a new mission.
        
        Returns:
            Mission ID
        """
        mission_id = f"mission_{datetime.now().isoformat()}"
        self.current_mission_id = mission_id
        self.current_goal = goal
        
        self.memory.episodic.start_mission(mission_id, goal)
        
        logger.info(f"Started mission {mission_id}: {goal}")
        return mission_id
    
    def end_mission(self, success: bool, outcome: str = "") -> None:
        """
        End the current mission.
        
        Args:
            success: Whether mission succeeded
            outcome: Description of outcome
        """
        if not self.current_mission_id:
            return
        
        self.memory.episodic.end_mission(
            self.current_mission_id,
            success=success,
            outcome=outcome,
        )
        
        # Generate summary
        if self.current_mission_id in [m.mission_id for m in self.memory.episodic.missions]:
            mission = next(
                m for m in self.memory.episodic.missions
                if m.mission_id == self.current_mission_id
            )
            summary = self.language.summarize_mission(
                goal=mission.goal,
                success=mission.success,
                duration=(mission.end_time - mission.start_time).total_seconds(),
                observations=mission.observations,
                outcome=mission.outcome,
            )
            logger.info(f"Mission summary:\n{summary}")
        
        self.current_mission_id = None
        self.current_goal = None
    
    def add_observation(self, observation: str) -> None:
        """Add an observation to memory."""
        self.memory.short_term.add(observation, role="observation")
        
        if self.current_mission_id:
            self.memory.episodic.add_observation(
                self.current_mission_id,
                observation
            )
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about decisions made."""
        return self.router.get_statistics()
