"""
LLM Cognition Layer for CogniSeer Robot Brain

This module provides a universal, provider-agnostic LLM integration layer
that enables the slow brain (planning, language, memory) to complement
the fast brain (PyTorch control, obstacle avoidance, emergency stop).

Supported Providers:
- OpenAI (GPT-4, GPT-4V for vision)
- Anthropic Claude (Claude 3 with vision)
- Ollama (local models: Llama2, Mistral, etc.)
- HuggingFace Inference API
- LiteLLM (unified proxy, 100+ models)

Architecture:
- Fast Brain: 20 Hz control loop (PyTorch)
- Slow Brain: LLM-driven planning (0.1-1 Hz)
- Hybrid Router: Decides when to invoke LLM vs fast heuristics
- Memory Stack: Short-term, long-term, episodic, spatial

Example:
    from llm_cognition import CognitionLayer
    
    cognition = CognitionLayer(
        provider="ollama",
        model="llama2",
        memory_config={
            "short_term_size": 10,
            "long_term_enabled": True,
            "episodic_enabled": True,
        }
    )
    
    # Fast decision (routine)
    decision = cognition.decide(
        observation=sensor_data,
        urgency="routine",  # fast heuristics only
        goal="navigate_to_target"
    )
    
    # Slow decision (complex)
    decision = cognition.decide(
        observation=sensor_data,
        urgency="complex",  # invoke LLM
        goal="plan_recovery_from_failure"
    )
"""

from .providers.base import LLMProvider
from .providers.openai_provider import OpenAIProvider
from .providers.claude_provider import ClaudeProvider
from .providers.ollama_provider import OllamaProvider
from .providers.huggingface_provider import HuggingFaceProvider
from .providers.litellm_provider import LiteLLMProvider

from .cognition_layer import CognitionLayer
from .decision_router import DecisionRouter
from .memory_stack import (
    MemoryStack,
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
    SpatialMemory,
)
from .planning import PlanningEngine, Plan, RecoveryStrategy
from .language_interface import LanguageInterface

__version__ = "1.0.0"

__all__ = [
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "ClaudeProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "LiteLLMProvider",
    # Core
    "CognitionLayer",
    "DecisionRouter",
    # Memory
    "MemoryStack",
    "ShortTermMemory",
    "LongTermMemory",
    "EpisodicMemory",
    "SpatialMemory",
    # Planning
    "PlanningEngine",
    "Plan",
    "RecoveryStrategy",
    # Language
    "LanguageInterface",
]
