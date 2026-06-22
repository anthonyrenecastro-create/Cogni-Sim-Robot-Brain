"""
Abstract base class for LLM providers.

All LLM providers (OpenAI, Claude, Ollama, etc.) inherit from this
to provide a unified interface. This enables easy swapping of providers
without changing the cognition layer code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from enum import Enum


class ModelType(Enum):
    """Types of models supported."""
    TEXT = "text"  # Text-only LLM
    VISION = "vision"  # Vision-language model (VLM)
    AUDIO = "audio"  # Audio understanding


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    type: ModelType
    context_window: int  # max tokens
    cost_per_1k_input: float = 0.0  # for cloud models
    cost_per_1k_output: float = 0.0
    local: bool = False  # True if runs locally


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str  # Text content
    images: Optional[List[str]] = None  # List of image URLs/paths (for VLM)


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str  # Generated text
    model: str
    tokens_used: Dict[str, int]  # {"input": N, "output": M}
    stop_reason: str  # "stop", "length", "error", etc.
    raw_response: Any = None  # Provider-specific raw response


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement this interface to be compatible with
    the CogniSeer cognition layer. This enables swapping providers
    without code changes.
    
    Providers should handle:
    - Authentication (API keys, credentials)
    - Rate limiting
    - Error handling and retries
    - Cost tracking (for cloud providers)
    - Local model loading (for edge providers)
    """
    
    def __init__(self, model_name: str, **config):
        """
        Initialize the provider.
        
        Args:
            model_name: Name of the model to use
            **config: Provider-specific configuration
        """
        self.model_name = model_name
        self.config = config
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the provider (load models, authenticate, etc.).
        
        Subclasses should override to implement provider-specific setup.
        Raise exceptions on failure.
        """
        pass
    
    @abstractmethod
    def list_available_models(self) -> List[ModelInfo]:
        """
        List models available from this provider.
        
        Returns:
            List of ModelInfo objects describing available models
        """
        pass
    
    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Get a completion from the LLM.
        
        Args:
            messages: Conversation history and prompt
            temperature: Sampling temperature [0, 2]
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that stop generation
            
        Returns:
            LLMResponse with generated text
            
        Raises:
            Exception: On API error, authentication failure, etc.
        """
        pass
    
    @abstractmethod
    def stream_complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Stream a completion token-by-token.
        
        Args:
            messages: Conversation history and prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callback: Called for each token with the token text
            
        Yields:
            Token strings
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text without calling the model.
        
        Args:
            text: Text to count
            
        Returns:
            Number of tokens
        """
        pass
    
    @abstractmethod
    def supports_vision(self) -> bool:
        """Check if this provider/model supports vision (VLM)."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        pass
    
    def is_local(self) -> bool:
        """Check if this is a local model (no API calls)."""
        info = self.get_model_info()
        return info.local
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost of a completion (for cloud providers).
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        info = self.get_model_info()
        input_cost = (input_tokens / 1000) * info.cost_per_1k_input
        output_cost = (output_tokens / 1000) * info.cost_per_1k_output
        return input_cost + output_cost
