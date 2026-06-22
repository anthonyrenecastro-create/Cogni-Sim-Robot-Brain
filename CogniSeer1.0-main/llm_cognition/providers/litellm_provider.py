"""LiteLLM provider - unified proxy for 100+ models."""

from typing import List, Optional, Callable
from .base import LLMProvider, Message, LLMResponse, ModelInfo, ModelType


class LiteLLMProvider(LLMProvider):
    """
    Provider using LiteLLM proxy (unified interface for 100+ models).
    
    LiteLLM supports: OpenAI, Claude, Ollama, Cohere, Replicate,
    HuggingFace, AWS Bedrock, Azure, GCP, and more.
    
    This is useful for:
    - Switching providers without code changes
    - Load balancing across providers
    - Cost optimization
    - Model fallback chains
    """
    
    def _initialize(self) -> None:
        """Initialize LiteLLM provider."""
        try:
            import litellm
            self.litellm = litellm
            litellm.api_key = self.config.get("api_key", "")
        except ImportError:
            raise ImportError("LiteLLM provider requires: pip install litellm")
    
    def list_available_models(self) -> List[ModelInfo]:
        """List supported models via LiteLLM."""
        # This would query LiteLLM for supported models
        return [
            ModelInfo("gpt-4", ModelType.TEXT, 8192, 0.03, 0.06),
            ModelInfo("claude-3-sonnet", ModelType.VISION, 200000, 0.003, 0.015),
            ModelInfo("ollama/llama2", ModelType.TEXT, 4096, 0, 0),
            ModelInfo("replicate/meta/llama-2-70b", ModelType.TEXT, 4096, 0.001, 0.001),
        ]
    
    def complete(self, messages: List[Message], temperature: float = 0.7,
                 max_tokens: int = 1024, top_p: float = 0.9,
                 stop_sequences: Optional[List[str]] = None) -> LLMResponse:
        """Get completion via LiteLLM proxy."""
        raise NotImplementedError("Use: pip install litellm, then implement this")
    
    def stream_complete(self, messages: List[Message], temperature: float = 0.7,
                       max_tokens: int = 1024,
                       callback: Optional[Callable[[str], None]] = None):
        """Stream completion via LiteLLM proxy."""
        raise NotImplementedError("Use: pip install litellm, then implement this")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using LiteLLM."""
        try:
            token_count = self.litellm.token_counter(
                model=self.model_name,
                text=text
            )
            return token_count
        except Exception:
            return len(text) // 4
    
    def supports_vision(self) -> bool:
        """Check if model supports vision."""
        # Vision models in LiteLLM
        vision_models = ["gpt-4-vision", "claude-3", "llava"]
        return any(vm in self.model_name for vm in vision_models)
    
    def get_model_info(self) -> ModelInfo:
        """Get current model info."""
        models = {
            m.name: m for m in self.list_available_models()
        }
        return models.get(self.model_name, ModelInfo(
            self.model_name, ModelType.TEXT, 4096
        ))
