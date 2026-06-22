"""
Ollama provider for local LLM models.

Ollama enables running open-source models (Llama2, Mistral, Qwen, etc.)
locally without internet or API costs. Perfect for edge deployment.

Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama2
    3. Start server: ollama serve
    4. Configure CogniSeer to use Ollama
"""

import requests
import json
from typing import List, Optional, Dict, Any, Callable
import logging

from .base import LLMProvider, Message, LLMResponse, ModelInfo, ModelType


logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Provider for Ollama local models.
    
    Supports models like:
    - llama2 (7B-70B)
    - mistral
    - neural-chat
    - dolphin-mixtral
    - openchat
    """
    
    def __init__(self, model_name: str = "llama2", **config):
        """
        Initialize Ollama provider.
        
        Args:
            model_name: Model to use (default: llama2)
            **config:
                - base_url: Ollama server URL (default: http://localhost:11434)
                - temperature: Default temperature [0, 2]
                - timeout: Request timeout in seconds
        """
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 120)
        
        super().__init__(model_name, **config)
    
    def _initialize(self) -> None:
        """Verify Ollama server is running and model is available."""
        try:
            # Test connection
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            
            # Check if model is available
            models = [m["name"].split(":")[0] for m in response.json().get("models", [])]
            if self.model_name not in models:
                logger.warning(
                    f"Model {self.model_name} not found. Available: {models}. "
                    f"Pull it with: ollama pull {self.model_name}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: ollama serve"
            ) from e
    
    def list_available_models(self) -> List[ModelInfo]:
        """List models available on the Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = []
            for model in response.json().get("models", []):
                name = model["name"].split(":")[0]
                # Determine if model supports vision
                # (Ollama models with vision capabilities in name)
                supports_vision = "vision" in name or "llava" in name
                
                model_type = ModelType.VISION if supports_vision else ModelType.TEXT
                
                models.append(ModelInfo(
                    name=name,
                    type=model_type,
                    context_window=4096,  # varies by model
                    local=True,
                ))
            return models
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def get_model_info(self) -> ModelInfo:
        """Get information about current model."""
        # Try to get from list, otherwise create default
        models = self.list_available_models()
        for m in models:
            if m.name == self.model_name:
                return m
        
        # Default info if model not found in list
        return ModelInfo(
            name=self.model_name,
            type=ModelType.TEXT,
            context_window=4096,
            local=True,
        )
    
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Get a completion from Ollama."""
        try:
            # Convert messages to prompt
            prompt = self._format_prompt(messages)
            
            # Make request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data.get("response", ""),
                model=self.model_name,
                tokens_used={
                    "input": data.get("prompt_eval_count", 0),
                    "output": data.get("eval_count", 0),
                },
                stop_reason="stop",
                raw_response=data,
            )
        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            raise
    
    def stream_complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """Stream completion from Ollama."""
        try:
            prompt = self._format_prompt(messages)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stream": True,
                },
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    yield token
                    if callback:
                        callback(token)
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (Ollama doesn't provide exact count)."""
        # Rough estimation: ~4 chars per token
        return len(text) // 4
    
    def supports_vision(self) -> bool:
        """Check if current model supports vision."""
        return "vision" in self.model_name or "llava" in self.model_name
    
    def _format_prompt(self, messages: List[Message]) -> str:
        """Format messages into a prompt string."""
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        
        prompt += "Assistant: "
        return prompt
