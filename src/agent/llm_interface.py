"""LLM interface for communicating with various language models."""

from typing import Optional, Dict, Any, List
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import LLMConfig, LLMProvider


class LLMInterface:
    """Interface for interacting with LLM providers."""

    def __init__(self, config: LLMConfig):
        """Initialize LLM interface with configuration.

        Args:
            config: LLM configuration
        """
        if not isinstance(config, LLMConfig):
            raise TypeError("config must be an instance of LLMConfig")

        self.config = config
        self._validate_config()
        self._client = None
        self._call_count = 0
        self._total_tokens = 0

    def _validate_config(self) -> None:
        """Validate the LLM configuration."""
        if self.config.provider != LLMProvider.LOCAL and not self.config.api_key:
            raise ValueError(f"API key required for {self.config.provider} provider")

        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if not 0.0 <= self.config.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response from LLM.

        Args:
            prompt: Input prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response

        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        if not prompt.strip():
            raise ValueError("Prompt cannot be only whitespace")

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if not 0.0 <= temp <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        if tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self._call_count += 1

        # Simulate LLM call for now
        response = self._simulate_llm_call(prompt, temp, tokens, **kwargs)
        self._total_tokens += len(prompt.split()) + len(response.split())

        return response

    def _simulate_llm_call(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Simulate LLM API call.

        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Simulated response
        """
        # Simple simulation based on prompt
        if "error" in prompt.lower():
            raise RuntimeError("Simulated LLM error")

        response_template = f"Response to: {prompt[:50]}..."
        return response_template

    async def generate_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Async version of generate.

        Args:
            prompt: Input prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        # For now, just call sync version
        return self.generate(prompt, temperature, max_tokens, **kwargs)

    def batch_generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            List of generated responses

        Raises:
            ValueError: If prompts is empty or invalid
        """
        if not prompts:
            raise ValueError("prompts list cannot be empty")

        if not isinstance(prompts, list):
            raise TypeError("prompts must be a list")

        if not all(isinstance(p, str) for p in prompts):
            raise TypeError("all prompts must be strings")

        return [self.generate(p, temperature, max_tokens, **kwargs) for p in prompts]

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with call count and token usage
        """
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "provider": self.config.provider.value,
            "model": self.config.model,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._call_count = 0
        self._total_tokens = 0

    def validate_prompt(self, prompt: str) -> bool:
        """Validate if a prompt is suitable for the LLM.

        Args:
            prompt: Prompt to validate

        Returns:
            True if valid, False otherwise
        """
        if not prompt or not isinstance(prompt, str):
            return False

        if not prompt.strip():
            return False

        # Check length (rough token estimation)
        word_count = len(prompt.split())
        if word_count > self.config.max_tokens * 0.75:  # Leave room for response
            return False

        return True
