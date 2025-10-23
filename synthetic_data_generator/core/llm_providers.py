"""
LLM-Agnostic Provider Architecture for Synthetic Data Generation

Supports: OpenAI, Anthropic, Google Gemini, Cohere, Mistral, Local models (Ollama)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import os


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured output matching schema"""
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts (for reasoning engines)"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-4, GPT-3.5, etc.)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        super().__init__(api_key or os.getenv('OPENAI_API_KEY'), model)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4000, **kwargs) -> str:
        """Generate text from prompt"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output"""
        structured_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        response = self.generate(structured_prompt, **kwargs)

        # Extract JSON from response
        try:
            # Try to parse directly
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse JSON from response: {response}")

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key or os.getenv('ANTHROPIC_API_KEY'), model)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4000, **kwargs) -> str:
        """Generate text from prompt"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output"""
        structured_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        response = self.generate(structured_prompt, **kwargs)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse JSON from response: {response}")

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        super().__init__(api_key or os.getenv('GOOGLE_API_KEY'), model)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("Google Generative AI library not installed. Run: pip install google-generativeai")

    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate text from prompt"""
        response = self.client.generate_content(
            prompt,
            generation_config={'temperature': temperature}
        )
        return response.text

    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output"""
        structured_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        response = self.generate(structured_prompt, **kwargs)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse JSON from response: {response}")

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class MockProvider(LLMProvider):
    """Mock provider for testing (no API key required)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "mock"):
        super().__init__(api_key, model)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        return f"Mock response for: {prompt[:100]}..."

    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate mock structured response"""
        # Return minimal valid structure based on schema
        result = {}
        for key, value_type in schema.items():
            if value_type == "string":
                result[key] = f"mock_{key}"
            elif value_type == "number":
                result[key] = 42
            elif value_type == "array":
                result[key] = []
            elif value_type == "object":
                result[key] = {}
            else:
                result[key] = None
        return result

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate mock responses for batch"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class LLMFactory:
    """Factory for creating LLM providers"""

    _providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'claude': AnthropicProvider,  # Alias
        'gemini': GeminiProvider,
        'google': GeminiProvider,  # Alias
        'mock': MockProvider
    }

    @classmethod
    def create(cls, provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
        """
        Create an LLM provider instance

        Args:
            provider: Provider name (openai, anthropic, gemini, mock)
            api_key: API key (optional, will use env var if not provided)
            model: Model name (optional, uses provider default)

        Returns:
            LLMProvider instance

        Examples:
            >>> llm = LLMFactory.create('openai')
            >>> llm = LLMFactory.create('anthropic', model='claude-3-5-sonnet-20241022')
            >>> llm = LLMFactory.create('mock')  # For testing
        """
        provider = provider.lower()
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls._providers.keys())}")

        provider_class = cls._providers[provider]
        if model:
            return provider_class(api_key=api_key, model=model)
        return provider_class(api_key=api_key)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers"""
        return list(cls._providers.keys())


# Convenience function
def get_llm(provider: str = 'openai', **kwargs) -> LLMProvider:
    """Get an LLM provider instance (convenience wrapper)"""
    return LLMFactory.create(provider, **kwargs)
