"""
LLM Service Integration Layer
Supports multiple LLM providers: OpenAI, Anthropic, local models
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI provider initialized with model: {model}")
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
            except Exception as e:
                logger.error(f"Error initializing OpenAI: {e}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        if not self.client:
            return {
                'success': False,
                'error': 'OpenAI client not initialized',
                'response': ''
            }

        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            end_time = time.time()

            return {
                'success': True,
                'response': response.choices[0].message.content,
                'model': self.model,
                'tokens_used': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'latency': end_time - start_time
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except:
            # Rough estimation: ~4 chars per token
            return len(text) // 4


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Anthropic provider initialized with model: {model}")
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
            except Exception as e:
                logger.error(f"Error initializing Anthropic: {e}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        if not self.client:
            return {
                'success': False,
                'error': 'Anthropic client not initialized',
                'response': ''
            }

        try:
            start_time = time.time()

            response = self.client.messages.create(
                model=kwargs.get('model', self.model),
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )

            end_time = time.time()

            return {
                'success': True,
                'response': response.content[0].text,
                'model': self.model,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'latency': end_time - start_time
            }

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimation
        return len(text) // 4


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(model)
                logger.info(f"Gemini provider initialized with model: {model}")
            except ImportError:
                logger.error("google-generativeai package not installed. Run: pip install google-generativeai")
            except Exception as e:
                logger.error(f"Error initializing Gemini: {e}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Gemini API"""
        if not self.client:
            return {
                'success': False,
                'error': 'Gemini client not initialized',
                'response': ''
            }

        try:
            start_time = time.time()

            response = self.client.generate_content(
                prompt,
                generation_config={
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_output_tokens': kwargs.get('max_tokens', 2000),
                }
            )

            end_time = time.time()

            return {
                'success': True,
                'response': response.text,
                'model': self.model,
                'tokens_used': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                'latency': end_time - start_time
            }

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4


class CohereProvider(LLMProvider):
    """Cohere AI provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "command"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                import cohere
                self.client = cohere.Client(api_key=self.api_key)
                logger.info(f"Cohere provider initialized with model: {model}")
            except ImportError:
                logger.error("cohere package not installed. Run: pip install cohere")
            except Exception as e:
                logger.error(f"Error initializing Cohere: {e}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Cohere API"""
        if not self.client:
            return {
                'success': False,
                'error': 'Cohere client not initialized',
                'response': ''
            }

        try:
            start_time = time.time()

            response = self.client.generate(
                model=kwargs.get('model', self.model),
                prompt=prompt,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            end_time = time.time()

            # Estimate tokens (Cohere doesn't always provide exact counts)
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(response.generations[0].text) // 4

            return {
                'success': True,
                'response': response.generations[0].text,
                'model': self.model,
                'tokens_used': prompt_tokens + completion_tokens,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'latency': end_time - start_time
            }

        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4


class MistralProvider(LLMProvider):
    """Mistral AI provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-medium"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                from mistralai.client import MistralClient
                self.client = MistralClient(api_key=self.api_key)
                logger.info(f"Mistral provider initialized with model: {model}")
            except ImportError:
                logger.error("mistralai package not installed. Run: pip install mistralai")
            except Exception as e:
                logger.error(f"Error initializing Mistral: {e}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Mistral API"""
        if not self.client:
            return {
                'success': False,
                'error': 'Mistral client not initialized',
                'response': ''
            }

        try:
            start_time = time.time()

            response = self.client.chat(
                model=kwargs.get('model', self.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            end_time = time.time()

            return {
                'success': True,
                'response': response.choices[0].message.content,
                'model': self.model,
                'tokens_used': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'latency': end_time - start_time
            }

        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        logger.info(f"Ollama provider initialized with model: {model} at {base_url}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Ollama API"""
        try:
            import requests

            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": kwargs.get('model', self.model),
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', 0.7),
                        "num_predict": kwargs.get('max_tokens', 2000)
                    }
                }
            )

            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'response': data.get('response', ''),
                    'model': self.model,
                    'tokens_used': data.get('eval_count', 0) + data.get('prompt_eval_count', 0),
                    'prompt_tokens': data.get('prompt_eval_count', 0),
                    'completion_tokens': data.get('eval_count', 0),
                    'latency': end_time - start_time
                }
            else:
                return {
                    'success': False,
                    'error': f'Ollama API error: {response.status_code}',
                    'response': ''
                }

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4


class LMStudioProvider(LLMProvider):
    """LM Studio local LLM provider (OpenAI-compatible API)"""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "local-model"):
        self.base_url = base_url
        self.model = model
        logger.info(f"LM Studio provider initialized with model: {model} at {base_url}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using LM Studio API"""
        try:
            import requests

            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": kwargs.get('model', self.model),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get('temperature', 0.7),
                    "max_tokens": kwargs.get('max_tokens', 2000)
                }
            )

            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                usage = data.get('usage', {})

                return {
                    'success': True,
                    'response': data['choices'][0]['message']['content'],
                    'model': self.model,
                    'tokens_used': usage.get('total_tokens', 0),
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'latency': end_time - start_time
                }
            else:
                return {
                    'success': False,
                    'error': f'LM Studio API error: {response.status_code}',
                    'response': ''
                }

        except Exception as e:
            logger.error(f"LM Studio API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4


class MockLLMProvider(LLMProvider):
    """Mock LLM for testing without API calls"""

    def __init__(self):
        logger.info("Mock LLM provider initialized (for testing)")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate mock response"""
        time.sleep(0.5)  # Simulate API latency

        response = f"""
MOCK LLM ANALYSIS:
==================

HIGH-VALUE TRANSACTIONS (>250 GBP):
- Transaction #123: 500.00 GBP - Large purchase at retail store
- Transaction #456: 750.00 GBP - Hotel booking
- Transaction #789: 1200.00 GBP - Electronics purchase

ANOMALY DETECTION:
- Transaction #321: 5000.00 GBP - Unusually large amount (Z-score: 4.5)
- Transaction #654: Multiple small transactions in short time span
- Transaction #987: Foreign currency transaction with unusual pattern

SUMMARY:
Found 3 high-value transactions totaling 2,450.00 GBP
Detected 3 potential anomalies requiring review
"""
        return {
            'success': True,
            'response': response,
            'model': 'mock-llm',
            'tokens_used': 200,
            'prompt_tokens': 100,
            'completion_tokens': 100,
            'latency': 0.5
        }

    def count_tokens(self, text: str) -> int:
        """Mock token count"""
        return len(text) // 4


class LLMService:
    """Main LLM service with provider management"""

    def __init__(self, provider: str = "mock", **kwargs):
        """
        Initialize LLM service

        Args:
            provider: 'openai', 'anthropic', or 'mock'
            **kwargs: Provider-specific arguments
        """
        self.provider_name = provider
        self.provider = self._initialize_provider(provider, **kwargs)
        self.request_history = []

    def _initialize_provider(self, provider: str, **kwargs) -> LLMProvider:
        """Initialize the specified LLM provider"""
        providers = {
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider,
            'gemini': GeminiProvider,
            'cohere': CohereProvider,
            'mistral': MistralProvider,
            'ollama': OllamaProvider,
            'lmstudio': LMStudioProvider,
            'mock': MockLLMProvider
        }

        if provider not in providers:
            logger.warning(f"Unknown provider '{provider}', falling back to mock")
            provider = 'mock'

        provider_class = providers[provider]

        try:
            return provider_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize {provider}: {e}")
            logger.info("Falling back to mock provider")
            return MockLLMProvider()

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from LLM"""
        logger.info(f"Generating response using {self.provider_name}")

        result = self.provider.generate(prompt, **kwargs)

        # Store in history
        self.request_history.append({
            'timestamp': time.time(),
            'provider': self.provider_name,
            'prompt_length': len(prompt),
            'success': result.get('success', False),
            'tokens_used': result.get('tokens_used', 0),
            'latency': result.get('latency', 0)
        })

        return result

    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts"""
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return self.provider.count_tokens(text)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        if not self.request_history:
            return {}

        total_tokens = sum(r['tokens_used'] for r in self.request_history)
        avg_latency = sum(r['latency'] for r in self.request_history) / len(self.request_history)
        success_rate = sum(1 for r in self.request_history if r['success']) / len(self.request_history)

        return {
            'total_requests': len(self.request_history),
            'total_tokens': total_tokens,
            'average_latency': avg_latency,
            'success_rate': success_rate
        }

    def switch_provider(self, provider: str, **kwargs):
        """Switch to a different LLM provider"""
        logger.info(f"Switching from {self.provider_name} to {provider}")
        self.provider_name = provider
        self.provider = self._initialize_provider(provider, **kwargs)
