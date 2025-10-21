"""
Tests for LLM Service Layer
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.llm_service import LLMService, MockLLMProvider, OpenAIProvider, AnthropicProvider


class TestMockLLMProvider:
    """Test mock LLM provider"""

    def test_initialization(self):
        """Test mock provider initializes correctly"""
        provider = MockLLMProvider()
        assert provider is not None

    def test_generate(self):
        """Test mock provider generates responses"""
        provider = MockLLMProvider()
        result = provider.generate("Test prompt")

        assert result['success'] is True
        assert 'response' in result
        assert len(result['response']) > 0
        assert result['model'] == 'mock-llm'

    def test_count_tokens(self):
        """Test token counting"""
        provider = MockLLMProvider()
        count = provider.count_tokens("This is a test message")
        assert count > 0
        assert isinstance(count, int)


class TestLLMService:
    """Test LLM service"""

    def test_initialization_mock(self):
        """Test service initializes with mock provider"""
        service = LLMService(provider='mock')
        assert service.provider_name == 'mock'
        assert service.provider is not None

    def test_initialization_unknown_provider(self):
        """Test service falls back to mock for unknown provider"""
        service = LLMService(provider='unknown_provider')
        assert service.provider_name == 'unknown_provider'
        assert isinstance(service.provider, MockLLMProvider)

    def test_generate(self):
        """Test generate method"""
        service = LLMService(provider='mock')
        result = service.generate("Test prompt")

        assert result['success'] is True
        assert 'response' in result
        assert len(service.request_history) == 1

    def test_batch_generate(self):
        """Test batch generation"""
        service = LLMService(provider='mock')
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = service.batch_generate(prompts)

        assert len(results) == 3
        assert all(r['success'] for r in results)
        assert len(service.request_history) == 3

    def test_usage_stats(self):
        """Test usage statistics tracking"""
        service = LLMService(provider='mock')

        # Generate some requests
        service.generate("Test 1")
        service.generate("Test 2")

        stats = service.get_usage_stats()
        assert stats['total_requests'] == 2
        assert 'total_tokens' in stats
        assert 'average_latency' in stats
        assert 'success_rate' in stats

    def test_switch_provider(self):
        """Test switching providers"""
        service = LLMService(provider='mock')
        original_provider = service.provider_name

        service.switch_provider('mock')
        assert service.provider_name == 'mock'


class TestProviderInitialization:
    """Test provider initialization edge cases"""

    def test_openai_without_key(self):
        """Test OpenAI provider without API key"""
        import os
        # Temporarily remove key if exists
        old_key = os.environ.get('OPENAI_API_KEY')
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

        provider = OpenAIProvider()
        assert provider is not None

        # Restore key
        if old_key:
            os.environ['OPENAI_API_KEY'] = old_key

    def test_anthropic_without_key(self):
        """Test Anthropic provider without API key"""
        import os
        old_key = os.environ.get('ANTHROPIC_API_KEY')
        if 'ANTHROPIC_API_KEY' in os.environ:
            del os.environ['ANTHROPIC_API_KEY']

        provider = AnthropicProvider()
        assert provider is not None

        if old_key:
            os.environ['ANTHROPIC_API_KEY'] = old_key


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
