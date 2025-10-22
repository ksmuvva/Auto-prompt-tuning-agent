"""
Tests for LLM Service Layer
Note: These tests require actual API keys to be set in .env file
Tests will be skipped if API keys are not available
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.llm_service import LLMService, OpenAIProvider, AnthropicProvider


class TestLLMService:
    """Test LLM service"""

    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not set")
    def test_initialization_openai(self):
        """Test service initializes with OpenAI provider"""
        service = LLMService(provider='openai')
        assert service.provider_name == 'openai'
        assert service.provider is not None

    def test_initialization_unknown_provider(self):
        """Test service raises error for unknown provider"""
        with pytest.raises(ValueError):
            service = LLMService(provider='unknown_provider')

    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not set")
    def test_generate(self):
        """Test generate method with OpenAI"""
        service = LLMService(provider='openai')
        result = service.generate("What is 2+2? Answer briefly.")

        assert result['success'] is True
        assert 'response' in result
        assert len(service.request_history) == 1

    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not set")
    def test_usage_stats(self):
        """Test usage statistics tracking"""
        service = LLMService(provider='openai')

        # Generate some requests
        service.generate("Test 1")
        service.generate("Test 2")

        stats = service.get_usage_stats()
        assert stats['total_requests'] == 2
        assert 'total_tokens' in stats
        assert 'average_latency' in stats
        assert 'success_rate' in stats

    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not set")
    def test_switch_provider(self):
        """Test switching providers"""
        service = LLMService(provider='openai')
        original_provider = service.provider_name

        # Switch to same provider (should work)
        service.switch_provider('openai')
        assert service.provider_name == 'openai'


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
