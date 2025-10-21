"""Tests for LLM interface module."""

import pytest
from src.agent.llm_interface import LLMInterface
from src.agent.config import LLMConfig, LLMProvider


class TestLLMInterface:
    """Tests for LLMInterface."""

    @pytest.mark.unit
    def test_initialization(self, default_llm_config):
        """Test LLM interface initialization."""
        llm = LLMInterface(default_llm_config)
        assert llm.config == default_llm_config
        assert llm._call_count == 0
        assert llm._total_tokens == 0

    @pytest.mark.edge_case
    def test_initialization_invalid_config(self):
        """Test initialization with invalid config."""
        with pytest.raises(TypeError):
            LLMInterface("not a config")

        with pytest.raises(TypeError):
            LLMInterface(None)

    @pytest.mark.edge_case
    def test_initialization_missing_api_key(self):
        """Test initialization without API key for non-local provider."""
        config = LLMConfig(provider=LLMProvider.OPENAI, api_key=None)
        with pytest.raises(ValueError, match="API key required"):
            LLMInterface(config)

    @pytest.mark.unit
    def test_generate_basic(self, llm_interface):
        """Test basic text generation."""
        response = llm_interface.generate("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.edge_case
    def test_generate_empty_prompt(self, llm_interface):
        """Test generation with empty prompt."""
        with pytest.raises(ValueError, match="non-empty string"):
            llm_interface.generate("")

    @pytest.mark.edge_case
    def test_generate_whitespace_prompt(self, llm_interface):
        """Test generation with whitespace-only prompt."""
        with pytest.raises(ValueError, match="only whitespace"):
            llm_interface.generate("   ")

        with pytest.raises(ValueError, match="only whitespace"):
            llm_interface.generate("\n\t  ")

    @pytest.mark.edge_case
    def test_generate_none_prompt(self, llm_interface):
        """Test generation with None prompt."""
        with pytest.raises(ValueError):
            llm_interface.generate(None)

    @pytest.mark.edge_case
    def test_generate_non_string_prompt(self, llm_interface):
        """Test generation with non-string prompt."""
        with pytest.raises(ValueError):
            llm_interface.generate(123)

        with pytest.raises(ValueError):
            llm_interface.generate(["list", "of", "words"])

    @pytest.mark.micro
    def test_generate_with_custom_temperature(self, llm_interface):
        """Test generation with custom temperature."""
        response = llm_interface.generate("Test", temperature=0.5)
        assert isinstance(response, str)

    @pytest.mark.edge_case
    def test_generate_invalid_temperature(self, llm_interface):
        """Test generation with invalid temperature."""
        with pytest.raises(ValueError):
            llm_interface.generate("Test", temperature=-0.1)

        with pytest.raises(ValueError):
            llm_interface.generate("Test", temperature=2.1)

    @pytest.mark.micro
    def test_generate_with_custom_max_tokens(self, llm_interface):
        """Test generation with custom max tokens."""
        response = llm_interface.generate("Test", max_tokens=500)
        assert isinstance(response, str)

    @pytest.mark.edge_case
    def test_generate_invalid_max_tokens(self, llm_interface):
        """Test generation with invalid max tokens."""
        with pytest.raises(ValueError):
            llm_interface.generate("Test", max_tokens=0)

        with pytest.raises(ValueError):
            llm_interface.generate("Test", max_tokens=-1)

    @pytest.mark.edge_case
    def test_generate_very_long_prompt(self, llm_interface):
        """Test generation with very long prompt."""
        long_prompt = "word " * 10000
        response = llm_interface.generate(long_prompt)
        assert isinstance(response, str)

    @pytest.mark.edge_case
    def test_generate_special_characters(self, llm_interface):
        """Test generation with special characters."""
        special_prompts = [
            "Test with émojis 🎉🔥",
            "Test with unicode: こんにちは",
            "Test with symbols: @#$%^&*()",
            "Test\nwith\nnewlines",
            "Test\twith\ttabs",
        ]
        for prompt in special_prompts:
            response = llm_interface.generate(prompt)
            assert isinstance(response, str)

    @pytest.mark.unit
    def test_batch_generate(self, llm_interface):
        """Test batch generation."""
        prompts = ["Test 1", "Test 2", "Test 3"]
        responses = llm_interface.batch_generate(prompts)
        assert len(responses) == len(prompts)
        assert all(isinstance(r, str) for r in responses)

    @pytest.mark.edge_case
    def test_batch_generate_empty_list(self, llm_interface):
        """Test batch generation with empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            llm_interface.batch_generate([])

    @pytest.mark.edge_case
    def test_batch_generate_non_list(self, llm_interface):
        """Test batch generation with non-list input."""
        with pytest.raises(TypeError, match="must be a list"):
            llm_interface.batch_generate("not a list")

    @pytest.mark.edge_case
    def test_batch_generate_mixed_types(self, llm_interface):
        """Test batch generation with mixed types."""
        with pytest.raises(TypeError, match="must be strings"):
            llm_interface.batch_generate(["valid", 123, "also valid"])

    @pytest.mark.edge_case
    def test_batch_generate_single_item(self, llm_interface):
        """Test batch generation with single item."""
        responses = llm_interface.batch_generate(["Single prompt"])
        assert len(responses) == 1
        assert isinstance(responses[0], str)

    @pytest.mark.micro
    def test_get_stats(self, llm_interface):
        """Test getting usage statistics."""
        stats = llm_interface.get_stats()
        assert "call_count" in stats
        assert "total_tokens" in stats
        assert "provider" in stats
        assert "model" in stats

    @pytest.mark.micro
    def test_stats_tracking(self, llm_interface):
        """Test that stats are tracked correctly."""
        initial_stats = llm_interface.get_stats()
        assert initial_stats["call_count"] == 0

        llm_interface.generate("Test")
        stats = llm_interface.get_stats()
        assert stats["call_count"] == 1
        assert stats["total_tokens"] > 0

    @pytest.mark.micro
    def test_reset_stats(self, llm_interface):
        """Test resetting statistics."""
        llm_interface.generate("Test")
        assert llm_interface.get_stats()["call_count"] > 0

        llm_interface.reset_stats()
        stats = llm_interface.get_stats()
        assert stats["call_count"] == 0
        assert stats["total_tokens"] == 0

    @pytest.mark.micro
    def test_validate_prompt_valid(self, llm_interface):
        """Test prompt validation with valid prompts."""
        assert llm_interface.validate_prompt("Valid prompt") is True
        assert llm_interface.validate_prompt("Short") is True

    @pytest.mark.edge_case
    def test_validate_prompt_invalid(self, llm_interface):
        """Test prompt validation with invalid prompts."""
        assert llm_interface.validate_prompt("") is False
        assert llm_interface.validate_prompt("   ") is False
        assert llm_interface.validate_prompt(None) is False

    @pytest.mark.edge_case
    def test_validate_prompt_too_long(self, llm_interface):
        """Test prompt validation with very long prompt."""
        # Create a prompt longer than max_tokens allows
        very_long = "word " * (llm_interface.config.max_tokens + 1)
        assert llm_interface.validate_prompt(very_long) is False

    @pytest.mark.edge_case
    def test_generate_error_handling(self, llm_interface):
        """Test error handling during generation."""
        # Simulate error by using special keyword
        with pytest.raises(RuntimeError, match="Simulated LLM error"):
            llm_interface.generate("This should cause an error")

    @pytest.mark.micro
    async def test_generate_async(self, llm_interface):
        """Test async generation."""
        response = await llm_interface.generate_async("Test async")
        assert isinstance(response, str)

    @pytest.mark.edge_case
    async def test_generate_async_invalid_prompt(self, llm_interface):
        """Test async generation with invalid prompt."""
        with pytest.raises(ValueError):
            await llm_interface.generate_async("")

    @pytest.mark.micro
    def test_multiple_generations_increment_count(self, llm_interface):
        """Test that multiple generations increment count."""
        for i in range(5):
            llm_interface.generate(f"Test {i}")

        stats = llm_interface.get_stats()
        assert stats["call_count"] == 5

    @pytest.mark.edge_case
    def test_configuration_boundaries(self):
        """Test LLM interface with boundary configurations."""
        # Minimum values
        config_min = LLMConfig(
            api_key="test",
            temperature=0.0,
            max_tokens=1,
            timeout=1,
            max_retries=0,
        )
        llm_min = LLMInterface(config_min)
        assert llm_min.config.temperature == 0.0

        # Maximum values
        config_max = LLMConfig(
            api_key="test",
            temperature=2.0,
            max_tokens=100000,
            timeout=600,
            max_retries=10,
        )
        llm_max = LLMInterface(config_max)
        assert llm_max.config.temperature == 2.0
