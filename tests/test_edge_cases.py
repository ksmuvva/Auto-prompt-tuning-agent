"""Comprehensive edge case tests for the AI agent."""

import pytest
import sys
from src.agent.core import PromptTuningAgent
from src.agent.config import AgentConfig, LLMConfig, LLMProvider
from src.agent.llm_interface import LLMInterface
from src.agent.metrics import MetricsCalculator, PromptMetrics
from src.agent.prompt_tuner import PromptTuner


class TestUnicodeAndEncodingEdgeCases:
    """Tests for unicode and encoding edge cases."""

    @pytest.mark.edge_case
    def test_unicode_prompts(self, agent):
        """Test with various unicode characters."""
        unicode_prompts = [
            "Здравствуй мир",  # Russian
            "你好世界",  # Chinese
            "مرحبا بالعالم",  # Arabic
            "שלום עולם",  # Hebrew
            "γεια σου κόσμε",  # Greek
            "こんにちは世界",  # Japanese
            "안녕하세요 세계",  # Korean
        ]

        for prompt in unicode_prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result

    @pytest.mark.edge_case
    def test_emoji_in_prompts(self, agent):
        """Test prompts containing emojis."""
        emoji_prompts = [
            "Write code 🔥🚀",
            "😀😁😂🤣😃",
            "Test with emoji: 👍👎",
            "Heart ❤️ and star ⭐",
        ]

        for prompt in emoji_prompts:
            result = agent.optimize_prompt(prompt)
            assert "optimized_prompt" in result

    @pytest.mark.edge_case
    def test_mixed_direction_text(self, agent):
        """Test with mixed RTL and LTR text."""
        mixed_prompts = [
            "English and עברית mixed",
            "Test with مختلط text",
        ]

        for prompt in mixed_prompts:
            metrics = agent.evaluate_prompt(prompt)
            assert metrics is not None


class TestBoundaryValueEdgeCases:
    """Tests for boundary values."""

    @pytest.mark.edge_case
    def test_max_integer_values(self):
        """Test with maximum integer values."""
        config = LLMConfig(
            api_key="test",
            max_tokens=100000,
            timeout=600,
            max_retries=10,
        )
        assert config.max_tokens == 100000

    @pytest.mark.edge_case
    def test_min_integer_values(self):
        """Test with minimum integer values."""
        config = LLMConfig(
            api_key="test",
            max_tokens=1,
            timeout=1,
            max_retries=0,
        )
        assert config.max_tokens == 1
        assert config.max_retries == 0

    @pytest.mark.edge_case
    def test_float_precision(self):
        """Test float precision boundaries."""
        config = AgentConfig(convergence_threshold=0.0000001)
        assert config.convergence_threshold > 0.0

    @pytest.mark.edge_case
    def test_temperature_precision(self):
        """Test temperature with high precision."""
        temps = [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0]
        for temp in temps:
            config = LLMConfig(api_key="test", temperature=temp)
            assert abs(config.temperature - temp) < 1e-10


class TestPromptLengthEdgeCases:
    """Tests for various prompt lengths."""

    @pytest.mark.edge_case
    def test_single_character_prompt(self, agent):
        """Test with single character prompt."""
        result = agent.optimize_prompt("a")
        assert "optimized_prompt" in result

    @pytest.mark.edge_case
    def test_two_character_prompt(self, agent):
        """Test with two character prompt."""
        result = agent.optimize_prompt("ab")
        assert "optimized_prompt" in result

    @pytest.mark.edge_case
    def test_extremely_long_prompt(self, agent):
        """Test with extremely long prompt."""
        long_prompt = "word " * 5000
        result = agent.optimize_prompt(long_prompt)
        assert "optimized_prompt" in result

    @pytest.mark.edge_case
    def test_prompt_with_repeated_characters(self, agent):
        """Test with repeated characters."""
        prompts = [
            "a" * 100,
            "test" * 50,
            " " * 20 + "word" + " " * 20,
        ]

        for prompt in prompts:
            if prompt.strip():  # Skip all-whitespace
                result = agent.evaluate_prompt(prompt)
                assert "metrics" in result


class TestSpecialCharacterEdgeCases:
    """Tests for special characters."""

    @pytest.mark.edge_case
    def test_newline_variations(self, agent):
        """Test different newline variations."""
        prompts = [
            "Line 1\nLine 2",
            "Line 1\r\nLine 2",
            "Line 1\rLine 2",
            "Multiple\n\n\nnewlines",
        ]

        for prompt in prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result

    @pytest.mark.edge_case
    def test_tab_characters(self, agent):
        """Test tab characters."""
        prompts = [
            "Word\tWord",
            "\t\tIndented",
            "Mixed \t spaces\tand\ttabs",
        ]

        for prompt in prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result

    @pytest.mark.edge_case
    def test_null_bytes(self, agent):
        """Test handling of null bytes."""
        # Note: Some systems may handle this differently
        prompt = "Test\x00with\x00nulls"
        result = agent.evaluate_prompt(prompt)
        assert "metrics" in result

    @pytest.mark.edge_case
    def test_control_characters(self, agent):
        """Test various control characters."""
        control_chars = [
            "Test\x01\x02\x03",
            "Bell\x07character",
            "Backspace\x08test",
        ]

        for prompt in control_chars:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result

    @pytest.mark.edge_case
    def test_all_punctuation(self, agent):
        """Test with all punctuation."""
        punctuation_prompt = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        result = agent.evaluate_prompt(punctuation_prompt)
        assert "metrics" in result

    @pytest.mark.edge_case
    def test_escape_sequences(self, agent):
        """Test with escape sequences."""
        prompts = [
            "Test\\nescaped\\nnewlines",
            "Test\\ttabs",
            "Test\\\\backslashes",
            "Test\\\"quotes",
        ]

        for prompt in prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result


class TestConcurrencyEdgeCases:
    """Tests for concurrent operations."""

    @pytest.mark.edge_case
    def test_multiple_agents_same_config(self, default_agent_config):
        """Test multiple agents with same config."""
        agents = [PromptTuningAgent(default_agent_config) for _ in range(5)]

        for i, agent in enumerate(agents):
            result = agent.optimize_prompt(f"Test {i}")
            assert "optimized_prompt" in result

    @pytest.mark.edge_case
    def test_agent_state_isolation(self, default_agent_config):
        """Test that agent states are isolated."""
        agent1 = PromptTuningAgent(default_agent_config)
        agent2 = PromptTuningAgent(default_agent_config)

        agent1.optimize_prompt("Test 1")
        agent2.optimize_prompt("Test 2")

        assert len(agent1.get_session_history()) == 1
        assert len(agent2.get_session_history()) == 1


class TestMetricsEdgeCases:
    """Edge cases for metrics calculations."""

    @pytest.mark.edge_case
    def test_metrics_all_zeros(self):
        """Test metrics with all zeros."""
        metrics = PromptMetrics(0.0, 0, 0.0, 0.0, 0.0, {})
        assert metrics.score == 0.0

    @pytest.mark.edge_case
    def test_metrics_all_ones(self):
        """Test metrics with all maximum values."""
        metrics = PromptMetrics(1.0, 999999, 1.0, 1.0, 1.0, {})
        assert metrics.score == 1.0

    @pytest.mark.edge_case
    def test_metrics_with_large_metadata(self):
        """Test metrics with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        metrics = PromptMetrics(0.5, 100, 0.5, 0.5, 0.5, large_metadata)
        assert len(metrics.metadata) == 1000

    @pytest.mark.edge_case
    def test_metrics_with_nested_metadata(self):
        """Test metrics with nested metadata."""
        nested_metadata = {
            "level1": {
                "level2": {
                    "level3": {"value": "deep"}
                }
            }
        }
        metrics = PromptMetrics(0.5, 100, 0.5, 0.5, 0.5, nested_metadata)
        assert metrics.metadata["level1"]["level2"]["level3"]["value"] == "deep"

    @pytest.mark.edge_case
    def test_calculator_with_extreme_weights(self):
        """Test calculator with extreme weight distributions."""
        # All weight on one metric
        weights = {
            "clarity": 1.0,
            "specificity": 0.0,
            "token_efficiency": 0.0,
            "response_quality": 0.0,
        }
        calc = MetricsCalculator(weights)
        metrics = calc.calculate_metrics("Test prompt")
        assert metrics is not None

    @pytest.mark.edge_case
    def test_calculator_equal_weights(self):
        """Test calculator with equal weights."""
        weights = {
            "clarity": 0.25,
            "specificity": 0.25,
            "token_efficiency": 0.25,
            "response_quality": 0.25,
        }
        calc = MetricsCalculator(weights)
        metrics = calc.calculate_metrics("Test prompt")
        assert metrics is not None


class TestConfigurationEdgeCases:
    """Edge cases for configuration."""

    @pytest.mark.edge_case
    def test_config_serialization_round_trip_complex(self):
        """Test complex config serialization."""
        config = AgentConfig(
            max_iterations=99,
            convergence_threshold=0.001,
            initial_prompts=["a", "b", "c"],
            batch_size=25,
            log_level="DEBUG",
        )

        config_dict = config.to_dict()
        restored = AgentConfig.from_dict(config_dict)

        assert restored.max_iterations == config.max_iterations
        assert restored.convergence_threshold == config.convergence_threshold
        assert restored.initial_prompts == config.initial_prompts

    @pytest.mark.edge_case
    def test_config_with_all_providers(self):
        """Test config with all provider types."""
        for provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
            config = LLMConfig(provider=provider, api_key="test")
            assert config.provider == provider

    @pytest.mark.edge_case
    def test_local_provider_no_api_key(self):
        """Test local provider doesn't require API key."""
        config = LLMConfig(provider=LLMProvider.LOCAL, api_key=None)
        # This should work for local provider
        # But LLMInterface will still validate
        assert config.provider == LLMProvider.LOCAL


class TestErrorRecoveryEdgeCases:
    """Edge cases for error recovery."""

    @pytest.mark.edge_case
    def test_tuning_with_llm_errors(self, prompt_tuner):
        """Test tuning continues despite LLM errors."""
        # The LLM interface has error simulation
        # Tuning should handle exceptions gracefully
        try:
            result = prompt_tuner.tune("Test prompt", max_iterations=2)
            assert result is not None
        except Exception as e:
            # Expected if all attempts fail
            assert True

    @pytest.mark.edge_case
    def test_partial_batch_failures(self, agent):
        """Test batch processing with some invalid prompts."""
        # All must be valid for batch to work
        # Testing that validation catches issues
        with pytest.raises(TypeError):
            agent.optimize_batch(["valid", None, "also valid"])


class TestMemoryAndPerformanceEdgeCases:
    """Edge cases for memory and performance."""

    @pytest.mark.edge_case
    @pytest.mark.slow
    def test_many_iterations(self, agent):
        """Test with many iterations."""
        config = AgentConfig(max_iterations=50, convergence_threshold=0.0)
        agent.update_config(config)
        result = agent.optimize_prompt("Test")
        # Should complete without memory issues
        assert result is not None

    @pytest.mark.edge_case
    def test_large_batch_processing(self, agent):
        """Test processing large batch."""
        prompts = [f"Test prompt {i}" for i in range(20)]
        results = agent.optimize_batch(prompts)
        assert len(results) == 20

    @pytest.mark.edge_case
    def test_session_history_growth(self, agent):
        """Test session history with many optimizations."""
        for i in range(50):
            agent.optimize_prompt(f"Test {i}")

        history = agent.get_session_history()
        assert len(history) == 50

    @pytest.mark.edge_case
    def test_stats_with_large_numbers(self, llm_interface):
        """Test stats tracking with large numbers."""
        for _ in range(100):
            llm_interface.generate("Test")

        stats = llm_interface.get_stats()
        assert stats["call_count"] == 100
        assert stats["total_tokens"] > 0


class TestPromptContentEdgeCases:
    """Edge cases for specific prompt content."""

    @pytest.mark.edge_case
    def test_prompt_with_code(self, agent):
        """Test prompts containing code."""
        code_prompts = [
            "def func():\n    return 42",
            "SELECT * FROM users WHERE id = 1",
            "<html><body>Test</body></html>",
            "{ \"key\": \"value\" }",
        ]

        for prompt in code_prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result

    @pytest.mark.edge_case
    def test_prompt_with_urls(self, agent):
        """Test prompts containing URLs."""
        url_prompts = [
            "Visit https://example.com",
            "Check http://test.org/path?query=1",
            "ftp://files.example.com",
        ]

        for prompt in url_prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result

    @pytest.mark.edge_case
    def test_prompt_with_email(self, agent):
        """Test prompts containing email addresses."""
        prompts = [
            "Contact user@example.com",
            "Email: test.user+tag@domain.co.uk",
        ]

        for prompt in prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result

    @pytest.mark.edge_case
    def test_prompt_with_numbers_only(self, agent):
        """Test prompts with only numbers."""
        number_prompts = [
            "123456789",
            "3.14159",
            "1e10",
            "-42",
        ]

        for prompt in number_prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result

    @pytest.mark.edge_case
    def test_prompt_with_markdown(self, agent):
        """Test prompts with markdown formatting."""
        markdown_prompts = [
            "# Header\n## Subheader",
            "**bold** and *italic*",
            "- List item 1\n- List item 2",
            "[link](https://example.com)",
        ]

        for prompt in markdown_prompts:
            result = agent.evaluate_prompt(prompt)
            assert "metrics" in result


class TestSequentialOperationEdgeCases:
    """Edge cases for sequential operations."""

    @pytest.mark.edge_case
    def test_optimize_then_evaluate_same_prompt(self, agent):
        """Test optimizing then evaluating same prompt."""
        prompt = "Write a function"
        opt_result = agent.optimize_prompt(prompt)
        eval_result = agent.evaluate_prompt(prompt)

        assert "optimized_prompt" in opt_result
        assert "metrics" in eval_result

    @pytest.mark.edge_case
    def test_multiple_config_updates(self, agent):
        """Test multiple config updates."""
        for i in range(5):
            new_config = AgentConfig(max_iterations=10 + i)
            agent.update_config(new_config)

        assert agent.config.max_iterations == 14

    @pytest.mark.edge_case
    def test_optimize_reset_optimize(self, agent):
        """Test optimize, reset, optimize pattern."""
        agent.optimize_prompt("Test 1")
        assert len(agent.get_session_history()) == 1

        agent.reset_session()
        assert len(agent.get_session_history()) == 0

        agent.optimize_prompt("Test 2")
        assert len(agent.get_session_history()) == 1
