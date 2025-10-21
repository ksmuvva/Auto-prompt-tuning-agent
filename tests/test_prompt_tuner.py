"""Tests for prompt tuner module."""

import pytest
from src.agent.prompt_tuner import PromptTuner
from src.agent.metrics import PromptMetrics


class TestPromptTuner:
    """Tests for PromptTuner."""

    @pytest.mark.unit
    def test_initialization(self, llm_interface, metrics_calculator, default_agent_config):
        """Test prompt tuner initialization."""
        tuner = PromptTuner(llm_interface, metrics_calculator, default_agent_config)
        assert tuner.llm == llm_interface
        assert tuner.metrics == metrics_calculator
        assert tuner.config == default_agent_config
        assert tuner.history == []
        assert tuner.best_prompt is None
        assert tuner.best_score == 0.0

    @pytest.mark.edge_case
    def test_initialization_invalid_llm(self, metrics_calculator, default_agent_config):
        """Test initialization with invalid LLM interface."""
        with pytest.raises(TypeError, match="must be an LLMInterface"):
            PromptTuner("not llm", metrics_calculator, default_agent_config)

    @pytest.mark.edge_case
    def test_initialization_invalid_metrics(self, llm_interface, default_agent_config):
        """Test initialization with invalid metrics calculator."""
        with pytest.raises(TypeError, match="must be a MetricsCalculator"):
            PromptTuner(llm_interface, "not metrics", default_agent_config)

    @pytest.mark.edge_case
    def test_initialization_invalid_config(self, llm_interface, metrics_calculator):
        """Test initialization with invalid config."""
        with pytest.raises(TypeError, match="must be an AgentConfig"):
            PromptTuner(llm_interface, metrics_calculator, "not config")

    @pytest.mark.unit
    def test_tune_basic(self, prompt_tuner):
        """Test basic prompt tuning."""
        best_prompt, best_score = prompt_tuner.tune("Write a function")
        assert isinstance(best_prompt, str)
        assert len(best_prompt) > 0
        assert 0.0 <= best_score <= 1.0

    @pytest.mark.edge_case
    def test_tune_empty_prompt(self, prompt_tuner):
        """Test tuning with empty prompt."""
        with pytest.raises(ValueError, match="non-empty string"):
            prompt_tuner.tune("")

    @pytest.mark.edge_case
    def test_tune_whitespace_prompt(self, prompt_tuner):
        """Test tuning with whitespace-only prompt."""
        with pytest.raises(ValueError, match="only whitespace"):
            prompt_tuner.tune("   ")

    @pytest.mark.edge_case
    def test_tune_none_prompt(self, prompt_tuner):
        """Test tuning with None prompt."""
        with pytest.raises(ValueError):
            prompt_tuner.tune(None)

    @pytest.mark.micro
    def test_tune_with_expected_output(self, prompt_tuner):
        """Test tuning with expected output."""
        best_prompt, best_score = prompt_tuner.tune(
            "Write hello world",
            expected_output="print('Hello, World!')"
        )
        assert isinstance(best_prompt, str)
        assert best_score >= 0.0

    @pytest.mark.micro
    def test_tune_custom_iterations(self, prompt_tuner):
        """Test tuning with custom iteration count."""
        best_prompt, best_score = prompt_tuner.tune(
            "Write a function",
            max_iterations=5
        )
        assert prompt_tuner.get_iteration_count() <= 5

    @pytest.mark.edge_case
    def test_tune_zero_iterations(self, prompt_tuner):
        """Test tuning with zero iterations."""
        with pytest.raises(ValueError, match="must be positive"):
            prompt_tuner.tune("Write a function", max_iterations=0)

    @pytest.mark.edge_case
    def test_tune_negative_iterations(self, prompt_tuner):
        """Test tuning with negative iterations."""
        with pytest.raises(ValueError, match="must be positive"):
            prompt_tuner.tune("Write a function", max_iterations=-1)

    @pytest.mark.micro
    def test_tune_single_iteration(self, prompt_tuner):
        """Test tuning with single iteration."""
        best_prompt, best_score = prompt_tuner.tune("Test", max_iterations=1)
        assert prompt_tuner.get_iteration_count() == 1

    @pytest.mark.unit
    def test_tune_updates_history(self, prompt_tuner):
        """Test that tuning updates history."""
        prompt_tuner.tune("Write a function")
        history = prompt_tuner.get_history()
        assert len(history) > 0
        assert all("iteration" in h for h in history)
        assert all("prompt" in h for h in history)
        assert all("score" in h for h in history)

    @pytest.mark.micro
    def test_tune_updates_best_prompt(self, prompt_tuner):
        """Test that tuning updates best prompt."""
        prompt_tuner.tune("Write a function")
        assert prompt_tuner.get_best_prompt() is not None
        assert isinstance(prompt_tuner.get_best_prompt(), str)

    @pytest.mark.unit
    def test_tune_batch(self, prompt_tuner, sample_prompts):
        """Test batch tuning."""
        results = prompt_tuner.tune_batch(sample_prompts)
        assert len(results) == len(sample_prompts)
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    @pytest.mark.edge_case
    def test_tune_batch_empty_list(self, prompt_tuner):
        """Test batch tuning with empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            prompt_tuner.tune_batch([])

    @pytest.mark.edge_case
    def test_tune_batch_non_list(self, prompt_tuner):
        """Test batch tuning with non-list input."""
        with pytest.raises(TypeError, match="must be a list"):
            prompt_tuner.tune_batch("not a list")

    @pytest.mark.edge_case
    def test_tune_batch_mixed_types(self, prompt_tuner):
        """Test batch tuning with mixed types."""
        with pytest.raises(TypeError, match="must be strings"):
            prompt_tuner.tune_batch(["valid", 123, "also valid"])

    @pytest.mark.micro
    def test_tune_batch_with_expected_outputs(self, prompt_tuner):
        """Test batch tuning with expected outputs."""
        prompts = ["Test 1", "Test 2"]
        expected = ["Output 1", "Output 2"]
        results = prompt_tuner.tune_batch(prompts, expected)
        assert len(results) == 2

    @pytest.mark.edge_case
    def test_tune_batch_mismatched_lengths(self, prompt_tuner):
        """Test batch tuning with mismatched list lengths."""
        prompts = ["Test 1", "Test 2"]
        expected = ["Output 1"]  # Different length
        with pytest.raises(ValueError, match="same length"):
            prompt_tuner.tune_batch(prompts, expected)

    @pytest.mark.micro
    def test_reset_state(self, prompt_tuner):
        """Test state reset."""
        prompt_tuner.tune("Test")
        prompt_tuner._reset_state()
        assert prompt_tuner.history == []
        assert prompt_tuner.best_prompt is None
        assert prompt_tuner.best_score == 0.0
        assert prompt_tuner._iteration_count == 0

    @pytest.mark.micro
    def test_check_convergence_early_iterations(self, prompt_tuner):
        """Test convergence check in early iterations."""
        assert prompt_tuner._check_convergence(0) is False
        assert prompt_tuner._check_convergence(1) is False

    @pytest.mark.micro
    def test_check_convergence_with_history(self, prompt_tuner):
        """Test convergence check with history."""
        # Add similar scores to history
        prompt_tuner.history = [
            {"score": 0.70},
            {"score": 0.701},
            {"score": 0.702},
        ]
        # Should converge since range < threshold (0.01)
        assert prompt_tuner._check_convergence(2) is True

    @pytest.mark.micro
    def test_check_convergence_no_convergence(self, prompt_tuner):
        """Test no convergence with varying scores."""
        prompt_tuner.history = [
            {"score": 0.5},
            {"score": 0.7},
            {"score": 0.9},
        ]
        # Should not converge since range > threshold
        assert prompt_tuner._check_convergence(2) is False

    @pytest.mark.micro
    def test_generate_variation_low_specificity(self, prompt_tuner):
        """Test variation generation for low specificity."""
        metrics = PromptMetrics(0.5, 10, 0.8, 0.3, 0.8, {})  # Low specificity
        variation = prompt_tuner._generate_variation("Test prompt", metrics)
        assert isinstance(variation, str)
        assert len(variation) > len("Test prompt")

    @pytest.mark.micro
    def test_generate_variation_low_clarity(self, prompt_tuner):
        """Test variation generation for low clarity."""
        metrics = PromptMetrics(0.5, 10, 0.3, 0.8, 0.8, {})  # Low clarity
        variation = prompt_tuner._generate_variation("Test prompt", metrics)
        assert isinstance(variation, str)

    @pytest.mark.micro
    def test_generate_variation_low_efficiency(self, prompt_tuner):
        """Test variation generation for low token efficiency."""
        long_prompt = " ".join(["word"] * 100)
        metrics = PromptMetrics(0.5, 500, 0.8, 0.8, 0.3, {})  # Low efficiency
        variation = prompt_tuner._generate_variation(long_prompt, metrics)
        assert isinstance(variation, str)

    @pytest.mark.edge_case
    def test_generate_variation_empty_prompt(self, prompt_tuner):
        """Test variation generation with empty prompt."""
        metrics = PromptMetrics(0.5, 0, 0.5, 0.5, 0.5, {})
        variation = prompt_tuner._generate_variation("", metrics)
        assert variation == ""

    @pytest.mark.unit
    def test_get_history(self, prompt_tuner):
        """Test getting history."""
        prompt_tuner.tune("Test prompt")
        history = prompt_tuner.get_history()
        assert isinstance(history, list)
        # Should return a copy
        history.append("modified")
        assert len(prompt_tuner.get_history()) != len(history)

    @pytest.mark.unit
    def test_get_best_prompt_before_tuning(self, prompt_tuner):
        """Test getting best prompt before any tuning."""
        assert prompt_tuner.get_best_prompt() is None

    @pytest.mark.unit
    def test_get_iteration_count(self, prompt_tuner):
        """Test getting iteration count."""
        assert prompt_tuner.get_iteration_count() == 0
        prompt_tuner.tune("Test", max_iterations=3)
        assert prompt_tuner.get_iteration_count() > 0

    @pytest.mark.unit
    def test_evaluate_prompt(self, prompt_tuner):
        """Test evaluating a single prompt."""
        metrics = prompt_tuner.evaluate_prompt("Test prompt")
        assert isinstance(metrics, PromptMetrics)
        assert 0.0 <= metrics.score <= 1.0

    @pytest.mark.edge_case
    def test_evaluate_prompt_empty(self, prompt_tuner):
        """Test evaluating empty prompt."""
        with pytest.raises(ValueError, match="non-empty string"):
            prompt_tuner.evaluate_prompt("")

    @pytest.mark.micro
    def test_evaluate_prompt_with_expected(self, prompt_tuner):
        """Test evaluating with expected output."""
        metrics = prompt_tuner.evaluate_prompt("Test", "Expected")
        assert isinstance(metrics, PromptMetrics)

    @pytest.mark.edge_case
    def test_multiple_tune_calls_reset_state(self, prompt_tuner):
        """Test that multiple tune calls properly reset state."""
        prompt_tuner.tune("First prompt")
        first_history_len = len(prompt_tuner.get_history())

        prompt_tuner.tune("Second prompt")
        second_history_len = len(prompt_tuner.get_history())

        # Each tune should start fresh
        assert second_history_len > 0

    @pytest.mark.edge_case
    def test_very_long_prompt_tuning(self, prompt_tuner):
        """Test tuning with very long prompt."""
        long_prompt = "Write a function " * 100
        best_prompt, best_score = prompt_tuner.tune(long_prompt, max_iterations=2)
        assert isinstance(best_prompt, str)

    @pytest.mark.micro
    def test_tuning_improves_score(self, prompt_tuner):
        """Test that tuning attempts to improve score."""
        prompt_tuner.tune("Simple prompt", max_iterations=5)
        history = prompt_tuner.get_history()
        # Best score should be >= first score
        if len(history) > 0:
            first_score = history[0]["score"]
            assert prompt_tuner.best_score >= first_score
