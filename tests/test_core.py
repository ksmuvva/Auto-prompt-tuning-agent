"""Tests for core agent module."""

import pytest
from src.agent.core import PromptTuningAgent
from src.agent.config import AgentConfig


class TestPromptTuningAgent:
    """Tests for PromptTuningAgent."""

    @pytest.mark.unit
    def test_initialization_default(self):
        """Test agent initialization with defaults."""
        agent = PromptTuningAgent()
        assert agent.config is not None
        assert agent.llm is not None
        assert agent.metrics is not None
        assert agent.tuner is not None

    @pytest.mark.unit
    def test_initialization_custom_config(self, default_agent_config):
        """Test agent initialization with custom config."""
        agent = PromptTuningAgent(default_agent_config)
        assert agent.config == default_agent_config

    @pytest.mark.edge_case
    def test_initialization_invalid_config(self):
        """Test initialization with invalid config."""
        with pytest.raises(TypeError, match="must be an AgentConfig"):
            PromptTuningAgent("not a config")

    @pytest.mark.unit
    def test_optimize_prompt_basic(self, agent, sample_prompt):
        """Test basic prompt optimization."""
        result = agent.optimize_prompt(sample_prompt)
        assert "original_prompt" in result
        assert "optimized_prompt" in result
        assert "score" in result
        assert "improvement" in result
        assert "iterations" in result
        assert "history" in result

    @pytest.mark.edge_case
    def test_optimize_prompt_empty(self, agent):
        """Test optimization with empty prompt."""
        with pytest.raises(ValueError, match="non-empty string"):
            agent.optimize_prompt("")

    @pytest.mark.edge_case
    def test_optimize_prompt_whitespace(self, agent):
        """Test optimization with whitespace-only prompt."""
        with pytest.raises(ValueError, match="only whitespace"):
            agent.optimize_prompt("   ")

    @pytest.mark.edge_case
    def test_optimize_prompt_none(self, agent):
        """Test optimization with None prompt."""
        with pytest.raises(ValueError):
            agent.optimize_prompt(None)

    @pytest.mark.micro
    def test_optimize_prompt_with_expected_output(self, agent):
        """Test optimization with expected output."""
        result = agent.optimize_prompt(
            "Write hello world",
            expected_output="print('Hello, World!')"
        )
        assert result["score"] >= 0.0

    @pytest.mark.micro
    def test_optimize_prompt_updates_session_history(self, agent):
        """Test that optimization updates session history."""
        initial_len = len(agent.get_session_history())
        agent.optimize_prompt("Test prompt")
        assert len(agent.get_session_history()) == initial_len + 1

    @pytest.mark.unit
    def test_optimize_batch(self, agent, sample_prompts):
        """Test batch optimization."""
        results = agent.optimize_batch(sample_prompts)
        assert len(results) == len(sample_prompts)
        assert all("original_prompt" in r for r in results)
        assert all("optimized_prompt" in r for r in results)

    @pytest.mark.edge_case
    def test_optimize_batch_empty_list(self, agent):
        """Test batch optimization with empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            agent.optimize_batch([])

    @pytest.mark.edge_case
    def test_optimize_batch_non_list(self, agent):
        """Test batch optimization with non-list."""
        with pytest.raises(TypeError, match="must be a list"):
            agent.optimize_batch("not a list")

    @pytest.mark.edge_case
    def test_optimize_batch_mixed_types(self, agent):
        """Test batch optimization with mixed types."""
        with pytest.raises(TypeError, match="must be strings"):
            agent.optimize_batch(["valid", 123, None])

    @pytest.mark.micro
    def test_optimize_batch_with_expected_outputs(self, agent):
        """Test batch optimization with expected outputs."""
        prompts = ["Test 1", "Test 2"]
        expected = ["Output 1", "Output 2"]
        results = agent.optimize_batch(prompts, expected)
        assert len(results) == 2

    @pytest.mark.edge_case
    def test_optimize_batch_mismatched_lengths(self, agent):
        """Test batch optimization with mismatched lengths."""
        prompts = ["Test 1", "Test 2"]
        expected = ["Output 1"]
        with pytest.raises(ValueError, match="same length"):
            agent.optimize_batch(prompts, expected)

    @pytest.mark.unit
    def test_evaluate_prompt(self, agent):
        """Test prompt evaluation."""
        result = agent.evaluate_prompt("Test prompt")
        assert "prompt" in result
        assert "metrics" in result
        assert isinstance(result["metrics"], dict)

    @pytest.mark.edge_case
    def test_evaluate_prompt_empty(self, agent):
        """Test evaluation with empty prompt."""
        with pytest.raises(ValueError, match="non-empty string"):
            agent.evaluate_prompt("")

    @pytest.mark.edge_case
    def test_evaluate_prompt_none(self, agent):
        """Test evaluation with None prompt."""
        with pytest.raises(ValueError):
            agent.evaluate_prompt(None)

    @pytest.mark.unit
    def test_get_session_history(self, agent):
        """Test getting session history."""
        history = agent.get_session_history()
        assert isinstance(history, list)

        # Optimize and check history updates
        agent.optimize_prompt("Test")
        new_history = agent.get_session_history()
        assert len(new_history) > len(history)

    @pytest.mark.micro
    def test_get_session_history_returns_copy(self, agent):
        """Test that session history returns a copy."""
        agent.optimize_prompt("Test")
        history = agent.get_session_history()
        history.append("modified")
        assert len(agent.get_session_history()) != len(history)

    @pytest.mark.unit
    def test_reset_session(self, agent):
        """Test resetting session."""
        agent.optimize_prompt("Test")
        assert len(agent.get_session_history()) > 0

        agent.reset_session()
        assert len(agent.get_session_history()) == 0

    @pytest.mark.micro
    def test_reset_session_resets_llm_stats(self, agent):
        """Test that reset_session also resets LLM stats."""
        agent.optimize_prompt("Test")
        stats_before = agent.get_stats()["llm_stats"]["call_count"]
        assert stats_before > 0

        agent.reset_session()
        stats_after = agent.get_stats()["llm_stats"]["call_count"]
        assert stats_after == 0

    @pytest.mark.unit
    def test_get_stats(self, agent):
        """Test getting agent statistics."""
        stats = agent.get_stats()
        assert "llm_stats" in stats
        assert "session_optimizations" in stats
        assert "config" in stats

    @pytest.mark.micro
    def test_get_stats_tracks_optimizations(self, agent):
        """Test that stats track optimization count."""
        initial_stats = agent.get_stats()
        assert initial_stats["session_optimizations"] == 0

        agent.optimize_prompt("Test 1")
        agent.optimize_prompt("Test 2")

        final_stats = agent.get_stats()
        assert final_stats["session_optimizations"] == 2

    @pytest.mark.unit
    def test_update_config(self, agent):
        """Test updating agent configuration."""
        new_config = AgentConfig(max_iterations=20)
        agent.update_config(new_config)
        assert agent.config.max_iterations == 20

    @pytest.mark.edge_case
    def test_update_config_invalid_type(self, agent):
        """Test updating with invalid config type."""
        with pytest.raises(TypeError, match="must be an AgentConfig"):
            agent.update_config("not a config")

    @pytest.mark.micro
    def test_update_config_reinitializes_components(self, agent):
        """Test that config update reinitializes components."""
        old_llm = agent.llm
        new_config = AgentConfig(max_iterations=20)
        agent.update_config(new_config)
        # Components should be recreated
        assert agent.llm is not old_llm

    @pytest.mark.edge_case
    def test_multiple_optimizations_accumulate_history(self, agent):
        """Test that multiple optimizations accumulate in history."""
        agent.optimize_prompt("Test 1")
        agent.optimize_prompt("Test 2")
        agent.optimize_prompt("Test 3")

        history = agent.get_session_history()
        assert len(history) == 3

    @pytest.mark.edge_case
    def test_very_long_prompt_optimization(self, agent):
        """Test optimizing very long prompt."""
        long_prompt = "Write a detailed function " * 100
        result = agent.optimize_prompt(long_prompt)
        assert "optimized_prompt" in result

    @pytest.mark.micro
    def test_optimization_result_structure(self, agent):
        """Test that optimization result has correct structure."""
        result = agent.optimize_prompt("Test")

        # Check all required keys
        assert "original_prompt" in result
        assert "optimized_prompt" in result
        assert "score" in result
        assert "improvement" in result
        assert "iterations" in result
        assert "history" in result

        # Check types
        assert isinstance(result["original_prompt"], str)
        assert isinstance(result["optimized_prompt"], str)
        assert isinstance(result["score"], (int, float))
        assert isinstance(result["improvement"], (int, float))
        assert isinstance(result["iterations"], int)
        assert isinstance(result["history"], list)

    @pytest.mark.micro
    def test_evaluation_result_structure(self, agent):
        """Test that evaluation result has correct structure."""
        result = agent.evaluate_prompt("Test")

        assert "prompt" in result
        assert "metrics" in result
        assert isinstance(result["prompt"], str)
        assert isinstance(result["metrics"], dict)

    @pytest.mark.edge_case
    def test_optimization_with_special_characters(self, agent):
        """Test optimization with special characters."""
        special_prompts = [
            "Explain émojis 🎉",
            "Japanese: こんにちは",
            "Symbols: @#$%^&*()",
            "Multi\nline\nprompt",
        ]

        for prompt in special_prompts:
            result = agent.optimize_prompt(prompt)
            assert "optimized_prompt" in result

    @pytest.mark.integration
    def test_full_workflow(self, agent):
        """Test complete agent workflow."""
        # Evaluate initial prompt
        eval_result = agent.evaluate_prompt("Write a sorting function")
        assert "metrics" in eval_result

        # Optimize the prompt
        opt_result = agent.optimize_prompt("Write a sorting function")
        assert opt_result["score"] >= 0.0

        # Check stats
        stats = agent.get_stats()
        assert stats["session_optimizations"] == 1
        assert stats["llm_stats"]["call_count"] > 0

        # Reset and verify
        agent.reset_session()
        assert len(agent.get_session_history()) == 0

    @pytest.mark.edge_case
    def test_batch_optimization_single_item(self, agent):
        """Test batch optimization with single item."""
        results = agent.optimize_batch(["Single prompt"])
        assert len(results) == 1

    @pytest.mark.edge_case
    def test_optimization_improvement_calculation(self, agent):
        """Test that improvement is calculated correctly."""
        result = agent.optimize_prompt("Test prompt")

        # Improvement should be score difference
        if result["history"]:
            initial_score = result["history"][0]["score"]
            expected_improvement = result["score"] - initial_score
            assert abs(result["improvement"] - expected_improvement) < 0.001
