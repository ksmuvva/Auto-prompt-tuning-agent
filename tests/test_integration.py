"""Integration tests for the AI agent."""

import pytest
from src.agent.core import PromptTuningAgent
from src.agent.config import AgentConfig, LLMConfig, LLMProvider


class TestEndToEndWorkflows:
    """End-to-end integration tests."""

    @pytest.mark.integration
    def test_complete_optimization_workflow(self):
        """Test complete optimization workflow from start to finish."""
        # Create agent
        agent = PromptTuningAgent()

        # Evaluate initial prompt
        initial_eval = agent.evaluate_prompt("Write a sorting algorithm")
        assert "metrics" in initial_eval
        initial_score = initial_eval["metrics"]["score"]

        # Optimize the prompt
        opt_result = agent.optimize_prompt("Write a sorting algorithm")
        assert opt_result["score"] >= 0.0

        # Verify history was recorded
        history = agent.get_session_history()
        assert len(history) == 1

        # Get stats
        stats = agent.get_stats()
        assert stats["session_optimizations"] == 1
        assert stats["llm_stats"]["call_count"] > 0

    @pytest.mark.integration
    def test_batch_optimization_workflow(self):
        """Test batch optimization workflow."""
        agent = PromptTuningAgent()

        prompts = [
            "Explain machine learning",
            "Write a fibonacci function",
            "Describe quantum computing",
        ]

        # Batch optimize
        results = agent.optimize_batch(prompts)
        assert len(results) == 3

        # Verify each result
        for i, result in enumerate(results):
            assert result["original_prompt"] == prompts[i]
            assert "optimized_prompt" in result
            assert result["score"] >= 0.0

        # Check session history
        history = agent.get_session_history()
        assert len(history) == 3

    @pytest.mark.integration
    def test_config_update_workflow(self):
        """Test configuration update workflow."""
        agent = PromptTuningAgent()

        # Optimize with default config
        result1 = agent.optimize_prompt("Test prompt 1")
        iter1 = result1["iterations"]

        # Update config to use fewer iterations
        new_config = AgentConfig(max_iterations=3)
        agent.update_config(new_config)

        # Optimize with new config
        agent.reset_session()  # Clear previous history
        result2 = agent.optimize_prompt("Test prompt 2")
        iter2 = result2["iterations"]

        # Second optimization should use fewer iterations (or equal)
        assert iter2 <= 3

    @pytest.mark.integration
    def test_session_management_workflow(self):
        """Test session management workflow."""
        agent = PromptTuningAgent()

        # First session
        agent.optimize_prompt("Test 1")
        agent.optimize_prompt("Test 2")
        assert len(agent.get_session_history()) == 2

        # Reset session
        agent.reset_session()
        assert len(agent.get_session_history()) == 0

        # Second session
        agent.optimize_prompt("Test 3")
        assert len(agent.get_session_history()) == 1


class TestComponentIntegration:
    """Tests for component integration."""

    @pytest.mark.integration
    def test_llm_metrics_integration(self, llm_interface, metrics_calculator):
        """Test LLM and metrics integration."""
        # Generate response
        response = llm_interface.generate("Test prompt")

        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(
            "Test prompt",
            response=response
        )

        assert metrics.score >= 0.0
        assert metrics.length > 0

    @pytest.mark.integration
    def test_tuner_components_integration(self, prompt_tuner):
        """Test prompt tuner with all components."""
        # Tune a prompt
        best_prompt, best_score = prompt_tuner.tune("Write code", max_iterations=3)

        # Verify history was created
        history = prompt_tuner.get_history()
        assert len(history) > 0

        # Verify best prompt was selected
        assert prompt_tuner.get_best_prompt() == best_prompt

        # Verify LLM was called
        stats = prompt_tuner.llm.get_stats()
        assert stats["call_count"] > 0

    @pytest.mark.integration
    def test_full_agent_stack(self):
        """Test full agent stack integration."""
        # Create custom config
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            api_key="test-key",
            temperature=0.8,
            max_tokens=500,
        )

        agent_config = AgentConfig(
            llm_config=llm_config,
            max_iterations=5,
            convergence_threshold=0.02,
        )

        # Create agent with custom config
        agent = PromptTuningAgent(agent_config)

        # Perform optimization
        result = agent.optimize_prompt("Write a test function")

        # Verify all components worked together
        assert result["iterations"] <= 5
        assert len(result["history"]) > 0
        assert result["score"] >= 0.0


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.mark.integration
    def test_invalid_prompt_handling(self, agent):
        """Test handling of invalid prompts across the system."""
        invalid_prompts = ["", "   ", None]

        for prompt in invalid_prompts:
            with pytest.raises(ValueError):
                agent.optimize_prompt(prompt)

    @pytest.mark.integration
    def test_config_validation_chain(self):
        """Test that config validation works through the chain."""
        # Invalid LLM config should be caught
        with pytest.raises(ValueError):
            llm_config = LLMConfig(temperature=3.0)  # Too high

        # Invalid agent config should be caught
        with pytest.raises(ValueError):
            agent_config = AgentConfig(max_iterations=0)  # Too low

    @pytest.mark.integration
    def test_type_validation_chain(self, agent):
        """Test type validation through the system."""
        # Invalid types should be caught at various levels
        with pytest.raises(TypeError):
            agent.optimize_batch("not a list")

        with pytest.raises(TypeError):
            agent.update_config("not a config")


class TestDataFlowIntegration:
    """Tests for data flow through the system."""

    @pytest.mark.integration
    def test_prompt_to_metrics_flow(self, agent):
        """Test data flow from prompt to metrics."""
        prompt = "Write a hello world program"

        # Evaluate
        result = agent.evaluate_prompt(prompt)

        # Check that prompt flowed through to metrics
        assert result["prompt"] == prompt
        assert "metrics" in result
        assert "score" in result["metrics"]
        assert "length" in result["metrics"]

    @pytest.mark.integration
    def test_optimization_history_flow(self, agent):
        """Test that optimization data flows to history."""
        prompt = "Test prompt"
        result = agent.optimize_prompt(prompt)

        # Check result contains history
        assert "history" in result
        assert len(result["history"]) > 0

        # Check history entries have required data
        for entry in result["history"]:
            assert "iteration" in entry
            assert "prompt" in entry
            assert "score" in entry
            assert "metrics" in entry

    @pytest.mark.integration
    def test_stats_aggregation_flow(self, agent):
        """Test that stats aggregate properly."""
        # Perform multiple operations
        agent.optimize_prompt("Test 1")
        agent.optimize_prompt("Test 2")
        agent.evaluate_prompt("Test 3")

        # Get stats
        stats = agent.get_stats()

        # Verify aggregation
        assert stats["session_optimizations"] == 2
        assert stats["llm_stats"]["call_count"] > 0


class TestConcurrentOperationsIntegration:
    """Integration tests for concurrent operations."""

    @pytest.mark.integration
    def test_multiple_agents_independent(self):
        """Test multiple agents operate independently."""
        agent1 = PromptTuningAgent()
        agent2 = PromptTuningAgent()

        # Perform different operations
        agent1.optimize_prompt("Agent 1 prompt")
        agent2.optimize_prompt("Agent 2 prompt")

        # Verify independence
        history1 = agent1.get_session_history()
        history2 = agent2.get_session_history()

        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0]["original_prompt"] != history2[0]["original_prompt"]

    @pytest.mark.integration
    def test_sequential_batch_operations(self, agent):
        """Test sequential batch operations."""
        batch1 = ["Test 1", "Test 2"]
        batch2 = ["Test 3", "Test 4"]

        results1 = agent.optimize_batch(batch1)
        results2 = agent.optimize_batch(batch2)

        assert len(results1) == 2
        assert len(results2) == 2

        # Total history should include all
        history = agent.get_session_history()
        assert len(history) == 4


class TestConfigurationPersistence:
    """Tests for configuration persistence."""

    @pytest.mark.integration
    def test_config_serialization_integration(self):
        """Test config serialization in real workflow."""
        # Create agent with custom config
        original_config = AgentConfig(
            max_iterations=15,
            convergence_threshold=0.03,
        )
        agent = PromptTuningAgent(original_config)

        # Perform operation
        agent.optimize_prompt("Test")

        # Get config dict
        config_dict = agent.config.to_dict()

        # Create new agent from dict
        restored_config = AgentConfig.from_dict(config_dict)
        new_agent = PromptTuningAgent(restored_config)

        # Verify configs match
        assert new_agent.config.max_iterations == original_config.max_iterations
        assert new_agent.config.convergence_threshold == original_config.convergence_threshold


class TestRealWorldScenarios:
    """Tests simulating real-world usage scenarios."""

    @pytest.mark.integration
    def test_iterative_refinement_scenario(self, agent):
        """Test iterative prompt refinement scenario."""
        # Start with rough prompt
        rough_prompt = "code"

        # Evaluate it
        eval1 = agent.evaluate_prompt(rough_prompt)
        score1 = eval1["metrics"]["score"]

        # Refine manually
        better_prompt = "Write code for sorting"

        # Evaluate again
        eval2 = agent.evaluate_prompt(better_prompt)
        score2 = eval2["metrics"]["score"]

        # Better prompt should score higher
        assert score2 >= score1

        # Now optimize the better prompt
        result = agent.optimize_prompt(better_prompt)
        assert result["score"] >= score2

    @pytest.mark.integration
    def test_batch_processing_scenario(self, agent):
        """Test batch processing scenario."""
        # Simulate processing multiple user prompts
        user_prompts = [
            "How do I sort a list?",
            "Explain recursion",
            "What is a hash table?",
            "Write a binary search",
        ]

        # Process all at once
        results = agent.optimize_batch(user_prompts)

        # All should succeed
        assert len(results) == len(user_prompts)
        assert all(r["score"] >= 0.0 for r in results)

        # Stats should reflect all operations
        stats = agent.get_stats()
        assert stats["session_optimizations"] == len(user_prompts)

    @pytest.mark.integration
    def test_configuration_tuning_scenario(self, agent):
        """Test scenario of tuning agent configuration."""
        prompt = "Write a function"

        # Try with aggressive convergence
        config1 = AgentConfig(
            max_iterations=20,
            convergence_threshold=0.001,
        )
        agent.update_config(config1)
        result1 = agent.optimize_prompt(prompt)

        # Reset and try with quick convergence
        agent.reset_session()
        config2 = AgentConfig(
            max_iterations=5,
            convergence_threshold=0.1,
        )
        agent.update_config(config2)
        result2 = agent.optimize_prompt(prompt)

        # Both should complete
        assert result1["iterations"] > 0
        assert result2["iterations"] > 0
        assert result2["iterations"] <= 5

    @pytest.mark.integration
    def test_monitoring_scenario(self, agent):
        """Test monitoring and analytics scenario."""
        # Perform various operations
        for i in range(5):
            agent.optimize_prompt(f"Test prompt {i}")

        # Check monitoring data
        history = agent.get_session_history()
        stats = agent.get_stats()

        # Should have complete audit trail
        assert len(history) == 5
        assert stats["session_optimizations"] == 5

        # Each history entry should have full data
        for entry in history:
            assert "original_prompt" in entry
            assert "optimized_prompt" in entry
            assert "score" in entry
            assert "iterations" in entry
            assert "history" in entry

    @pytest.mark.integration
    def test_error_recovery_scenario(self, agent):
        """Test error recovery scenario."""
        # Try invalid operation
        try:
            agent.optimize_prompt("")
        except ValueError:
            pass  # Expected

        # Agent should still work
        result = agent.optimize_prompt("Valid prompt")
        assert "optimized_prompt" in result

        # Stats should still be accurate
        stats = agent.get_stats()
        assert stats["session_optimizations"] == 1
