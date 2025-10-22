"""
Test CLI Interactive Mode and AI Agent Capabilities
Tests user input handling, AI reasoning, and autonomous behavior
"""

import pytest
from io import StringIO
import sys
from unittest.mock import patch
from agent.cli import AgentCLI
from agent.core import PromptTuningAgent


class TestCLIInteractive:
    """Test CLI with simulated user input"""

    def test_cli_handles_help_command(self):
        """Test that CLI responds to help command"""
        cli = AgentCLI()
        cli.initialize_agent("mock")

        # Simulate help command
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = cli.handle_command("help")
            output = fake_out.getvalue()

        assert "AVAILABLE COMMANDS" in output
        assert result == True

    def test_cli_handles_status_command(self):
        """Test that CLI responds to status command"""
        cli = AgentCLI()
        cli.initialize_agent("mock")

        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = cli.handle_command("status")
            output = fake_out.getvalue()

        assert "Agent Status" in output or "STATUS" in output
        assert result == True

    def test_cli_handles_config_command(self):
        """Test that CLI responds to config command"""
        cli = AgentCLI()
        cli.initialize_agent("mock")

        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = cli.handle_command("config")
            output = fake_out.getvalue()

        # Config outputs JSON, check for config keys
        assert "agent" in output or "llm" in output or "data" in output
        assert result == True

    def test_cli_handles_quit_command(self):
        """Test that CLI handles quit command"""
        cli = AgentCLI()
        result = cli.handle_command("quit")
        assert result == False  # Should return False to exit

    def test_cli_handles_exit_command(self):
        """Test that CLI handles exit command"""
        cli = AgentCLI()
        result = cli.handle_command("exit")
        assert result == False  # Should return False to exit

    def test_cli_handles_unknown_command(self):
        """Test that CLI handles unknown commands gracefully"""
        cli = AgentCLI()
        cli.initialize_agent("mock")

        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = cli.handle_command("unknown_command_xyz")
            output = fake_out.getvalue()

        assert "Unknown command" in output or "not found" in output.lower()
        assert result == True  # Should continue running

    def test_cli_handles_list_prompts_command(self):
        """Test that CLI can list prompts"""
        cli = AgentCLI()
        cli.initialize_agent("mock")

        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = cli.handle_command("list-prompts")
            output = fake_out.getvalue()

        assert "PROMPTS" in output or "prompt" in output.lower()
        assert result == True

    def test_cli_handles_ai_ask_command(self):
        """Test that CLI can handle AI 'ask' command"""
        cli = AgentCLI()
        cli.initialize_agent("mock")

        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = cli.handle_command("ask How can I improve my prompts?")
            output = fake_out.getvalue()

        # Should show agent thinking and response
        assert "Agent" in output or "thinking" in output.lower()
        assert result == True

    def test_cli_handles_ai_think_command(self):
        """Test that CLI can handle AI 'think' command"""
        cli = AgentCLI()
        cli.initialize_agent("mock")

        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = cli.handle_command("think What is the best prompt strategy?")
            output = fake_out.getvalue()

        # Should show agent thinking and response
        assert "Agent" in output or "thinking" in output.lower()
        assert result == True

    def test_cli_handles_reset_command(self):
        """Test that CLI can reset agent state"""
        cli = AgentCLI()
        cli.initialize_agent("mock")

        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = cli.handle_command("reset")
            output = fake_out.getvalue()

        assert "reset" in output.lower()
        assert result == True


class TestAIAgentCapabilities:
    """Test that the agent is truly an AI agent with autonomous capabilities"""

    def test_agent_has_memory_system(self):
        """Verify agent has memory for learning and context"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Agent should have memory
        assert hasattr(agent, 'memory')
        assert hasattr(agent.memory, 'short_term')
        assert hasattr(agent.memory, 'long_term')
        assert hasattr(agent.memory, 'learned_patterns')

    def test_agent_can_store_and_recall_knowledge(self):
        """Verify agent can learn and remember"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Store knowledge
        agent.memory.store_knowledge('test_key', 'test_value')

        # Recall knowledge
        value = agent.memory.recall('test_key')
        assert value == 'test_value'

    def test_agent_can_learn_patterns(self):
        """Verify agent can learn patterns over time"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Learn a pattern
        pattern = {'type': 'prompt_improvement', 'strategy': 'more_specific'}
        agent.memory.learn_pattern(pattern)

        # Verify pattern was learned
        assert len(agent.memory.learned_patterns) > 0
        assert agent.memory.learned_patterns[-1]['pattern'] == pattern

    def test_agent_has_thinking_capability(self):
        """Verify agent can reason about queries (AI reasoning)"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Agent should be able to think
        response = agent.think("How can I improve prompt performance?")

        # Should return a response
        assert isinstance(response, str)
        assert len(response) > 0

    def test_agent_can_provide_recommendations(self):
        """Verify agent can autonomously provide recommendations"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Get recommendations
        recommendations = agent.get_recommendations()

        # Should return recommendations dict
        assert isinstance(recommendations, dict)

    def test_agent_maintains_state(self):
        """Verify agent maintains internal state"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Should have state
        assert hasattr(agent, 'state')
        assert 'initialized' in agent.state
        assert agent.state['initialized'] == True

    def test_agent_has_autonomous_tuning(self):
        """Verify agent can autonomously tune prompts"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Should have prompt tuner for autonomous optimization
        assert hasattr(agent, 'prompt_tuner')
        assert hasattr(agent.prompt_tuner, 'run_adaptive_tuning')

    def test_agent_integrates_llm_service(self):
        """Verify agent uses LLM for intelligent operations"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Should have LLM service
        assert hasattr(agent, 'llm_service')
        assert hasattr(agent.llm_service, 'generate')

        # Test LLM generation
        result = agent.llm_service.generate("Test prompt")
        assert result.get('success') == True

    def test_agent_tracks_best_results(self):
        """Verify agent learns from results"""
        agent = PromptTuningAgent(llm_provider="mock")

        # Agent should track best results
        assert hasattr(agent, 'best_results')
        assert hasattr(agent.prompt_tuner, 'best_prompt')
        assert hasattr(agent.prompt_tuner, 'best_score')


class TestCLIUserInputPrompt:
    """Test that CLI has proper user input prompt"""

    def test_cli_uses_agent_prompt(self):
        """Verify CLI uses 'agent>' as prompt"""
        # This is verified by reading the CLI source code
        # The interactive mode uses input("agent> ")
        from agent.cli import AgentCLI
        import inspect

        # Get the run_interactive method source
        source = inspect.getsource(AgentCLI.run_interactive)

        # Verify it uses "agent>" prompt
        assert "agent>" in source

    def test_cli_initialization_message(self):
        """Verify CLI shows initialization message"""
        cli = AgentCLI()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            with patch('builtins.input', side_effect=EOFError):
                try:
                    cli.run_interactive()
                except (EOFError, StopIteration):
                    pass

            output = fake_out.getvalue()

        # Should show banner and initialization
        assert "PROMPT TUNING" in output or "Agent" in output


class TestAdvancedFeatureIntegration:
    """Test that all advanced features work together"""

    def test_agent_loads_all_advanced_modules(self):
        """Verify agent can import and use all advanced features"""
        # Try importing all advanced modules
        from agent.multi_objective import MultiObjectiveOptimizer
        from agent.ab_testing import ABTest
        from agent.monitoring import MetricsCollector
        from agent.distributed_testing import DistributedTestExecutor
        from agent.neural_optimization import GeneticPromptOptimizer
        from agent.prompt_formats import PromptFormatConverter

        # All imports should succeed
        assert MultiObjectiveOptimizer is not None
        assert ABTest is not None
        assert MetricsCollector is not None
        assert DistributedTestExecutor is not None
        assert GeneticPromptOptimizer is not None
        assert PromptFormatConverter is not None

    def test_prompt_format_converter_works(self):
        """Test prompt format conversion feature"""
        from agent.prompt_formats import PromptFormatConverter, PromptFormat

        converter = PromptFormatConverter()

        # Test conversion
        plain_text = "Analyze the following data"
        json_text = converter.convert(plain_text, PromptFormat.PLAIN, PromptFormat.JSON)

        assert isinstance(json_text, str)
        assert len(json_text) > 0

    def test_metrics_collector_works(self):
        """Test real-time monitoring feature"""
        from agent.monitoring import MetricsCollector

        collector = MetricsCollector()

        # Record metrics
        collector.record_event("test_metric", 0.95)
        collector.record_event("test_metric", 0.87)

        # Verify events were recorded
        assert len(collector.events) == 2

    def test_distributed_executor_initializes(self):
        """Test distributed testing feature initializes"""
        from agent.distributed_testing import DistributedTestExecutor

        executor = DistributedTestExecutor(num_workers=2)

        # Should initialize successfully
        assert executor.num_workers == 2

    def test_llm_service_supports_multiple_providers(self):
        """Test that LLM service supports new providers"""
        from agent.llm_service import LLMService

        # Test with mock provider
        service = LLMService(provider="mock")
        assert service is not None

        # Verify provider switching capability exists
        assert hasattr(service, 'switch_provider')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
