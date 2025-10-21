"""
Integration Tests for the AI Agent System
"""

import pytest
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.core import PromptTuningAgent, AgentMemory
from agent.llm_service import LLMService
from prompts.templates import PromptTemplateLibrary


class TestAgentIntegration:
    """Integration tests for the full agent"""

    def test_agent_initialization(self):
        """Test agent initializes with all components"""
        agent = PromptTuningAgent(llm_provider='mock')

        assert agent.state['initialized'] is True
        assert agent.llm_service is not None
        assert agent.data_processor is not None
        assert agent.template_library is not None
        assert agent.prompt_tuner is not None
        assert agent.memory is not None

    def test_agent_status(self):
        """Test getting agent status"""
        agent = PromptTuningAgent(llm_provider='mock')
        status = agent.get_status()

        assert 'state' in status
        assert 'memory_size' in status
        assert 'llm_stats' in status
        assert 'templates_available' in status

    def test_custom_prompt_workflow(self):
        """Test adding and using custom prompts"""
        agent = PromptTuningAgent(llm_provider='mock')

        success = agent.add_custom_prompt(
            name="test_custom",
            template_text="Analyze: {data}, Threshold: {threshold}",
            description="Test custom prompt"
        )

        assert success is True
        assert 'test_custom' in agent.template_library.list_templates()

    def test_agent_reset(self):
        """Test resetting agent state"""
        agent = PromptTuningAgent(llm_provider='mock')

        # Set some state
        agent.best_results = {'test': 'data'}
        agent.prompt_tuner.best_score = 0.75

        # Reset
        agent.reset()

        assert agent.best_results is None
        assert agent.prompt_tuner.best_score == 0.0


class TestAgentMemory:
    """Test agent memory system"""

    def test_memory_initialization(self):
        """Test memory system initializes"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            memory_file = f.name

        try:
            memory = AgentMemory(memory_file=memory_file)
            assert memory is not None
            assert isinstance(memory.short_term, list)
            assert isinstance(memory.long_term, dict)
            assert isinstance(memory.learned_patterns, list)
        finally:
            if os.path.exists(memory_file):
                os.unlink(memory_file)

    def test_memory_store_and_recall(self):
        """Test storing and recalling knowledge"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            memory_file = f.name

        try:
            memory = AgentMemory(memory_file=memory_file)

            # Store knowledge
            memory.store_knowledge('test_key', 'test_value')

            # Recall
            value = memory.recall('test_key')
            assert value == 'test_value'

            # Recall non-existent
            value = memory.recall('nonexistent')
            assert value is None
        finally:
            if os.path.exists(memory_file):
                os.unlink(memory_file)

    def test_memory_interactions(self):
        """Test adding interactions to short-term memory"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            memory_file = f.name

        try:
            memory = AgentMemory(memory_file=memory_file)

            # Add interactions
            memory.add_interaction({'action': 'test1'})
            memory.add_interaction({'action': 'test2'})

            assert len(memory.short_term) == 2
        finally:
            if os.path.exists(memory_file):
                os.unlink(memory_file)

    def test_memory_learned_patterns(self):
        """Test learning patterns"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            memory_file = f.name

        try:
            memory = AgentMemory(memory_file=memory_file)

            # Learn pattern
            memory.learn_pattern({'best_prompt': 'test', 'score': 0.85})

            assert len(memory.learned_patterns) == 1
            assert memory.learned_patterns[0]['pattern']['best_prompt'] == 'test'
        finally:
            if os.path.exists(memory_file):
                os.unlink(memory_file)


class TestEndToEndWorkflow:
    """Test end-to-end workflows"""

    def test_llm_service_to_metrics(self):
        """Test LLM service integration with metrics"""
        from agent.metrics import PromptMetrics

        llm_service = LLMService(provider='mock')
        metrics = PromptMetrics()

        # Generate response
        result = llm_service.generate("Test prompt")

        # Evaluate
        ground_truth = {'count_above_threshold': 3, 'high_value_transactions': []}
        evaluation = metrics.evaluate_prompt('test', result, ground_truth)

        assert evaluation['success'] is True
        assert 'composite_score' in evaluation

    def test_template_to_llm_workflow(self):
        """Test template formatting to LLM generation"""
        library = PromptTemplateLibrary()
        llm_service = LLMService(provider='mock')

        # Format template
        prompt = library.format_template(
            'direct_concise',
            data='test data',
            threshold=250
        )

        # Generate
        result = llm_service.generate(prompt)

        assert result['success'] is True
        assert len(result['response']) > 0


class TestConfigurationHandling:
    """Test configuration handling"""

    def test_agent_with_config(self):
        """Test agent initialization with config"""
        config = {
            'llm': {'model': 'test-model'},
            'tuning': {'max_iterations': 5}
        }

        agent = PromptTuningAgent(
            llm_provider='mock',
            config=config
        )

        assert agent.config == config

    def test_agent_without_config(self):
        """Test agent initialization without config"""
        agent = PromptTuningAgent(llm_provider='mock')
        assert agent.config == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
