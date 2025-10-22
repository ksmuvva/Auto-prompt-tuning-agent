"""
Tests for Dynamic Prompt Generator
Validates meta-prompting and iterative optimization
"""

import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.dynamic_prompts import DynamicPromptGenerator
from agent.llm_service import LLMService


class TestDynamicPromptGenerator:
    """Test suite for dynamic prompt generation"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.llm_service = LLMService(provider='mock')
        self.generator = DynamicPromptGenerator(self.llm_service)

    def test_generator_initialization(self):
        """Test generator initializes correctly"""
        assert self.generator.llm_service is not None
        assert hasattr(self.generator, 'generate_from_failures')

    def test_generate_from_failures(self):
        """Test generating prompts from failure analysis"""
        failures = [
            {'id': 1, 'reason': 'Missed high-value transaction'},
            {'id': 2, 'reason': 'False positive on luxury brand'}
        ]

        current_prompt = "Identify high-value transactions"
        current_metrics = {'precision': 0.85, 'accuracy': 0.87}

        new_prompt = self.generator.generate_from_failures(
            failures,
            current_prompt,
            current_metrics
        )

        assert new_prompt is not None
        assert isinstance(new_prompt, str)
        assert len(new_prompt) > 0

    def test_optimize_for_metric(self):
        """Test optimizing prompt for specific metric"""
        prompt = self.generator.optimize_for_metric(
            target_metric='precision',
            target_value=0.98,
            current_value=0.85,
            context="High-value transaction detection"
        )

        assert prompt is not None
        assert 'precision' in prompt.lower() or '98%' in prompt

    def test_generate_reasoning_prompt(self):
        """Test generating chain-of-thought reasoning prompts"""
        task_description = "Detect luxury brand purchases"
        examples = [
            {'merchant': 'Gucci', 'amount': 500, 'result': 'luxury'},
            {'merchant': 'Tesco', 'amount': 500, 'result': 'not_luxury'}
        ]

        prompt = self.generator.generate_reasoning_prompt(task_description, examples)

        assert prompt is not None
        assert 'step by step' in prompt.lower() or 'reasoning' in prompt.lower()

    def test_generate_fw_specific_prompt(self):
        """Test generating FW requirement-specific prompts"""
        # Test FW15
        fw15_prompt = self.generator.generate_fw_specific_prompt('fw15', threshold=250)
        assert '250' in fw15_prompt
        assert 'high' in fw15_prompt.lower() or 'value' in fw15_prompt.lower()

        # Test FW45
        fw45_prompt = self.generator.generate_fw_specific_prompt('fw45', threshold=None)
        assert 'gambling' in fw45_prompt.lower()

        # Test FW50
        fw50_prompt = self.generator.generate_fw_specific_prompt('fw50', threshold=500)
        assert 'debt' in fw50_prompt.lower() or 'payment' in fw50_prompt.lower()

    def test_iterate_until_target(self):
        """Test iterative prompt improvement until target is met"""
        initial_prompt = "Find high-value transactions"

        # Mock evaluation function
        iteration_count = [0]

        def evaluate_function(prompt):
            iteration_count[0] += 1
            # Simulate improvement over iterations
            return {
                'precision': 0.85 + (iteration_count[0] * 0.05),
                'accuracy': 0.87 + (iteration_count[0] * 0.04),
                'meets_target': (0.85 + (iteration_count[0] * 0.05)) >= 0.98
            }

        result = self.generator.iterate_until_target(
            initial_prompt,
            evaluate_function,
            max_iterations=5
        )

        assert result is not None
        assert 'final_prompt' in result
        assert 'iterations' in result
        assert result['iterations'] <= 5

    def test_prompt_quality_metrics(self):
        """Test that generated prompts meet quality standards"""
        prompt = self.generator.generate_fw_specific_prompt('fw15', threshold=250)

        # Check prompt length (should be substantial)
        assert len(prompt) > 100, "Prompt too short"

        # Check prompt contains key elements
        assert 'transaction' in prompt.lower()
        assert '250' in prompt or 'threshold' in prompt.lower()

        # Check prompt structure (should have clear instructions)
        assert prompt.count('\n') > 3, "Prompt lacks structure"

    def test_meta_prompting_capability(self):
        """Test meta-prompting (using LLM to generate prompts)"""
        # This tests the core meta-prompting feature
        context = {
            'requirement': 'fw15',
            'current_performance': {'precision': 0.85},
            'target_performance': {'precision': 0.98}
        }

        meta_prompt = self.generator.generate_from_failures(
            failures=[],
            current_prompt="Basic prompt",
            current_metrics=context['current_performance']
        )

        assert meta_prompt is not None
        # Meta-generated prompts should be more sophisticated
        assert len(meta_prompt) > len("Basic prompt")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
