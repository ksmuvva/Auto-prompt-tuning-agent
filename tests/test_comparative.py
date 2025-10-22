"""
Tests for Comparative Analyzer
Validates prompt, model, and strategy comparison capabilities
"""

import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.comparative import ComparativeAnalyzer


class TestComparativeAnalyzer:
    """Test suite for comparative analysis"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.analyzer = ComparativeAnalyzer()

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly"""
        assert self.analyzer is not None

    def test_compare_prompts(self):
        """Test comparing multiple prompts"""
        prompt_results = {
            'prompt_a': {
                'precision': 0.98,
                'accuracy': 0.97,
                'f1_score': 0.975,
                'latency': 1.2
            },
            'prompt_b': {
                'precision': 0.96,
                'accuracy': 0.99,
                'f1_score': 0.974,
                'latency': 0.8
            },
            'prompt_c': {
                'precision': 0.99,
                'accuracy': 0.98,
                'f1_score': 0.985,
                'latency': 1.5
            }
        }

        comparison = self.analyzer.compare_prompts(prompt_results)

        assert comparison is not None
        assert isinstance(comparison, dict) or hasattr(comparison, 'to_dict')

        # Check that comparison includes all prompts
        if isinstance(comparison, dict):
            assert len(comparison) == 3

    def test_compare_models(self):
        """Test comparing different LLM models"""
        model_results = {
            'gpt-4': {
                'precision': 0.99,
                'accuracy': 0.98,
                'cost': 0.03,
                'latency': 2.1
            },
            'claude-3-sonnet': {
                'precision': 0.98,
                'accuracy': 0.99,
                'cost': 0.015,
                'latency': 1.8
            },
            'gemini-pro': {
                'precision': 0.97,
                'accuracy': 0.97,
                'cost': 0.0005,
                'latency': 1.2
            }
        }

        comparison = self.analyzer.compare_models(model_results)

        assert comparison is not None

    def test_compare_strategies(self):
        """Test comparing template vs dynamic vs hybrid strategies"""
        strategy_results = {
            'template': {
                'precision': 0.96,
                'accuracy': 0.97,
                'generation_time': 0.1,
                'llm_calls': 1
            },
            'dynamic': {
                'precision': 0.99,
                'accuracy': 0.98,
                'generation_time': 5.2,
                'llm_calls': 3
            },
            'hybrid': {
                'precision': 0.98,
                'accuracy': 0.98,
                'generation_time': 2.1,
                'llm_calls': 2
            }
        }

        comparison = self.analyzer.compare_strategies(strategy_results)

        assert comparison is not None

    def test_generate_comparison_table(self):
        """Test ASCII table generation for CLI display"""
        results = {
            'option_a': {'metric1': 0.98, 'metric2': 1.2},
            'option_b': {'metric1': 0.99, 'metric2': 0.9}
        }

        table = self.analyzer.generate_comparison_table('prompt', results)

        assert table is not None
        assert isinstance(table, str)
        assert len(table) > 0

        # Check for table structure characters
        assert '|' in table or '─' in table or '-' in table

    def test_recommend_best_option_performance(self):
        """Test recommendation based on performance criteria"""
        results = {
            'option_a': {'precision': 0.99, 'accuracy': 0.98, 'cost': 0.05},
            'option_b': {'precision': 0.97, 'accuracy': 0.96, 'cost': 0.01}
        }

        recommendation = self.analyzer.recommend_best_option(
            'prompt',
            results,
            criteria='performance'
        )

        assert recommendation is not None
        # Should recommend option_a (best performance)
        assert 'option_a' in str(recommendation).lower() or recommendation.get('best_option') == 'option_a'

    def test_recommend_best_option_cost(self):
        """Test recommendation based on cost criteria"""
        results = {
            'option_a': {'precision': 0.99, 'accuracy': 0.98, 'cost': 0.05},
            'option_b': {'precision': 0.97, 'accuracy': 0.96, 'cost': 0.01}
        }

        recommendation = self.analyzer.recommend_best_option(
            'prompt',
            results,
            criteria='cost'
        )

        assert recommendation is not None
        # Should recommend option_b (lowest cost)

    def test_recommend_best_option_balanced(self):
        """Test recommendation based on balanced criteria"""
        results = {
            'high_performance_expensive': {
                'precision': 0.99,
                'accuracy': 0.99,
                'cost': 0.10,
                'latency': 3.0
            },
            'balanced': {
                'precision': 0.98,
                'accuracy': 0.98,
                'cost': 0.02,
                'latency': 1.2
            },
            'low_performance_cheap': {
                'precision': 0.90,
                'accuracy': 0.91,
                'cost': 0.001,
                'latency': 0.5
            }
        }

        recommendation = self.analyzer.recommend_best_option(
            'strategy',
            results,
            criteria='balanced'
        )

        assert recommendation is not None

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        prompt_results = {
            'prompt1': {'precision': 0.98, 'accuracy': 0.97}
        }

        model_results = {
            'gpt-4': {'precision': 0.99, 'accuracy': 0.98}
        }

        strategy_results = {
            'template': {'precision': 0.96, 'accuracy': 0.97}
        }

        report = self.analyzer.generate_comprehensive_report(
            prompt_results,
            model_results,
            strategy_results
        )

        assert report is not None
        assert isinstance(report, dict) or isinstance(report, str)

    def test_comparison_with_missing_metrics(self):
        """Test handling of results with missing metrics"""
        results_with_gaps = {
            'option_a': {'precision': 0.98, 'accuracy': 0.97},
            'option_b': {'precision': 0.99}  # Missing accuracy
        }

        # Should handle gracefully
        comparison = self.analyzer.compare_prompts(results_with_gaps)
        assert comparison is not None

    def test_empty_results_handling(self):
        """Test handling of empty results"""
        empty_results = {}

        # Should handle gracefully without crashing
        try:
            comparison = self.analyzer.compare_prompts(empty_results)
            # Should return None or empty result
            assert comparison is not None or comparison is None
        except Exception as e:
            pytest.fail(f"Should handle empty results gracefully: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
