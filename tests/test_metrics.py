"""
Tests for Metrics Evaluation System
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.metrics import PromptMetrics


class TestPromptMetrics:
    """Test metrics evaluation"""

    def test_initialization(self):
        """Test metrics system initializes"""
        metrics = PromptMetrics()
        assert metrics is not None
        assert len(metrics.metrics_history) == 0

    def test_parse_llm_response_json(self):
        """Test parsing JSON response"""
        metrics = PromptMetrics()
        response = '''
        {
            "high_value_transactions": [
                {"transaction_id": "T001", "amount": 500},
                {"transaction_id": "T002", "amount": 750}
            ],
            "anomalies": [
                {"transaction_id": "T003", "type": "unusual"}
            ]
        }
        '''

        parsed = metrics.parse_llm_response(response)

        assert 'high_value_transactions' in parsed
        assert len(parsed['high_value_transactions']) == 2

    def test_parse_llm_response_text(self):
        """Test parsing text response"""
        metrics = PromptMetrics()
        response = '''
        HIGH-VALUE TRANSACTIONS:
        - Transaction #123: £500.00
        - Transaction #456: £750.00

        ANOMALIES DETECTED:
        - Unusual pattern in transaction #789
        '''

        parsed = metrics.parse_llm_response(response)

        assert 'raw_response' in parsed
        assert isinstance(parsed['high_value_transactions'], list)

    def test_calculate_precision_recall_f1(self):
        """Test precision, recall, F1 calculation"""
        metrics = PromptMetrics()

        predicted = {'count_above_threshold': 10}
        ground_truth = {'count_above_threshold': 12}

        result = metrics.calculate_precision_recall_f1(predicted, ground_truth)

        assert 'precision' in result
        assert 'recall' in result
        assert 'f1_score' in result
        assert 0 <= result['precision'] <= 1
        assert 0 <= result['recall'] <= 1
        assert 0 <= result['f1_score'] <= 1

    def test_calculate_completeness(self):
        """Test completeness calculation"""
        metrics = PromptMetrics()

        response = '''
        HIGH-VALUE TRANSACTIONS:
        - Transaction 1
        ANOMALIES:
        - Anomaly 1
        SUMMARY:
        - Total found: 5
        '''

        ground_truth = {'count_above_threshold': 5}
        completeness = metrics.calculate_completeness(response, ground_truth)

        assert isinstance(completeness, float)
        assert 0 <= completeness <= 1

    def test_calculate_format_quality(self):
        """Test format quality evaluation"""
        metrics = PromptMetrics()

        good_response = '''
        HIGH-VALUE TRANSACTIONS:
        - Transaction 1: £500
        - Transaction 2: £750

        ANOMALIES:
        - Suspicious pattern

        SUMMARY:
        Total: 2 transactions
        '''

        quality = metrics.calculate_format_quality(good_response)

        assert isinstance(quality, float)
        assert 0 <= quality <= 1
        assert quality > 0  # Should have some quality

    def test_calculate_specificity(self):
        """Test specificity scoring"""
        metrics = PromptMetrics()

        specific_response = '''
        Transaction #T001 on 2024-01-15: £500.00
        Transaction #T002 on 2024-01-16: £750.00
        '''

        specificity = metrics.calculate_specificity(specific_response)

        assert isinstance(specificity, float)
        assert 0 <= specificity <= 1

    def test_evaluate_prompt_success(self):
        """Test successful prompt evaluation"""
        metrics = PromptMetrics()

        llm_result = {
            'success': True,
            'response': 'HIGH-VALUE: Transaction 1: £500',
            'tokens_used': 100,
            'latency': 1.5
        }

        ground_truth = {
            'high_value_transactions': ['T001', 'T002'],
            'count_above_threshold': 2
        }

        result = metrics.evaluate_prompt('test_prompt', llm_result, ground_truth)

        assert result['success'] is True
        assert 'composite_score' in result
        assert 'accuracy' in result
        assert 'f1_score' in result
        assert 0 <= result['composite_score'] <= 1

    def test_evaluate_prompt_failure(self):
        """Test evaluation with failed LLM call"""
        metrics = PromptMetrics()

        llm_result = {
            'success': False,
            'error': 'API error'
        }

        ground_truth = {'count_above_threshold': 5}

        result = metrics.evaluate_prompt('test_prompt', llm_result, ground_truth)

        assert result['success'] is False
        assert 'error' in result

    def test_compare_prompts(self):
        """Test comparing multiple prompts"""
        metrics = PromptMetrics()

        metrics_list = [
            {
                'prompt_name': 'prompt1',
                'composite_score': 0.75,
                'f1_score': 0.70,
                'accuracy': 0.80,
                'latency': 1.0
            },
            {
                'prompt_name': 'prompt2',
                'composite_score': 0.85,
                'f1_score': 0.82,
                'accuracy': 0.88,
                'latency': 1.5
            },
            {
                'prompt_name': 'prompt3',
                'composite_score': 0.60,
                'f1_score': 0.55,
                'accuracy': 0.65,
                'latency': 0.8
            }
        ]

        comparison = metrics.compare_prompts(metrics_list)

        assert 'best_prompt' in comparison
        assert comparison['best_prompt']['name'] == 'prompt2'
        assert comparison['best_prompt']['composite_score'] == 0.85
        assert 'worst_prompt' in comparison
        assert 'average_scores' in comparison
        assert len(comparison['rankings']) == 3

    def test_get_improvement_suggestions(self):
        """Test improvement suggestions generation"""
        metrics = PromptMetrics()

        low_metrics = {
            'accuracy': 0.5,
            'completeness': 0.4,
            'format_quality': 0.3,
            'specificity': 0.4,
            'f1_score': 0.5
        }

        suggestions = metrics.get_improvement_suggestions(low_metrics)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0  # Should have suggestions for low scores

    def test_metrics_history(self):
        """Test metrics history tracking"""
        metrics = PromptMetrics()

        llm_result = {
            'success': True,
            'response': 'Test response',
            'tokens_used': 50,
            'latency': 1.0
        }

        ground_truth = {'count_above_threshold': 5}

        # Evaluate multiple times
        metrics.evaluate_prompt('prompt1', llm_result, ground_truth)
        metrics.evaluate_prompt('prompt2', llm_result, ground_truth)

        assert len(metrics.metrics_history) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
