"""
Tests for FW15: High-Value Transactions (>£250)
Validates 98% precision and accuracy requirements
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.requirement_analyzer import RequirementAnalyzer
from agent.ground_truth import GroundTruthManager
from agent.metrics import calculate_precision_advanced, calculate_accuracy_advanced, meets_target_metrics


class TestFW15HighValue:
    """Test suite for FW15 requirement"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.analyzer = RequirementAnalyzer()
        self.ground_truth = GroundTruthManager()
        self.threshold = 250

    def test_fw15_basic_analysis(self):
        """Test basic FW15 analysis runs without errors"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        assert result is not None
        assert 'high_value_transactions' in result
        assert 'summary' in result
        assert isinstance(result['high_value_transactions'], list)

    def test_fw15_threshold_enforcement(self):
        """Test that only transactions >£250 are included"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        for transaction in result['high_value_transactions']:
            assert float(transaction['amount']) > self.threshold, \
                f"Transaction {transaction['id']} has amount {transaction['amount']} <= {self.threshold}"

    def test_fw15_completeness(self):
        """Test that all high-value transactions are detected"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        # Compare with ground truth
        ground_truth_count = len(self.ground_truth.ground_truth.get('high_value_transactions', []))
        detected_count = len(result['high_value_transactions'])

        # Should detect all (100% recall)
        assert detected_count == ground_truth_count, \
            f"Expected {ground_truth_count} high-value transactions, found {detected_count}"

    def test_fw15_precision_target(self):
        """Test precision meets 98% target"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        # Validate against ground truth
        ground_truth_ids = {
            tx['transaction_id']
            for tx in self.ground_truth.ground_truth.get('high_value_transactions', [])
        }

        detected_ids = {tx['id'] for tx in result['high_value_transactions']}

        # Calculate precision
        true_positives = len(ground_truth_ids.intersection(detected_ids))
        false_positives = len(detected_ids - ground_truth_ids)

        precision = calculate_precision_advanced(true_positives, false_positives)

        assert precision >= 0.98, f"Precision {precision:.4f} does not meet 98% target"

    def test_fw15_accuracy_target(self):
        """Test accuracy meets 98% target"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        # Load ground truth
        ground_truth_ids = {
            tx['transaction_id']
            for tx in self.ground_truth.ground_truth.get('high_value_transactions', [])
        }

        detected_ids = {tx['id'] for tx in result['high_value_transactions']}

        # Calculate confusion matrix
        all_transaction_ids = set(range(1, 3001))  # 3000 transactions

        true_positives = len(ground_truth_ids.intersection(detected_ids))
        false_positives = len(detected_ids - ground_truth_ids)
        false_negatives = len(ground_truth_ids - detected_ids)
        true_negatives = len(all_transaction_ids - ground_truth_ids - detected_ids)

        accuracy = calculate_accuracy_advanced(true_positives, true_negatives, false_positives, false_negatives)

        assert accuracy >= 0.98, f"Accuracy {accuracy:.4f} does not meet 98% target"

    def test_fw15_summary_statistics(self):
        """Test summary statistics are correctly calculated"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        summary = result['summary']

        assert 'total_count' in summary
        assert 'total_amount' in summary
        assert 'average_amount' in summary
        assert 'max_amount' in summary

        # Verify calculations
        transactions = result['high_value_transactions']
        total_amount = sum(float(tx['amount']) for tx in transactions)
        avg_amount = total_amount / len(transactions) if transactions else 0

        assert summary['total_count'] == len(transactions)
        assert abs(float(summary['total_amount']) - total_amount) < 0.01
        assert abs(float(summary['average_amount']) - avg_amount) < 0.01

    def test_fw15_grouping_by_merchant(self):
        """Test transactions are grouped by merchant correctly"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        if 'by_merchant' in result:
            # Verify grouping logic
            by_merchant = result['by_merchant']
            assert isinstance(by_merchant, dict)

            # Each merchant should have list of transactions
            for merchant, transactions in by_merchant.items():
                assert isinstance(transactions, list)
                assert len(transactions) > 0

    def test_fw15_no_false_negatives(self):
        """Test that no high-value transactions are missed (recall = 100%)"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        ground_truth_ids = {
            tx['transaction_id']
            for tx in self.ground_truth.ground_truth.get('high_value_transactions', [])
        }

        detected_ids = {tx['id'] for tx in result['high_value_transactions']}

        false_negatives = ground_truth_ids - detected_ids

        assert len(false_negatives) == 0, \
            f"Missed {len(false_negatives)} high-value transactions: {false_negatives}"

    def test_fw15_meets_targets(self):
        """Test overall that precision and accuracy both meet 98% targets"""
        result = self.analyzer.analyze_fw15_high_value(threshold=self.threshold)

        # Validate
        validation = self.ground_truth.validate_fw15_high_value(
            [tx['id'] for tx in result['high_value_transactions']]
        )

        precision = validation['precision']
        accuracy = validation['accuracy']

        # Check against targets
        meets_targets = meets_target_metrics(precision, accuracy, 0.98, 0.98)

        assert meets_targets['meets_targets'], \
            f"Failed to meet targets: Precision={precision:.4f}, Accuracy={accuracy:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
