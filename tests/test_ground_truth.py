"""
Tests for Ground Truth Manager
Validates that ground truth data is loaded correctly and validation works
"""

import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.ground_truth import GroundTruthManager


class TestGroundTruthManager:
    """Test suite for ground truth management system"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.manager = GroundTruthManager()

    def test_ground_truth_file_exists(self):
        """Test that ground truth file exists"""
        assert self.manager.ground_truth_file.exists(), \
            f"Ground truth file not found: {self.manager.ground_truth_file}"

    def test_ground_truth_loaded(self):
        """Test that ground truth data is loaded correctly"""
        assert self.manager.ground_truth is not None
        assert isinstance(self.manager.ground_truth, dict)

    def test_ground_truth_structure(self):
        """Test ground truth has all required fields"""
        required_fields = [
            'high_value_transactions',
            'luxury_brands',
            'money_transfers',
            'missing_audit_trail',
            'missing_months',
            'errors',
            'gambling',
            'debt_payments'
        ]

        for field in required_fields:
            assert field in self.manager.ground_truth, \
                f"Missing required field: {field}"

    def test_precision_calculation(self):
        """Test precision calculation accuracy"""
        # Test case 1: Perfect precision
        precision = self.manager.calculate_precision(100, 0)
        assert precision == 1.0

        # Test case 2: 98% precision
        precision = self.manager.calculate_precision(98, 2)
        assert precision == 0.98

        # Test case 3: Zero division handling
        precision = self.manager.calculate_precision(0, 0)
        assert precision == 0.0

    def test_accuracy_calculation(self):
        """Test accuracy calculation accuracy"""
        # Test case 1: Perfect accuracy
        accuracy = self.manager.calculate_accuracy(100, 100, 0, 0)
        assert accuracy == 1.0

        # Test case 2: 98% accuracy
        accuracy = self.manager.calculate_accuracy(98, 100, 2, 0)
        assert accuracy == 0.99  # (98+100)/(98+100+2+0) = 198/200 = 0.99

        # Test case 3: Zero division handling
        accuracy = self.manager.calculate_accuracy(0, 0, 0, 0)
        assert accuracy == 0.0

    def test_fw15_validation(self):
        """Test FW15 high-value transaction validation"""
        # Get ground truth IDs
        ground_truth_ids = [
            tx['transaction_id']
            for tx in self.manager.ground_truth.get('high_value_transactions', [])
        ]

        # Test perfect predictions
        validation = self.manager.validate_fw15_high_value(ground_truth_ids)

        assert validation['precision'] == 1.0
        assert validation['accuracy'] >= 0.98
        assert validation['meets_98_target'] is True

    def test_fw20_luxury_validation(self):
        """Test FW20 luxury brand validation"""
        ground_truth_ids = [
            tx['transaction_id']
            for tx in self.manager.ground_truth.get('luxury_brands', [])
        ]

        validation = self.manager.validate_fw20_luxury_brands(ground_truth_ids)

        assert 'precision' in validation
        assert 'accuracy' in validation
        assert isinstance(validation['meets_98_target'], bool)

    def test_fw25_missing_audit_validation(self):
        """Test FW25 missing audit trail validation"""
        ground_truth_ids = [
            tx['transaction_id']
            for tx in self.manager.ground_truth.get('missing_audit_trail', [])
        ]

        validation = self.manager.validate_fw25_missing_audit(ground_truth_ids)

        assert 'precision' in validation
        assert 'recall' in validation

    def test_fw30_missing_months_validation(self):
        """Test FW30 missing months validation"""
        expected_missing = self.manager.ground_truth.get('missing_months', [])

        validation = self.manager.validate_fw30_missing_months(expected_missing)

        assert validation['perfect_match'] is True
        assert validation['accuracy'] == 1.0

    def test_fw45_gambling_validation(self):
        """Test FW45 gambling transaction validation"""
        ground_truth_ids = [
            tx['transaction_id']
            for tx in self.manager.ground_truth.get('gambling', [])
        ]

        validation = self.manager.validate_fw45_gambling(ground_truth_ids)

        assert 'precision' in validation
        assert 'accuracy' in validation

    def test_fw50_debt_payments_validation(self):
        """Test FW50 debt payment validation"""
        ground_truth_ids = [
            tx['transaction_id']
            for tx in self.manager.ground_truth.get('debt_payments', [])
        ]

        validation = self.manager.validate_fw50_debt_payments(ground_truth_ids)

        assert 'precision' in validation
        assert 'accuracy' in validation

    def test_validation_report_generation(self):
        """Test comprehensive validation report generation"""
        all_validations = {
            'fw15': {'precision': 0.98, 'accuracy': 0.99, 'meets_98_target': True},
            'fw20': {'precision': 0.97, 'accuracy': 0.98, 'meets_98_target': False},
            'fw30': {'perfect_match': True, 'accuracy': 1.0}
        }

        report = self.manager.generate_validation_report(all_validations)

        assert 'summary' in report
        assert 'passed' in report['summary']
        assert 'failed' in report['summary']
        assert 'details' in report

    def test_ground_truth_never_exposed_to_llm(self):
        """Test that ground truth is separate from LLM analysis"""
        # This is a design test - ground truth should only be used for validation
        # Not for LLM prompts

        # Verify ground truth file is in data directory (not exposed)
        assert 'data' in str(self.manager.ground_truth_file)

        # Verify it's not in prompts or results directories
        assert 'prompts' not in str(self.manager.ground_truth_file)
        assert 'results' not in str(self.manager.ground_truth_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
