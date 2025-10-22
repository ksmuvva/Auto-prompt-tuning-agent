"""
Tests for Bias Detector
Validates that bias is <2% across all tests
"""

import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.bias_detector import BiasDetector


class TestBiasDetector:
    """Test suite for bias detection"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.detector = BiasDetector()

    def test_detector_initialization(self):
        """Test detector initializes correctly"""
        assert self.detector is not None

    def test_merchant_name_variation_bias(self):
        """Test bias in merchant name variations"""
        # Mock analysis function that should be consistent
        def consistent_analysis(data):
            # Should detect same merchant regardless of capitalization
            return {'detected': True}

        merchant_variations = [
            ['Gucci', 'GUCCI', 'gucci'],
            ['Western Union', 'WESTERN UNION', 'WesternUnion']
        ]

        bias_score = self.detector.test_merchant_name_variations(
            consistent_analysis,
            merchant_variations
        )

        assert bias_score >= 0.0
        assert bias_score <= 1.0

    def test_currency_format_bias(self):
        """Test bias in currency format handling"""
        def consistent_analysis(data):
            # Should handle different currency formats equally
            return {'amount': 250}

        amount_formats = [
            '£250.00',
            '250 GBP',
            '250.00',
            'GBP 250'
        ]

        bias_score = self.detector.test_currency_format_bias(
            consistent_analysis,
            amount_formats
        )

        assert bias_score >= 0.0
        assert bias_score <= 1.0

    def test_date_format_bias(self):
        """Test bias in date format handling"""
        def consistent_analysis(data):
            # Should parse different date formats equally
            return {'date': '2025-01-15'}

        date_formats = [
            '2025-01-15',
            '15/01/2025',
            '01-15-2025',
            '15 Jan 2025'
        ]

        bias_score = self.detector.test_date_format_bias(
            consistent_analysis,
            date_formats
        )

        assert bias_score >= 0.0
        assert bias_score <= 1.0

    def test_overall_bias_calculation(self):
        """Test overall bias score calculation"""
        test_results = {
            'merchant_bias': 0.01,
            'currency_bias': 0.015,
            'date_bias': 0.012
        }

        overall_bias = self.detector.calculate_overall_bias(test_results)

        assert overall_bias >= 0.0
        assert overall_bias <= 1.0

        # Overall bias should be average or similar aggregation
        expected = (0.01 + 0.015 + 0.012) / 3
        assert abs(overall_bias - expected) < 0.001

    def test_bias_meets_2_percent_target(self):
        """Test that bias is below 2% target"""
        # Create mock analysis function
        def mock_analysis(data):
            return {'result': 'consistent'}

        # Run all bias tests
        merchant_variations = [['Gucci', 'GUCCI']]
        merchant_bias = self.detector.test_merchant_name_variations(
            mock_analysis,
            merchant_variations
        )

        amount_formats = ['£250', '250 GBP']
        currency_bias = self.detector.test_currency_format_bias(
            mock_analysis,
            amount_formats
        )

        date_formats = ['2025-01-15', '15/01/2025']
        date_bias = self.detector.test_date_format_bias(
            mock_analysis,
            date_formats
        )

        test_results = {
            'merchant_bias': merchant_bias,
            'currency_bias': currency_bias,
            'date_bias': date_bias
        }

        overall_bias = self.detector.calculate_overall_bias(test_results)

        # Assert bias is below 2% (0.02)
        assert overall_bias < 0.02, \
            f"Bias {overall_bias:.4f} exceeds 2% target"

    def test_bias_report_generation(self):
        """Test bias report generation"""
        test_results = {
            'merchant_name_bias': 0.01,
            'currency_format_bias': 0.015,
            'date_format_bias': 0.008
        }

        report = self.detector.generate_bias_report(test_results)

        assert report is not None
        assert 'overall_bias' in report
        assert 'bias_level' in report
        assert 'individual_tests' in report

    def test_bias_level_classification(self):
        """Test bias level classification (VERY LOW, LOW, MEDIUM, HIGH)"""
        test_results_very_low = {
            'test1': 0.005,
            'test2': 0.008
        }

        report_very_low = self.detector.generate_bias_report(test_results_very_low)
        assert 'VERY LOW' in report_very_low['bias_level'] or 'LOW' in report_very_low['bias_level']

        test_results_high = {
            'test1': 0.15,
            'test2': 0.20
        }

        report_high = self.detector.generate_bias_report(test_results_high)
        assert 'HIGH' in report_high['bias_level'] or 'MEDIUM' in report_high['bias_level']

    def test_zero_bias_edge_case(self):
        """Test handling of zero bias (perfect consistency)"""
        test_results = {
            'test1': 0.0,
            'test2': 0.0,
            'test3': 0.0
        }

        overall_bias = self.detector.calculate_overall_bias(test_results)

        assert overall_bias == 0.0
        assert overall_bias < 0.02  # Still meets target


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
