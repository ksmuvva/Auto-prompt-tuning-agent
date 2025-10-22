"""
Bias Detector
Detects and reports bias in LLM outputs and analysis
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Detects bias in LLM analysis outputs
    
    Tests for:
    - Demographic bias
    - Format bias (different data formats treated differently)
    - Linguistic bias (name variations treated inconsistently)
    """

    def __init__(self):
        self.bias_reports = []

    def test_merchant_name_variations(
        self,
        analysis_function: callable,
        merchant_variations: List[str],
        expected_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Test if different spellings of same merchant are treated equally
        
        Args:
            analysis_function: Function that analyzes transactions
            merchant_variations: List of merchant name variations
                                (e.g., ['Amazon UK', 'AMAZON', 'amazon.co.uk'])
            expected_detection: Whether these should be detected
        
        Returns:
            Bias report
        """
        results = {}
        
        for variant in merchant_variations:
            result = analysis_function(variant)
            results[variant] = result
        
        # Calculate consistency
        detection_rates = [r.get('detected', False) for r in results.values()]
        all_detected = all(detection_rates)
        none_detected = not any(detection_rates)
        consistent = all_detected or none_detected
        
        bias_score = 0.0 if consistent else 1.0 - (sum(detection_rates) / len(detection_rates))
        
        report = {
            'test': 'merchant_name_variations',
            'timestamp': datetime.now().isoformat(),
            'merchant_variations': merchant_variations,
            'results': results,
            'consistent': consistent,
            'bias_score': bias_score,
            'bias_level': 'LOW' if bias_score < 0.05 else 'MEDIUM' if bias_score < 0.15 else 'HIGH',
            'passed': consistent
        }
        
        logger.info(f"Merchant variation bias test: {'PASSED' if consistent else 'FAILED'} (score: {bias_score:.2%})")
        
        return report

    def test_currency_format_bias(
        self,
        analysis_function: callable,
        amount_formats: List[str]
    ) -> Dict[str, Any]:
        """
        Test if different currency formats are treated equally
        
        Args:
            analysis_function: Function that analyzes amounts
            amount_formats: List of same amount in different formats
                           (e.g., ['£250', '250 GBP', '250.00', 'GBP 250'])
        
        Returns:
            Bias report
        """
        results = {}
        
        for format_variant in amount_formats:
            result = analysis_function(format_variant)
            results[format_variant] = result
        
        # Check if all formats produce same result
        values = [r.get('parsed_amount', 0) for r in results.values()]
        consistent = len(set(values)) == 1
        
        bias_score = 0.0 if consistent else 0.2  # Arbitrary penalty for inconsistency
        
        report = {
            'test': 'currency_format_bias',
            'timestamp': datetime.now().isoformat(),
            'amount_formats': amount_formats,
            'results': results,
            'consistent': consistent,
            'bias_score': bias_score,
            'bias_level': 'LOW' if bias_score < 0.05 else 'MEDIUM',
            'passed': consistent
        }
        
        logger.info(f"Currency format bias test: {'PASSED' if consistent else 'FAILED'}")
        
        return report

    def test_date_format_bias(
        self,
        analysis_function: callable,
        date_formats: List[str]
    ) -> Dict[str, Any]:
        """
        Test if different date formats are treated equally
        
        Args:
            analysis_function: Function that analyzes dates
            date_formats: List of same date in different formats
                         (e.g., ['2025-01-15', '15/01/2025', '15-Jan-2025'])
        
        Returns:
            Bias report
        """
        results = {}
        
        for format_variant in date_formats:
            result = analysis_function(format_variant)
            results[format_variant] = result
        
        # Check consistency
        parsed_dates = [r.get('parsed_date') for r in results.values()]
        consistent = len(set(str(d) for d in parsed_dates if d)) == 1
        
        bias_score = 0.0 if consistent else 0.15
        
        report = {
            'test': 'date_format_bias',
            'timestamp': datetime.now().isoformat(),
            'date_formats': date_formats,
            'results': results,
            'consistent': consistent,
            'bias_score': bias_score,
            'bias_level': 'LOW' if bias_score < 0.05 else 'MEDIUM',
            'passed': consistent
        }
        
        logger.info(f"Date format bias test: {'PASSED' if consistent else 'FAILED'}")
        
        return report

    def calculate_overall_bias(
        self,
        test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall bias score from multiple tests
        
        Args:
            test_results: List of individual bias test results
        
        Returns:
            Overall bias assessment
        """
        if not test_results:
            return {'overall_bias_score': 0.0, 'bias_level': 'UNKNOWN'}
        
        # Average bias scores
        total_bias = sum(t.get('bias_score', 0) for t in test_results)
        overall_bias = total_bias / len(test_results)
        
        # Count passed tests
        passed = sum(1 for t in test_results if t.get('passed', False))
        total = len(test_results)
        pass_rate = passed / total if total > 0 else 0
        
        # Determine level
        if overall_bias < 0.02:
            bias_level = 'VERY LOW'
        elif overall_bias < 0.05:
            bias_level = 'LOW'
        elif overall_bias < 0.10:
            bias_level = 'MEDIUM'
        else:
            bias_level = 'HIGH'
        
        assessment = {
            'overall_bias_score': round(overall_bias, 4),
            'bias_level': bias_level,
            'tests_conducted': total,
            'tests_passed': passed,
            'pass_rate': round(pass_rate, 4),
            'meets_target': overall_bias < 0.02,  # Target: < 2% bias
            'individual_tests': test_results
        }
        
        logger.info(f"Overall bias: {overall_bias:.2%} ({bias_level})")
        
        return assessment

    def generate_bias_report(
        self,
        test_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate formatted bias detection report
        
        Args:
            test_results: List of bias test results
        
        Returns:
            Formatted report string
        """
        assessment = self.calculate_overall_bias(test_results)
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("BIAS DETECTION REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("OVERALL ASSESSMENT")
        report_lines.append("-"*60)
        report_lines.append(f"Overall Bias Score: {assessment['overall_bias_score']:.2%}")
        report_lines.append(f"Bias Level: {assessment['bias_level']}")
        report_lines.append(f"Tests Passed: {assessment['tests_passed']}/{assessment['tests_conducted']}")
        report_lines.append(f"Pass Rate: {assessment['pass_rate']:.1%}")
        report_lines.append(f"Meets Target (< 2%): {'✓ YES' if assessment['meets_target'] else '✗ NO'}")
        report_lines.append("")
        
        report_lines.append("INDIVIDUAL TEST RESULTS")
        report_lines.append("-"*60)
        
        for i, test in enumerate(test_results, 1):
            report_lines.append(f"\n{i}. {test.get('test', 'Unknown Test').replace('_', ' ').title()}")
            report_lines.append(f"   Bias Score: {test.get('bias_score', 0):.2%}")
            report_lines.append(f"   Bias Level: {test.get('bias_level', 'UNKNOWN')}")
            report_lines.append(f"   Status: {'✓ PASSED' if test.get('passed', False) else '✗ FAILED'}")
            
            if not test.get('consistent', True):
                report_lines.append(f"   ⚠ Inconsistent behavior detected")
        
        report_lines.append("")
        report_lines.append("="*60)
        
        if assessment['meets_target']:
            report_lines.append("✓ System meets bias target (< 2%)")
        else:
            report_lines.append("✗ System does NOT meet bias target")
            report_lines.append("  Recommendations:")
            report_lines.append("  - Improve name normalization")
            report_lines.append("  - Standardize format parsing")
            report_lines.append("  - Add fuzzy matching for variations")
        
        report_lines.append("="*60)
        
        report = "\n".join(report_lines)
        
        # Store in history
        self.bias_reports.append({
            'timestamp': datetime.now().isoformat(),
            'assessment': assessment,
            'report': report
        })
        
        return report

    def export_bias_report(
        self,
        test_results: List[Dict[str, Any]],
        output_file: str
    ):
        """Export bias report to JSON file"""
        assessment = self.calculate_overall_bias(test_results)
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_assessment': assessment,
            'individual_tests': test_results,
            'formatted_report': self.generate_bias_report(test_results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Exported bias report to {output_file}")

    def export_history(self, filepath: str):
        """Export bias detection history"""
        with open(filepath, 'w') as f:
            json.dump(self.bias_reports, f, indent=2, default=str)
        
        logger.info(f"Exported bias detection history to {filepath}")
