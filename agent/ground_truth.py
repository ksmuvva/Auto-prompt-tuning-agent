"""
Ground Truth Manager
Manages the master ground truth file and validates predictions against it.
This file is NEVER exposed to the LLM - it's used only for validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundTruthManager:
    """
    Manages ground truth data for validation
    
    This is the "answer key" that the AI should never see.
    Used only for calculating precision, accuracy, and other metrics.
    """

    def __init__(self, ground_truth_file: str = "data/ground_truth_master.json"):
        self.ground_truth_file = Path(ground_truth_file)
        self.ground_truth = None
        self.load_ground_truth()

    def load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth from file"""
        if not self.ground_truth_file.exists():
            logger.error(f"Ground truth file not found: {self.ground_truth_file}")
            return {}

        try:
            with open(self.ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
            
            logger.info(f"Loaded ground truth: {self.ground_truth_file}")
            logger.info(f"  Coverage: {len(self.ground_truth.get('fw15_high_value', []))} high-value transactions")
            
            return self.ground_truth
        
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return {}

    def get_ground_truth_for_requirement(self, requirement: str) -> List[Dict]:
        """Get ground truth data for a specific FW requirement"""
        if not self.ground_truth:
            return []
        
        requirement_key = f"fw{requirement}_" if not requirement.startswith('fw') else requirement
        
        # Map requirement names to keys
        requirement_map = {
            'fw15': 'fw15_high_value',
            'fw20_luxury': 'fw20_luxury_brands',
            'fw20_transfer': 'fw20_money_transfers',
            'fw25': 'fw25_missing_audit',
            'fw30': 'fw30_missing_months',
            'fw40': 'fw40_errors',
            'fw45': 'fw45_gambling',
            'fw50': 'fw50_debt_payments'
        }
        
        key = requirement_map.get(requirement, requirement)
        return self.ground_truth.get(key, [])

    def calculate_precision(self, true_positives: int, false_positives: int) -> float:
        """
        Calculate precision
        Precision = TP / (TP + FP)
        """
        if true_positives + false_positives == 0:
            return 1.0  # Perfect if no predictions made and no ground truth
        
        precision = true_positives / (true_positives + false_positives)
        return round(precision, 4)

    def calculate_recall(self, true_positives: int, false_negatives: int) -> float:
        """
        Calculate recall (sensitivity)
        Recall = TP / (TP + FN)
        """
        if true_positives + false_negatives == 0:
            return 1.0
        
        recall = true_positives / (true_positives + false_negatives)
        return round(recall, 4)

    def calculate_accuracy(self, true_positives: int, true_negatives: int,
                          false_positives: int, false_negatives: int) -> float:
        """
        Calculate accuracy
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        total = true_positives + true_negatives + false_positives + false_negatives
        
        if total == 0:
            return 1.0
        
        accuracy = (true_positives + true_negatives) / total
        return round(accuracy, 4)

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score (harmonic mean of precision and recall)
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1, 4)

    def validate_fw15_high_value(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Validate FW15: High-value transactions (>£250)
        
        Args:
            predictions: List of predicted high-value transactions
        
        Returns:
            Validation metrics including precision, recall, accuracy
        """
        ground_truth_ids = set(
            t['transaction_id'] for t in self.ground_truth.get('fw15_high_value', [])
        )
        
        predicted_ids = set(
            p.get('transaction_id', '') for p in predictions
        )
        
        # Calculate confusion matrix
        true_positives = len(ground_truth_ids.intersection(predicted_ids))
        false_positives = len(predicted_ids - ground_truth_ids)
        false_negatives = len(ground_truth_ids - predicted_ids)
        
        # For accuracy, we need true negatives
        # Assuming total possible transactions is 3000
        total_transactions = 3000
        true_negatives = total_transactions - len(ground_truth_ids) - false_positives
        
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)
        accuracy = self.calculate_accuracy(true_positives, true_negatives, 
                                           false_positives, false_negatives)
        f1 = self.calculate_f1_score(precision, recall)
        
        return {
            'requirement': 'FW15',
            'description': 'High-value transactions (>£250)',
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1,
            'ground_truth_count': len(ground_truth_ids),
            'predicted_count': len(predicted_ids),
            'meets_target_precision': precision >= 0.98,
            'meets_target_accuracy': accuracy >= 0.98,
            'passed': precision >= 0.98 and accuracy >= 0.98
        }

    def validate_fw20_luxury_brands(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Validate FW20: Luxury brand detection"""
        ground_truth_ids = set(
            t['transaction_id'] for t in self.ground_truth.get('fw20_luxury_brands', [])
        )
        
        predicted_ids = set(
            p.get('transaction_id', '') for p in predictions
        )
        
        true_positives = len(ground_truth_ids.intersection(predicted_ids))
        false_positives = len(predicted_ids - ground_truth_ids)
        false_negatives = len(ground_truth_ids - predicted_ids)
        
        total_transactions = 3000
        true_negatives = total_transactions - len(ground_truth_ids) - false_positives
        
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)
        accuracy = self.calculate_accuracy(true_positives, true_negatives,
                                           false_positives, false_negatives)
        f1 = self.calculate_f1_score(precision, recall)
        
        return {
            'requirement': 'FW20-Luxury',
            'description': 'Luxury brand purchases',
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1,
            'ground_truth_count': len(ground_truth_ids),
            'predicted_count': len(predicted_ids),
            'meets_target_precision': precision >= 0.98,
            'meets_target_accuracy': accuracy >= 0.98,
            'passed': precision >= 0.98 and accuracy >= 0.98
        }

    def validate_fw25_missing_audit(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Validate FW25: Missing audit trail detection"""
        ground_truth_ids = set(
            t['transaction_id'] for t in self.ground_truth.get('fw25_missing_audit', [])
        )
        
        predicted_ids = set(
            p.get('transaction_id', '') for p in predictions
        )
        
        true_positives = len(ground_truth_ids.intersection(predicted_ids))
        false_positives = len(predicted_ids - ground_truth_ids)
        false_negatives = len(ground_truth_ids - predicted_ids)
        
        total_transactions = 3000
        true_negatives = total_transactions - len(ground_truth_ids) - false_positives
        
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)
        accuracy = self.calculate_accuracy(true_positives, true_negatives,
                                           false_positives, false_negatives)
        f1 = self.calculate_f1_score(precision, recall)
        
        return {
            'requirement': 'FW25',
            'description': 'Missing audit trail',
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1,
            'ground_truth_count': len(ground_truth_ids),
            'predicted_count': len(predicted_ids),
            'meets_target_precision': precision >= 0.98,
            'meets_target_accuracy': accuracy >= 0.98,
            'passed': precision >= 0.98 and accuracy >= 0.98
        }

    def validate_fw30_missing_months(self, predicted_missing_months: List[str]) -> Dict[str, Any]:
        """
        Validate FW30: Missing months detection
        
        Args:
            predicted_missing_months: List of month strings (e.g., ['2025-03', '2025-06'])
        """
        ground_truth_months = set(self.ground_truth.get('fw30_missing_months', []))
        predicted_months = set(predicted_missing_months)
        
        # Exact match for missing months
        correctly_identified = ground_truth_months.intersection(predicted_months)
        incorrectly_identified = predicted_months - ground_truth_months
        missed_months = ground_truth_months - predicted_months
        
        # For missing months, precision and recall are straightforward
        true_positives = len(correctly_identified)
        false_positives = len(incorrectly_identified)
        false_negatives = len(missed_months)
        
        # True negatives: months that exist and were not flagged as missing
        # In a 12-month period, we have 12 - len(ground_truth_months) = non-missing months
        total_possible_months = 12
        true_negatives = total_possible_months - len(ground_truth_months) - false_positives
        
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)
        accuracy = self.calculate_accuracy(true_positives, true_negatives,
                                           false_positives, false_negatives)
        f1 = self.calculate_f1_score(precision, recall)
        
        # For FW30, perfect detection is critical
        perfect_match = (ground_truth_months == predicted_months)
        
        return {
            'requirement': 'FW30',
            'description': 'Missing months detection',
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1,
            'perfect_match': perfect_match,
            'ground_truth_missing': list(ground_truth_months),
            'predicted_missing': list(predicted_months),
            'correctly_identified': list(correctly_identified),
            'incorrectly_identified': list(incorrectly_identified),
            'missed_months': list(missed_months),
            'meets_target_precision': precision >= 0.98,
            'meets_target_accuracy': accuracy >= 0.98,
            'passed': precision >= 0.98 and accuracy >= 0.98 and perfect_match
        }

    def validate_fw45_gambling(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Validate FW45: Gambling transaction detection"""
        ground_truth_ids = set(
            t['transaction_id'] for t in self.ground_truth.get('fw45_gambling', [])
        )
        
        predicted_ids = set(
            p.get('transaction_id', '') for p in predictions
        )
        
        true_positives = len(ground_truth_ids.intersection(predicted_ids))
        false_positives = len(predicted_ids - ground_truth_ids)
        false_negatives = len(ground_truth_ids - predicted_ids)
        
        total_transactions = 3000
        true_negatives = total_transactions - len(ground_truth_ids) - false_positives
        
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)
        accuracy = self.calculate_accuracy(true_positives, true_negatives,
                                           false_positives, false_negatives)
        f1 = self.calculate_f1_score(precision, recall)
        
        # Calculate total gambling spend
        ground_truth_total = sum(
            t['amount'] for t in self.ground_truth.get('fw45_gambling', [])
        )
        predicted_total = sum(p.get('amount', 0) for p in predictions)
        
        return {
            'requirement': 'FW45',
            'description': 'Gambling transactions',
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1,
            'ground_truth_count': len(ground_truth_ids),
            'predicted_count': len(predicted_ids),
            'ground_truth_total': ground_truth_total,
            'predicted_total': predicted_total,
            'amount_difference': abs(ground_truth_total - predicted_total),
            'meets_target_precision': precision >= 0.98,
            'meets_target_accuracy': accuracy >= 0.98,
            'passed': precision >= 0.98 and accuracy >= 0.98
        }

    def validate_fw50_debt_payments(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Validate FW50: Large debt payments (≥£500)"""
        ground_truth_ids = set(
            t['transaction_id'] for t in self.ground_truth.get('fw50_debt_payments', [])
        )
        
        predicted_ids = set(
            p.get('transaction_id', '') for p in predictions
        )
        
        true_positives = len(ground_truth_ids.intersection(predicted_ids))
        false_positives = len(predicted_ids - ground_truth_ids)
        false_negatives = len(ground_truth_ids - predicted_ids)
        
        total_transactions = 3000
        true_negatives = total_transactions - len(ground_truth_ids) - false_positives
        
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)
        accuracy = self.calculate_accuracy(true_positives, true_negatives,
                                           false_positives, false_negatives)
        f1 = self.calculate_f1_score(precision, recall)
        
        return {
            'requirement': 'FW50',
            'description': 'Large debt payments (≥£500)',
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1,
            'ground_truth_count': len(ground_truth_ids),
            'predicted_count': len(predicted_ids),
            'meets_target_precision': precision >= 0.98,
            'meets_target_accuracy': accuracy >= 0.98,
            'passed': precision >= 0.98 and accuracy >= 0.98
        }

    def generate_validation_report(self, all_validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for all requirements
        
        Args:
            all_validations: List of validation results from each FW requirement
        
        Returns:
            Comprehensive report with overall metrics
        """
        # Calculate aggregate metrics
        total_precision = sum(v['precision'] for v in all_validations) / len(all_validations)
        total_accuracy = sum(v['accuracy'] for v in all_validations) / len(all_validations)
        total_recall = sum(v['recall'] for v in all_validations) / len(all_validations)
        total_f1 = sum(v['f1_score'] for v in all_validations) / len(all_validations)
        
        # Count how many requirements passed
        passed_count = sum(1 for v in all_validations if v.get('passed', False))
        total_count = len(all_validations)
        
        # Determine if overall system meets targets
        meets_precision_target = total_precision >= 0.98
        meets_accuracy_target = total_accuracy >= 0.98
        all_requirements_passed = passed_count == total_count
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_requirements_tested': total_count,
                'requirements_passed': passed_count,
                'requirements_failed': total_count - passed_count,
                'pass_rate': round(passed_count / total_count, 4) if total_count > 0 else 0
            },
            'aggregate_metrics': {
                'precision': round(total_precision, 4),
                'accuracy': round(total_accuracy, 4),
                'recall': round(total_recall, 4),
                'f1_score': round(total_f1, 4)
            },
            'targets': {
                'precision_target': 0.98,
                'accuracy_target': 0.98,
                'meets_precision_target': meets_precision_target,
                'meets_accuracy_target': meets_accuracy_target,
                'all_requirements_passed': all_requirements_passed
            },
            'individual_results': all_validations,
            'overall_status': 'PASSED' if all_requirements_passed and meets_precision_target and meets_accuracy_target else 'FAILED'
        }
        
        return report

    def export_validation_report(self, report: Dict[str, Any], output_file: str = None):
        """Export validation report to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"results/validation_report_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report exported to: {output_path}")
        return str(output_path)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from ground truth"""
        if not self.ground_truth:
            return {}
        
        return {
            'fw15_high_value_count': len(self.ground_truth.get('fw15_high_value', [])),
            'fw20_luxury_count': len(self.ground_truth.get('fw20_luxury_brands', [])),
            'fw20_transfers_count': len(self.ground_truth.get('fw20_money_transfers', [])),
            'fw25_missing_audit_count': len(self.ground_truth.get('fw25_missing_audit', [])),
            'fw30_missing_months': self.ground_truth.get('fw30_missing_months', []),
            'fw40_errors_count': len(self.ground_truth.get('fw40_errors', [])),
            'fw45_gambling_count': len(self.ground_truth.get('fw45_gambling', [])),
            'fw50_debt_count': len(self.ground_truth.get('fw50_debt_payments', [])),
            'metadata': self.ground_truth.get('metadata', {})
        }
