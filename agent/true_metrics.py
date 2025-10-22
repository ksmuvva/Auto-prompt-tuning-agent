"""
TRUE Mathematical Metrics Calculator
Compares LLM output against ground truth with precise mathematical calculations
No approximations - exact precision, recall, accuracy, F1
"""

import json
import re
import logging
from typing import Dict, Any, List, Set, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrueMetricsCalculator:
    """
    Calculates TRUE metrics by comparing LLM output against ground truth

    Uses exact mathematical formulas:
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """

    def __init__(self):
        self.history = []

    def extract_transaction_ids_from_llm_response(self, response: str) -> Set[str]:
        """
        Extract transaction IDs from LLM response text

        Handles multiple formats:
        - Transaction ID: TXN_001
        - ID: TXN_001
        - TXN_001
        - Transaction TXN_001
        """
        ids = set()

        # Try JSON parsing first
        try:
            json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', response)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, dict):
                    # Look for transaction IDs in dict
                    if 'high_value_transactions' in data:
                        txns = data['high_value_transactions']
                        if isinstance(txns, list):
                            for txn in txns:
                                if isinstance(txn, dict) and 'transaction_id' in txn:
                                    ids.add(txn['transaction_id'])
                                elif isinstance(txn, str):
                                    ids.add(txn)
                    # Look for any ID fields
                    for key, value in data.items():
                        if 'id' in key.lower() and isinstance(value, (list, set)):
                            ids.update(str(v) for v in value)
                elif isinstance(data, list):
                    # List of IDs or transaction objects
                    for item in data:
                        if isinstance(item, dict) and 'transaction_id' in item:
                            ids.add(item['transaction_id'])
                        elif isinstance(item, str):
                            ids.add(item)
        except (json.JSONDecodeError, KeyError):
            pass

        # Regex patterns for transaction IDs
        patterns = [
            r'(?:transaction[_\s-]*id|id)[:\s]*([A-Z0-9_-]+)',  # Transaction ID: TXN_001
            r'\b(TXN_\d+)\b',  # TXN_001
            r'\b(TRANS_\d+)\b',  # TRANS_001
            r'\b([A-Z]+_\d+)\b',  # Any PREFIX_NUMBER format
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            ids.update(match if isinstance(match, str) else match[0] for match in matches)

        logger.debug(f"Extracted {len(ids)} transaction IDs from LLM response")
        return ids

    def calculate_confusion_matrix(
        self,
        predicted_ids: Set[str],
        ground_truth_ids: Set[str],
        total_population: int
    ) -> Dict[str, int]:
        """
        Calculate true confusion matrix values

        Args:
            predicted_ids: IDs predicted by LLM as positive cases
            ground_truth_ids: True positive cases from ground truth
            total_population: Total number of transactions

        Returns:
            Dict with TP, TN, FP, FN counts
        """
        # True Positives: Predicted positive AND actually positive
        true_positives = len(predicted_ids.intersection(ground_truth_ids))

        # False Positives: Predicted positive BUT actually negative
        false_positives = len(predicted_ids - ground_truth_ids)

        # False Negatives: Predicted negative BUT actually positive
        false_negatives = len(ground_truth_ids - predicted_ids)

        # True Negatives: Predicted negative AND actually negative
        # Total negatives = total - ground truth positives
        total_negatives = total_population - len(ground_truth_ids)
        true_negatives = total_negatives - false_positives

        return {
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def calculate_metrics(
        self,
        llm_response: str,
        ground_truth: Dict[str, Any],
        total_transactions: int
    ) -> Dict[str, Any]:
        """
        Calculate ALL metrics with TRUE mathematical precision

        Args:
            llm_response: Raw LLM output text
            ground_truth: Ground truth data with transaction IDs
            total_transactions: Total number of transactions in dataset

        Returns:
            Complete metrics including precision, recall, accuracy, F1, confusion matrix
        """
        # Extract predicted IDs from LLM response
        predicted_ids = self.extract_transaction_ids_from_llm_response(llm_response)

        # Get ground truth IDs
        ground_truth_ids = set(ground_truth.get('high_value_transactions', []))

        # Calculate confusion matrix
        confusion = self.calculate_confusion_matrix(
            predicted_ids,
            ground_truth_ids,
            total_transactions
        )

        tp = confusion['true_positives']
        tn = confusion['true_negatives']
        fp = confusion['false_positives']
        fn = confusion['false_negatives']

        # TRUE MATHEMATICAL CALCULATIONS

        # Precision = TP / (TP + FP) - Of all predicted positives, how many were correct?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN) - Of all actual positives, how many did we find?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Accuracy = (TP + TN) / (TP + TN + FP + FN) - Overall correctness
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Specificity = TN / (TN + FP) - True negative rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Identify specific failures for prompt improvement
        false_positive_ids = list(predicted_ids - ground_truth_ids)
        false_negative_ids = list(ground_truth_ids - predicted_ids)

        metrics = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1_score, 4),
            'specificity': round(specificity, 4),

            'confusion_matrix': confusion,

            'predicted_count': len(predicted_ids),
            'ground_truth_count': len(ground_truth_ids),
            'total_transactions': total_transactions,

            'false_positives': false_positive_ids[:20],  # Limit for display
            'false_negatives': false_negative_ids[:20],

            'meets_98_percent_target': precision >= 0.98 and accuracy >= 0.98,

            'timestamp': datetime.now().isoformat()
        }

        # Store in history
        self.history.append(metrics)

        logger.info(f"TRUE METRICS: P={precision:.2%}, R={recall:.2%}, A={accuracy:.2%}, F1={f1_score:.2%}")
        logger.info(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

        if not metrics['meets_98_percent_target']:
            logger.warning(f"Target not met! Precision: {precision:.2%}, Accuracy: {accuracy:.2%}")
            logger.warning(f"FP={len(false_positive_ids)}, FN={len(false_negative_ids)}")

        return metrics

    def compare_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict[str, Any]:
        """Compare two metric sets to see improvement"""
        improvements = {
            'precision_delta': metrics2['precision'] - metrics1['precision'],
            'recall_delta': metrics2['recall'] - metrics1['recall'],
            'accuracy_delta': metrics2['accuracy'] - metrics1['accuracy'],
            'f1_delta': metrics2['f1_score'] - metrics1['f1_score'],
            'improved': (
                metrics2['precision'] > metrics1['precision'] and
                metrics2['accuracy'] > metrics1['accuracy']
            )
        }
        return improvements

    def generate_failure_report(self, metrics: Dict[str, Any]) -> str:
        """Generate detailed failure analysis for prompt tuning"""
        report = f"""
FAILURE ANALYSIS REPORT
========================

CURRENT METRICS:
- Precision: {metrics['precision']:.2%} (Target: 98%)
- Accuracy: {metrics['accuracy']:.2%} (Target: 98%)
- Recall: {metrics['recall']:.2%}
- F1 Score: {metrics['f1_score']:.2%}

CONFUSION MATRIX:
- True Positives: {metrics['confusion_matrix']['true_positives']}
- True Negatives: {metrics['confusion_matrix']['true_negatives']}
- False Positives: {metrics['confusion_matrix']['false_positives']}
- False Negatives: {metrics['confusion_matrix']['false_negatives']}

FAILURES DETECTED:
- False Positives (incorrectly identified): {len(metrics['false_positives'])}
  Examples: {', '.join(metrics['false_positives'][:5])}

- False Negatives (missed): {len(metrics['false_negatives'])}
  Examples: {', '.join(metrics['false_negatives'][:5])}

RECOMMENDED IMPROVEMENTS:
"""

        if metrics['confusion_matrix']['false_positives'] > 0:
            report += "- Add stricter criteria to reduce false positives\n"
            report += "- Include explicit exclusion rules\n"

        if metrics['confusion_matrix']['false_negatives'] > 0:
            report += "- Broaden detection criteria to catch missed cases\n"
            report += "- Add edge case handling\n"

        if metrics['precision'] < 0.98:
            report += f"- CRITICAL: Precision {metrics['precision']:.2%} below 98% target\n"

        if metrics['accuracy'] < 0.98:
            report += f"- CRITICAL: Accuracy {metrics['accuracy']:.2%} below 98% target\n"

        return report
