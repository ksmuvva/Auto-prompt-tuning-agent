"""
Metrics Evaluation System
Evaluates LLM prompt performance using multiple metrics
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptMetrics:
    """Comprehensive metrics for evaluating prompt performance"""

    def __init__(self):
        self.metrics_history = []

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        parsed = {
            'high_value_transactions': [],
            'anomalies': [],
            'raw_response': response
        }

        # Try to extract JSON if present
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                if 'high_value_transactions' in json_data:
                    parsed['high_value_transactions'] = json_data['high_value_transactions']
                if 'anomalies' in json_data:
                    parsed['anomalies'] = json_data['anomalies']
                return parsed
            except json.JSONDecodeError:
                pass

        # Extract transaction IDs and amounts using regex
        # Pattern: Transaction ID or number followed by amount
        amount_pattern = r'(?:£|GBP)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        id_pattern = r'(?:transaction|id|#)\s*[:]?\s*(\w+)'

        amounts = re.findall(amount_pattern, response)
        ids = re.findall(id_pattern, response, re.IGNORECASE)

        # Create basic extracted data
        for amount in amounts:
            amount_clean = float(amount.replace(',', ''))
            parsed['high_value_transactions'].append({
                'amount': amount_clean
            })

        # Count anomaly mentions
        anomaly_keywords = ['anomaly', 'unusual', 'suspicious', 'fraud', 'irregular']
        anomaly_count = sum(response.lower().count(keyword) for keyword in anomaly_keywords)
        parsed['anomalies'] = [{'detected': True}] * min(anomaly_count, 10)  # Cap at 10

        return parsed

    def calculate_accuracy(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """Calculate accuracy of high-value transaction detection"""
        if not ground_truth.get('high_value_transactions'):
            return 0.0

        # Extract IDs from predicted (could be list of dicts or list of IDs)
        predicted_transactions = predicted.get('high_value_transactions', [])
        if predicted_transactions and isinstance(predicted_transactions[0], dict):
            predicted_ids = set(t.get('transaction_id', str(i)) for i, t in enumerate(predicted_transactions))
        else:
            predicted_ids = set(predicted_transactions)

        # Extract IDs from ground truth
        true_ids = set(ground_truth['high_value_transactions'])

        if not true_ids:
            return 1.0 if not predicted_ids else 0.0

        # Calculate intersection
        correct = len(predicted_ids.intersection(true_ids))
        total = len(true_ids)

        return correct / total if total > 0 else 0.0

    def calculate_precision_recall_f1(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        predicted_count = predicted.get('count_above_threshold', 0)
        true_count = ground_truth.get('count_above_threshold', 0)

        # Extract counts from parsed response
        if 'high_value_transactions' in predicted:
            predicted_count = len(predicted['high_value_transactions'])

        true_positives = min(predicted_count, true_count)
        false_positives = max(0, predicted_count - true_count)
        false_negatives = max(0, true_count - predicted_count)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def calculate_completeness(self, response: str, ground_truth: Dict[str, Any]) -> float:
        """Measure how complete the response is"""
        required_elements = [
            'high_value_transactions',
            'anomalies',
            'summary'
        ]

        response_lower = response.lower()
        found_elements = sum(1 for elem in required_elements if elem.replace('_', ' ') in response_lower)

        completeness = found_elements / len(required_elements)
        return completeness

    def calculate_format_quality(self, response: str) -> float:
        """Evaluate response formatting and structure"""
        quality_score = 0.0
        max_score = 5.0

        # Check for structured sections
        if re.search(r'(HIGH[\s-]*VALUE|TRANSACTIONS)', response, re.IGNORECASE):
            quality_score += 1.0

        if re.search(r'(ANOMAL|IRREGULAR|SUSPICIOUS)', response, re.IGNORECASE):
            quality_score += 1.0

        if re.search(r'(SUMMARY|CONCLUSION)', response, re.IGNORECASE):
            quality_score += 1.0

        # Check for bullet points or numbering
        if re.search(r'(^|\n)\s*[-*•]\s', response) or re.search(r'(^|\n)\s*\d+\.', response):
            quality_score += 1.0

        # Check for specific amounts
        if re.search(r'£?\s*\d+(?:,\d{3})*(?:\.\d{2})?', response):
            quality_score += 1.0

        return quality_score / max_score

    def calculate_specificity(self, response: str) -> float:
        """Measure how specific and detailed the response is"""
        specificity_score = 0.0
        max_score = 4.0

        # Check for transaction IDs
        if re.search(r'(transaction|id|#)\s*[:]?\s*\w+', response, re.IGNORECASE):
            specificity_score += 1.0

        # Check for specific amounts
        amounts = re.findall(r'£?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', response)
        if len(amounts) >= 3:
            specificity_score += 1.0

        # Check for dates
        if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', response):
            specificity_score += 1.0

        # Check for reasoning/explanations
        reasoning_words = ['because', 'due to', 'indicates', 'suggests', 'pattern']
        if any(word in response.lower() for word in reasoning_words):
            specificity_score += 1.0

        return specificity_score / max_score

    def calculate_efficiency_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate efficiency metrics (tokens, latency)"""
        return {
            'tokens_used': result.get('tokens_used', 0),
            'latency': result.get('latency', 0),
            'tokens_per_second': result.get('tokens_used', 0) / result.get('latency', 1) if result.get('latency', 0) > 0 else 0
        }

    def evaluate_prompt(
        self,
        prompt_name: str,
        llm_result: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a prompt's performance

        Returns a metrics dictionary with all scores
        """
        if not llm_result.get('success'):
            logger.warning(f"LLM call failed for prompt: {prompt_name}")
            return {
                'prompt_name': prompt_name,
                'success': False,
                'error': llm_result.get('error', 'Unknown error')
            }

        response = llm_result.get('response', '')

        # Parse the response
        parsed_response = self.parse_llm_response(response)

        # Calculate all metrics
        accuracy = self.calculate_accuracy(parsed_response, ground_truth)
        prf_metrics = self.calculate_precision_recall_f1(parsed_response, ground_truth)
        completeness = self.calculate_completeness(response, ground_truth)
        format_quality = self.calculate_format_quality(response)
        specificity = self.calculate_specificity(response)
        efficiency = self.calculate_efficiency_metrics(llm_result)

        # Calculate composite score (weighted average)
        composite_score = (
            accuracy * 0.30 +
            prf_metrics['f1_score'] * 0.25 +
            completeness * 0.15 +
            format_quality * 0.15 +
            specificity * 0.15
        )

        metrics = {
            'prompt_name': prompt_name,
            'success': True,
            'timestamp': datetime.now().isoformat(),

            # Accuracy metrics
            'accuracy': accuracy,
            'precision': prf_metrics['precision'],
            'recall': prf_metrics['recall'],
            'f1_score': prf_metrics['f1_score'],

            # Quality metrics
            'completeness': completeness,
            'format_quality': format_quality,
            'specificity': specificity,

            # Efficiency metrics
            'tokens_used': efficiency['tokens_used'],
            'latency': efficiency['latency'],
            'tokens_per_second': efficiency['tokens_per_second'],

            # Overall score
            'composite_score': composite_score,

            # Parsed data
            'parsed_response': parsed_response,
            'response_length': len(response)
        }

        # Store in history
        self.metrics_history.append(metrics)

        logger.info(f"Evaluated {prompt_name}: Composite Score = {composite_score:.3f}")

        return metrics

    def compare_prompts(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple prompt performances"""
        if not metrics_list:
            return {}

        # Sort by composite score
        sorted_metrics = sorted(
            metrics_list,
            key=lambda x: x.get('composite_score', 0),
            reverse=True
        )

        best_prompt = sorted_metrics[0]
        worst_prompt = sorted_metrics[-1]

        comparison = {
            'total_prompts_evaluated': len(metrics_list),
            'best_prompt': {
                'name': best_prompt['prompt_name'],
                'composite_score': best_prompt['composite_score'],
                'f1_score': best_prompt['f1_score'],
                'accuracy': best_prompt['accuracy']
            },
            'worst_prompt': {
                'name': worst_prompt['prompt_name'],
                'composite_score': worst_prompt['composite_score']
            },
            'average_scores': {
                'composite_score': np.mean([m['composite_score'] for m in metrics_list]),
                'f1_score': np.mean([m['f1_score'] for m in metrics_list]),
                'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
                'latency': np.mean([m['latency'] for m in metrics_list])
            },
            'rankings': [
                {
                    'rank': i + 1,
                    'name': m['prompt_name'],
                    'score': m['composite_score']
                }
                for i, m in enumerate(sorted_metrics)
            ]
        }

        return comparison

    def get_improvement_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving prompt performance"""
        suggestions = []

        if metrics.get('accuracy', 0) < 0.7:
            suggestions.append("Low accuracy: Add more specific instructions for identifying transactions")

        if metrics.get('completeness', 0) < 0.7:
            suggestions.append("Incomplete response: Explicitly request all required sections")

        if metrics.get('format_quality', 0) < 0.6:
            suggestions.append("Poor formatting: Request structured output (bullets, tables, or JSON)")

        if metrics.get('specificity', 0) < 0.6:
            suggestions.append("Low specificity: Ask for transaction IDs, dates, and amounts")

        if metrics.get('f1_score', 0) < 0.6:
            suggestions.append("Low F1 score: Provide examples or use few-shot learning")

        if not suggestions:
            suggestions.append("Performance is good! Consider minor optimizations for efficiency")

        return suggestions

    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        logger.info(f"Exported {len(self.metrics_history)} metric records to {filepath}")
