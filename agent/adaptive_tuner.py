"""
Adaptive Prompt Tuning Engine
Uses LLM to dynamically generate and optimize prompts based on real metrics
Iterates until 98% precision/accuracy targets are met
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from agent.true_metrics import TrueMetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptivePromptTuner:
    """
    Truly adaptive AI agent that:
    1. Tests initial prompt
    2. Calculates TRUE metrics
    3. Uses LLM to generate improved prompt based on failures
    4. Tests new prompt
    5. Repeats until 98% targets met or max iterations reached
    """

    def __init__(self, llm_service, max_iterations: int = 10):
        self.llm_service = llm_service
        self.metrics_calculator = TrueMetricsCalculator()
        self.max_iterations = max_iterations
        self.tuning_history = []

    def generate_initial_prompt(self, requirement: str, context: Dict[str, Any]) -> str:
        """Generate initial prompt for a requirement using LLM"""
        meta_prompt = f"""You are an expert prompt engineer for financial data analysis.

TASK: Create a prompt to analyze bank transaction data for: {requirement}

CONTEXT:
{json.dumps(context, indent=2)}

REQUIREMENTS:
- Must achieve 98% precision and 98% accuracy
- Output should include transaction IDs in a parseable format
- Be specific and unambiguous
- Include validation criteria

Generate a prompt that will accurately identify transactions matching this requirement.
Return ONLY the prompt text."""

        result = self.llm_service.generate(meta_prompt)
        if result.get('success'):
            return result['response'].strip()
        else:
            # Fallback to template
            return f"Analyze the following bank transactions and identify those matching: {requirement}"

    def improve_prompt_from_failures(
        self,
        current_prompt: str,
        metrics: Dict[str, Any],
        requirement: str,
        iteration: int
    ) -> str:
        """Use LLM to generate improved prompt based on failure analysis"""

        failure_report = self.metrics_calculator.generate_failure_report(metrics)

        meta_prompt = f"""You are an AI prompt optimization expert. Your goal is to improve a prompt to achieve 98% precision and 98% accuracy.

CURRENT PROMPT (Iteration {iteration}):
{current_prompt}

{failure_report}

REQUIREMENT: {requirement}

TASK: Generate an IMPROVED prompt that:
1. Reduces false positives by {metrics['confusion_matrix']['false_positives']}
2. Reduces false negatives by {metrics['confusion_matrix']['false_negatives']}
3. Achieves precision ≥ 98% and accuracy ≥ 98%
4. Outputs transaction IDs in a clear, parseable format (JSON or list)
5. Is more specific and handles edge cases

STRATEGIES TO CONSIDER:
- Add explicit exclusion criteria if false positives are high
- Broaden criteria if false negatives are high
- Use specific thresholds and rules
- Request structured output format
- Add validation steps

Generate the IMPROVED prompt. Return ONLY the new prompt text, no explanations."""

        result = self.llm_service.generate(meta_prompt)
        if result.get('success'):
            improved = result['response'].strip()
            logger.info(f"Generated improved prompt (iteration {iteration})")
            return improved
        else:
            logger.warning("Failed to generate improved prompt, using current prompt")
            return current_prompt

    def test_prompt(
        self,
        prompt: str,
        data: str,
        ground_truth: Dict[str, Any],
        total_transactions: int
    ) -> Dict[str, Any]:
        """Test a prompt and return TRUE metrics"""
        logger.info(f"Testing prompt...")

        # Generate LLM response using the prompt
        full_prompt = f"{prompt}\n\nDATA:\n{data}"
        result = self.llm_service.generate(full_prompt)

        if not result.get('success'):
            logger.error("LLM generation failed")
            return {
                'success': False,
                'error': 'LLM generation failed',
                'metrics': self._empty_metrics()
            }

        llm_response = result['response']

        # Calculate TRUE metrics
        metrics = self.metrics_calculator.calculate_metrics(
            llm_response,
            ground_truth,
            total_transactions
        )

        return {
            'success': True,
            'prompt': prompt,
            'llm_response': llm_response,
            'metrics': metrics
        }

    def adaptive_tune(
        self,
        requirement: str,
        data: str,
        ground_truth: Dict[str, Any],
        total_transactions: int,
        initial_prompt: Optional[str] = None,
        target_precision: float = 0.98,
        target_accuracy: float = 0.98
    ) -> Dict[str, Any]:
        """
        Run adaptive tuning loop until targets are met

        Args:
            requirement: FW requirement name/description
            data: Transaction data to analyze
            ground_truth: Ground truth with correct transaction IDs
            total_transactions: Total number of transactions
            initial_prompt: Starting prompt (generated if None)
            target_precision: Minimum precision (default 98%)
            target_accuracy: Minimum accuracy (default 98%)

        Returns:
            Final tuning results with best prompt and metrics
        """
        logger.info(f"Starting adaptive tuning for: {requirement}")
        logger.info(f"Targets: Precision={target_precision:.0%}, Accuracy={target_accuracy:.0%}")
        logger.info(f"Max iterations: {self.max_iterations}")

        # Generate or use initial prompt
        if initial_prompt is None:
            current_prompt = self.generate_initial_prompt(requirement, {
                'requirement': requirement,
                'total_transactions': total_transactions,
                'ground_truth_count': len(ground_truth.get('high_value_transactions', []))
            })
            logger.info(f"Generated initial prompt ({len(current_prompt)} chars)")
        else:
            current_prompt = initial_prompt
            logger.info(f"Using provided initial prompt")

        best_metrics = None
        best_prompt = current_prompt
        best_score = 0.0

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")

            # Test current prompt
            test_result = self.test_prompt(
                current_prompt,
                data,
                ground_truth,
                total_transactions
            )

            if not test_result['success']:
                logger.error(f"Iteration {iteration} failed")
                continue

            metrics = test_result['metrics']

            # Calculate composite score
            score = (metrics['precision'] + metrics['accuracy']) / 2

            # Log results
            logger.info(f"Precision: {metrics['precision']:.2%}")
            logger.info(f"Accuracy:  {metrics['accuracy']:.2%}")
            logger.info(f"F1 Score:  {metrics['f1_score']:.2%}")
            logger.info(f"Score:     {score:.2%}")

            # Track best
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_prompt = current_prompt
                logger.info(f"✓ New best score: {best_score:.2%}")

            # Store in history
            self.tuning_history.append({
                'iteration': iteration,
                'prompt': current_prompt,
                'metrics': metrics,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })

            # Check if targets met
            if metrics['precision'] >= target_precision and metrics['accuracy'] >= target_accuracy:
                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 TARGET ACHIEVED in {iteration} iterations!")
                logger.info(f"Precision: {metrics['precision']:.2%} ≥ {target_precision:.0%}")
                logger.info(f"Accuracy: {metrics['accuracy']:.2%} ≥ {target_accuracy:.0%}")
                logger.info(f"{'='*60}\n")

                return {
                    'success': True,
                    'target_achieved': True,
                    'iterations': iteration,
                    'best_prompt': current_prompt,
                    'best_metrics': metrics,
                    'final_score': score,
                    'history': self.tuning_history
                }

            # Generate improved prompt for next iteration
            if iteration < self.max_iterations:
                logger.info(f"\nGenerating improved prompt for iteration {iteration + 1}...")
                current_prompt = self.improve_prompt_from_failures(
                    current_prompt,
                    metrics,
                    requirement,
                    iteration
                )

        # Max iterations reached without hitting target
        logger.warning(f"\n{'='*60}")
        logger.warning(f"⚠ Max iterations reached without hitting target")
        logger.warning(f"Best Precision: {best_metrics['precision']:.2%}")
        logger.warning(f"Best Accuracy: {best_metrics['accuracy']:.2%}")
        logger.warning(f"{'='*60}\n")

        return {
            'success': True,
            'target_achieved': False,
            'iterations': self.max_iterations,
            'best_prompt': best_prompt,
            'best_metrics': best_metrics,
            'final_score': best_score,
            'history': self.tuning_history,
            'message': f'Best score {best_score:.2%} after {self.max_iterations} iterations'
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'specificity': 0.0,
            'confusion_matrix': {
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0
            },
            'predicted_count': 0,
            'ground_truth_count': 0,
            'total_transactions': 0,
            'false_positives': [],
            'false_negatives': [],
            'meets_98_percent_target': False
        }

    def test_multiple_formats(
        self,
        requirement: str,
        data: str,
        ground_truth: Dict[str, Any],
        total_transactions: int,
        formats: List[str] = None
    ) -> Dict[str, Any]:
        """
        Test prompts with different output formats

        Args:
            formats: List of output formats to test (json, markdown, text, etc.)
        """
        if formats is None:
            formats = ['json', 'markdown', 'text']

        logger.info(f"Testing multiple formats: {formats}")

        format_results = []

        for fmt in formats:
            logger.info(f"\nTesting format: {fmt}")

            # Generate format-specific prompt
            format_instruction = {
                'json': "Return results as JSON with transaction_ids array",
                'markdown': "Return results as markdown list with transaction IDs",
                'text': "Return results as plain text, one transaction ID per line"
            }.get(fmt, "Return results in clear format")

            # Run adaptive tuning for this format
            result = self.adaptive_tune(
                requirement=f"{requirement} ({format_instruction})",
                data=data,
                ground_truth=ground_truth,
                total_transactions=total_transactions,
                target_precision=0.98,
                target_accuracy=0.98
            )

            result['format'] = fmt
            format_results.append(result)

            if result.get('target_achieved'):
                logger.info(f"✓ Format '{fmt}' achieved target!")
                break

        # Find best format
        best_format_result = max(format_results, key=lambda x: x.get('final_score', 0))

        return {
            'success': True,
            'format_results': format_results,
            'best_format': best_format_result['format'],
            'best_result': best_format_result
        }
