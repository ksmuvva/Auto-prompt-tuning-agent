"""
Automated Prompt Tuning Engine
Optimizes prompts through iterative testing and refinement
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from datetime import datetime

from agent.llm_service import LLMService
from agent.metrics import PromptMetrics
from prompts.templates import PromptTemplateLibrary, PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTuner:
    """Automated prompt optimization engine"""

    def __init__(
        self,
        llm_service: LLMService,
        template_library: PromptTemplateLibrary,
        output_dir: str = "results"
    ):
        self.llm_service = llm_service
        self.template_library = template_library
        self.metrics_evaluator = PromptMetrics()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.tuning_history = []
        self.best_prompt = None
        self.best_score = 0.0

    def test_single_prompt(
        self,
        template_name: str,
        data: str,
        ground_truth: Dict[str, Any],
        threshold: float = 250.0
    ) -> Dict[str, Any]:
        """Test a single prompt template"""
        logger.info(f"Testing prompt template: {template_name}")

        # Get and format the template
        formatted_prompt = self.template_library.format_template(
            template_name,
            data=data,
            threshold=threshold
        )

        if not formatted_prompt:
            logger.error(f"Failed to format template: {template_name}")
            return {'success': False, 'error': 'Template formatting failed'}

        # Generate response from LLM
        llm_result = self.llm_service.generate(formatted_prompt)

        # Evaluate the result
        metrics = self.metrics_evaluator.evaluate_prompt(
            template_name,
            llm_result,
            ground_truth
        )

        return metrics

    def test_all_prompts(
        self,
        data: str,
        ground_truth: Dict[str, Any],
        threshold: float = 250.0
    ) -> List[Dict[str, Any]]:
        """Test all available prompt templates"""
        logger.info("Testing all prompt templates...")

        all_metrics = []
        template_names = self.template_library.list_templates()

        for i, template_name in enumerate(template_names):
            logger.info(f"Progress: {i+1}/{len(template_names)}")

            metrics = self.test_single_prompt(
                template_name,
                data,
                ground_truth,
                threshold
            )

            if metrics.get('success'):
                all_metrics.append(metrics)

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        return all_metrics

    def run_tuning_iteration(
        self,
        data: str,
        ground_truth: Dict[str, Any],
        threshold: float = 250.0,
        test_all: bool = True
    ) -> Dict[str, Any]:
        """Run a single tuning iteration"""
        logger.info("=== Starting Prompt Tuning Iteration ===")

        iteration_start = time.time()

        if test_all:
            all_metrics = self.test_all_prompts(data, ground_truth, threshold)
        else:
            # Test only a subset or specific prompts
            template_names = self.template_library.list_templates()[:3]
            all_metrics = [
                self.test_single_prompt(name, data, ground_truth, threshold)
                for name in template_names
            ]

        # Compare results
        comparison = self.metrics_evaluator.compare_prompts(all_metrics)

        # Update best prompt if improved
        if comparison.get('best_prompt'):
            best = comparison['best_prompt']
            if best['composite_score'] > self.best_score:
                self.best_score = best['composite_score']
                self.best_prompt = best['name']
                logger.info(f"NEW BEST PROMPT: {best['name']} (score: {best['composite_score']:.3f})")

        iteration_time = time.time() - iteration_start

        iteration_result = {
            'timestamp': datetime.now().isoformat(),
            'iteration_time': iteration_time,
            'prompts_tested': len(all_metrics),
            'best_prompt': self.best_prompt,
            'best_score': self.best_score,
            'comparison': comparison,
            'all_metrics': all_metrics
        }

        self.tuning_history.append(iteration_result)

        return iteration_result

    def generate_improved_prompt(
        self,
        baseline_metrics: Dict[str, Any]
    ) -> Optional[PromptTemplate]:
        """
        Use LLM to generate an improved prompt based on metrics feedback

        This implements true AI-powered prompt optimization
        """
        logger.info("Generating improved prompt using LLM...")

        # Get suggestions
        suggestions = self.metrics_evaluator.get_improvement_suggestions(baseline_metrics)

        # Create a meta-prompt to improve the prompt
        meta_prompt = f"""You are a prompt engineering expert. Analyze this prompt performance and create an improved version.

CURRENT PROMPT PERFORMANCE:
- Composite Score: {baseline_metrics.get('composite_score', 0):.2f}
- Accuracy: {baseline_metrics.get('accuracy', 0):.2f}
- F1 Score: {baseline_metrics.get('f1_score', 0):.2f}
- Completeness: {baseline_metrics.get('completeness', 0):.2f}
- Format Quality: {baseline_metrics.get('format_quality', 0):.2f}

IMPROVEMENT SUGGESTIONS:
{chr(10).join(f"- {s}" for s in suggestions)}

TASK:
Create an improved prompt for analyzing bank transactions to:
1. Find all transactions above a threshold ({{threshold}} GBP)
2. Detect anomalies in transaction patterns

The prompt should address the weaknesses identified above.

Return ONLY the improved prompt text, with placeholders {{data}} and {{threshold}}.
"""

        result = self.llm_service.generate(meta_prompt)

        if result.get('success'):
            improved_prompt_text = result['response']

            # Create new template
            new_template = self.template_library.create_custom_template(
                name=f"ai_optimized_{int(time.time())}",
                template_text=improved_prompt_text,
                description="AI-generated optimized prompt"
            )

            logger.info(f"Created new optimized prompt: {new_template.name}")
            return new_template

        return None

    def run_adaptive_tuning(
        self,
        data: str,
        ground_truth: Dict[str, Any],
        threshold: float = 250.0,
        max_iterations: int = 3,
        target_score: float = 0.85
    ) -> Dict[str, Any]:
        """
        Run adaptive tuning that generates new prompts based on performance

        This is the core AI agent capability - autonomous improvement
        """
        logger.info("=== Starting Adaptive Prompt Tuning ===")
        logger.info(f"Target: {target_score:.2f} | Max Iterations: {max_iterations}")

        for iteration in range(max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Run current iteration
            iteration_result = self.run_tuning_iteration(data, ground_truth, threshold, test_all=True)

            current_best_score = iteration_result['best_score']
            logger.info(f"Current best score: {current_best_score:.3f}")

            # Check if target achieved
            if current_best_score >= target_score:
                logger.info(f"Target score {target_score} achieved! Stopping.")
                break

            # Generate improved prompt for next iteration
            if iteration < max_iterations - 1:
                # Get metrics of current best prompt
                best_prompt_name = iteration_result['best_prompt']
                best_metrics = next(
                    (m for m in iteration_result['all_metrics'] if m['prompt_name'] == best_prompt_name),
                    None
                )

                if best_metrics and current_best_score < target_score:
                    logger.info("Generating AI-optimized prompt for next iteration...")
                    new_template = self.generate_improved_prompt(best_metrics)

                    if new_template:
                        logger.info("Testing new AI-generated prompt...")
                        new_metrics = self.test_single_prompt(
                            new_template.name,
                            data,
                            ground_truth,
                            threshold
                        )

                        if new_metrics.get('composite_score', 0) > current_best_score:
                            logger.info("AI-generated prompt improved performance!")
                        else:
                            logger.info("AI-generated prompt did not improve. Continuing...")

        # Final summary
        final_summary = {
            'total_iterations': len(self.tuning_history),
            'final_best_prompt': self.best_prompt,
            'final_best_score': self.best_score,
            'target_achieved': self.best_score >= target_score,
            'tuning_history': self.tuning_history
        }

        logger.info("\n=== Adaptive Tuning Complete ===")
        logger.info(f"Final Best Prompt: {self.best_prompt}")
        logger.info(f"Final Best Score: {self.best_score:.3f}")

        return final_summary

    def export_results(self, filename: str = None):
        """Export tuning results to JSON"""
        if filename is None:
            filename = f"tuning_results_{int(time.time())}.json"

        filepath = self.output_dir / filename

        results = {
            'best_prompt': self.best_prompt,
            'best_score': self.best_score,
            'tuning_history': self.tuning_history,
            'total_prompts_tested': sum(
                h['prompts_tested'] for h in self.tuning_history
            )
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results exported to {filepath}")
        return str(filepath)

    def get_best_prompt_template(self) -> Optional[PromptTemplate]:
        """Get the best performing prompt template"""
        if self.best_prompt:
            return self.template_library.get_template(self.best_prompt)
        return None

    def save_best_prompt(self, filename: str = None):
        """Save the best prompt template to a file"""
        if not self.best_prompt:
            logger.warning("No best prompt available to save")
            return None

        if filename is None:
            filename = f"best_prompt_{int(time.time())}.txt"

        filepath = self.output_dir / filename

        template = self.get_best_prompt_template()
        if template:
            with open(filepath, 'w') as f:
                f.write(f"# Best Prompt: {template.name}\n")
                f.write(f"# Score: {self.best_score:.3f}\n\n")
                f.write(template.template)

            logger.info(f"Best prompt saved to {filepath}")
            return str(filepath)

        return None
