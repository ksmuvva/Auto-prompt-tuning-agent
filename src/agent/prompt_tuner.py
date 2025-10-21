"""Prompt tuning logic for optimizing prompts."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .llm_interface import LLMInterface
from .metrics import MetricsCalculator, PromptMetrics
from .config import AgentConfig


class PromptTuner:
    """Tunes prompts to optimize performance."""

    def __init__(
        self,
        llm_interface: LLMInterface,
        metrics_calculator: MetricsCalculator,
        config: AgentConfig,
    ):
        """Initialize prompt tuner.

        Args:
            llm_interface: LLM interface for generation
            metrics_calculator: Metrics calculator
            config: Agent configuration

        Raises:
            TypeError: If arguments are invalid types
        """
        if not isinstance(llm_interface, LLMInterface):
            raise TypeError("llm_interface must be an LLMInterface instance")
        if not isinstance(metrics_calculator, MetricsCalculator):
            raise TypeError("metrics_calculator must be a MetricsCalculator instance")
        if not isinstance(config, AgentConfig):
            raise TypeError("config must be an AgentConfig instance")

        self.llm = llm_interface
        self.metrics = metrics_calculator
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.best_prompt: Optional[str] = None
        self.best_score: float = 0.0
        self._iteration_count = 0

    def tune(
        self,
        initial_prompt: str,
        expected_output: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Tune a prompt to optimize its performance.

        Args:
            initial_prompt: Starting prompt
            expected_output: Optional expected output for evaluation
            max_iterations: Override max iterations from config

        Returns:
            Tuple of (best_prompt, best_score)

        Raises:
            ValueError: If initial_prompt is invalid
        """
        if not initial_prompt or not isinstance(initial_prompt, str):
            raise ValueError("initial_prompt must be a non-empty string")

        if not initial_prompt.strip():
            raise ValueError("initial_prompt cannot be only whitespace")

        iterations = max_iterations if max_iterations is not None else self.config.max_iterations
        if iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self._reset_state()
        current_prompt = initial_prompt

        for i in range(iterations):
            self._iteration_count += 1

            # Generate response
            try:
                response = self.llm.generate(current_prompt)
            except Exception as e:
                # Log error and continue with variations
                response = None

            # Calculate metrics
            prompt_metrics = self.metrics.calculate_metrics(
                current_prompt,
                response,
                expected_output
            )

            # Update best if better
            if prompt_metrics.score > self.best_score:
                self.best_prompt = current_prompt
                self.best_score = prompt_metrics.score

            # Store history
            self.history.append({
                "iteration": i,
                "prompt": current_prompt,
                "score": prompt_metrics.score,
                "metrics": prompt_metrics.to_dict(),
            })

            # Check convergence
            if self._check_convergence(i):
                break

            # Generate variations for next iteration
            current_prompt = self._generate_variation(current_prompt, prompt_metrics)

        return self.best_prompt or initial_prompt, self.best_score

    def tune_batch(
        self,
        prompts: List[str],
        expected_outputs: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Tune multiple prompts.

        Args:
            prompts: List of initial prompts
            expected_outputs: Optional list of expected outputs

        Returns:
            List of tuples (best_prompt, best_score) for each input

        Raises:
            ValueError: If prompts is empty or invalid
        """
        if not prompts:
            raise ValueError("prompts list cannot be empty")

        if not isinstance(prompts, list):
            raise TypeError("prompts must be a list")

        if not all(isinstance(p, str) for p in prompts):
            raise TypeError("all prompts must be strings")

        if expected_outputs is not None:
            if len(prompts) != len(expected_outputs):
                raise ValueError("prompts and expected_outputs must have same length")

        results = []
        for i, prompt in enumerate(prompts):
            expected = expected_outputs[i] if expected_outputs else None
            result = self.tune(prompt, expected)
            results.append(result)

        return results

    def _reset_state(self) -> None:
        """Reset tuner state for new tuning session."""
        self.history = []
        self.best_prompt = None
        self.best_score = 0.0
        self._iteration_count = 0

    def _check_convergence(self, iteration: int) -> bool:
        """Check if tuning has converged.

        Args:
            iteration: Current iteration number

        Returns:
            True if converged, False otherwise
        """
        if len(self.history) < 3:
            return False

        # Get last few scores
        recent_scores = [h["score"] for h in self.history[-3:]]

        if len(recent_scores) < 3:
            return False

        # Check if improvement is below threshold
        score_range = max(recent_scores) - min(recent_scores)
        return score_range < self.config.convergence_threshold

    def _generate_variation(
        self,
        prompt: str,
        metrics: PromptMetrics
    ) -> str:
        """Generate a variation of the prompt.

        Args:
            prompt: Current prompt
            metrics: Current prompt metrics

        Returns:
            Variation of the prompt
        """
        if not prompt:
            return prompt

        variations = []

        # Make more specific if specificity is low
        if metrics.specificity < 0.5:
            variations.append(f"{prompt} Please provide specific details and examples.")

        # Make clearer if clarity is low
        if metrics.clarity < 0.5:
            variations.append(f"Please {prompt.lower()}")

        # Shorten if too long
        if metrics.token_efficiency < 0.5:
            words = prompt.split()
            if len(words) > 50:
                variations.append(" ".join(words[:50]))

        # If no variations needed, make small modification
        if not variations:
            variations.append(f"{prompt} Be concise and accurate.")

        # Return best variation (for now, just first one)
        return variations[0]

    def get_history(self) -> List[Dict[str, Any]]:
        """Get tuning history.

        Returns:
            List of history entries
        """
        return self.history.copy()

    def get_best_prompt(self) -> Optional[str]:
        """Get the best prompt found.

        Returns:
            Best prompt or None if no tuning has been done
        """
        return self.best_prompt

    def get_iteration_count(self) -> int:
        """Get the number of iterations run.

        Returns:
            Iteration count
        """
        return self._iteration_count

    def evaluate_prompt(
        self,
        prompt: str,
        expected_output: Optional[str] = None
    ) -> PromptMetrics:
        """Evaluate a single prompt without tuning.

        Args:
            prompt: Prompt to evaluate
            expected_output: Optional expected output

        Returns:
            PromptMetrics for the prompt

        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        try:
            response = self.llm.generate(prompt)
        except Exception:
            response = None

        return self.metrics.calculate_metrics(prompt, response, expected_output)
