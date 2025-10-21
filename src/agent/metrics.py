"""Metrics calculation for prompt evaluation."""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PromptMetrics:
    """Container for prompt evaluation metrics."""
    score: float
    length: int
    clarity: float
    specificity: float
    token_efficiency: float
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate metrics after initialization."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        if self.length < 0:
            raise ValueError("length must be non-negative")
        if not 0.0 <= self.clarity <= 1.0:
            raise ValueError("clarity must be between 0.0 and 1.0")
        if not 0.0 <= self.specificity <= 1.0:
            raise ValueError("specificity must be between 0.0 and 1.0")
        if not 0.0 <= self.token_efficiency <= 1.0:
            raise ValueError("token_efficiency must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "score": self.score,
            "length": self.length,
            "clarity": self.clarity,
            "specificity": self.specificity,
            "token_efficiency": self.token_efficiency,
            "metadata": self.metadata,
        }


class MetricsCalculator:
    """Calculator for prompt evaluation metrics."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize metrics calculator.

        Args:
            weights: Optional weights for different metrics
        """
        self.weights = weights or {
            "clarity": 0.3,
            "specificity": 0.3,
            "token_efficiency": 0.2,
            "response_quality": 0.2,
        }
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0."""
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        for key, value in self.weights.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Weight for {key} must be between 0.0 and 1.0")

    def calculate_metrics(
        self,
        prompt: str,
        response: Optional[str] = None,
        expected_output: Optional[str] = None,
    ) -> PromptMetrics:
        """Calculate metrics for a prompt.

        Args:
            prompt: The prompt to evaluate
            response: Optional LLM response
            expected_output: Optional expected output for comparison

        Returns:
            PromptMetrics object

        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        if not prompt.strip():
            raise ValueError("prompt cannot be only whitespace")

        clarity = self._calculate_clarity(prompt)
        specificity = self._calculate_specificity(prompt)
        token_efficiency = self._calculate_token_efficiency(prompt)

        response_quality = 0.5
        if response and expected_output:
            response_quality = self._calculate_response_quality(response, expected_output)

        score = (
            self.weights["clarity"] * clarity +
            self.weights["specificity"] * specificity +
            self.weights["token_efficiency"] * token_efficiency +
            self.weights["response_quality"] * response_quality
        )

        return PromptMetrics(
            score=score,
            length=len(prompt),
            clarity=clarity,
            specificity=specificity,
            token_efficiency=token_efficiency,
            metadata={
                "has_response": response is not None,
                "has_expected": expected_output is not None,
            }
        )

    def _calculate_clarity(self, prompt: str) -> float:
        """Calculate clarity score of prompt.

        Args:
            prompt: Prompt text

        Returns:
            Clarity score between 0.0 and 1.0
        """
        if not prompt:
            return 0.0

        # Simple heuristics for clarity
        words = prompt.split()
        if not words:
            return 0.0

        # Check for question marks or clear instructions
        has_question = "?" in prompt
        has_instruction_words = any(
            word.lower() in ["please", "describe", "explain", "write", "create"]
            for word in words
        )

        # Penalize very short or very long prompts
        length_score = 1.0
        if len(words) < 3:
            length_score = 0.5
        elif len(words) > 100:
            length_score = 0.7

        clarity = 0.3
        if has_question:
            clarity += 0.3
        if has_instruction_words:
            clarity += 0.4

        return min(1.0, clarity * length_score)

    def _calculate_specificity(self, prompt: str) -> float:
        """Calculate specificity score of prompt.

        Args:
            prompt: Prompt text

        Returns:
            Specificity score between 0.0 and 1.0
        """
        if not prompt:
            return 0.0

        words = prompt.split()
        if not words:
            return 0.0

        # Check for specific details
        has_numbers = any(char.isdigit() for char in prompt)
        has_examples = "example" in prompt.lower() or "e.g." in prompt.lower()
        has_constraints = any(
            word in prompt.lower()
            for word in ["must", "should", "exactly", "only", "specific"]
        )

        specificity = 0.3
        if has_numbers:
            specificity += 0.2
        if has_examples:
            specificity += 0.3
        if has_constraints:
            specificity += 0.2

        return min(1.0, specificity)

    def _calculate_token_efficiency(self, prompt: str) -> float:
        """Calculate token efficiency score.

        Args:
            prompt: Prompt text

        Returns:
            Efficiency score between 0.0 and 1.0
        """
        if not prompt:
            return 0.0

        words = prompt.split()
        if not words:
            return 0.0

        # Optimal range is 10-50 words
        word_count = len(words)

        if word_count < 5:
            return 0.3
        elif word_count <= 50:
            return 1.0
        elif word_count <= 100:
            return 0.7
        else:
            return 0.4

    def _calculate_response_quality(self, response: str, expected: str) -> float:
        """Calculate response quality score.

        Args:
            response: Actual LLM response
            expected: Expected output

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not response or not expected:
            return 0.0

        # Simple similarity measure
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 0.0

        intersection = response_words & expected_words
        union = response_words | expected_words

        # Jaccard similarity
        if not union:
            return 0.0

        return len(intersection) / len(union)

    def compare_prompts(self, metrics1: PromptMetrics, metrics2: PromptMetrics) -> int:
        """Compare two prompt metrics.

        Args:
            metrics1: First prompt metrics
            metrics2: Second prompt metrics

        Returns:
            1 if metrics1 is better, -1 if metrics2 is better, 0 if equal
        """
        if not isinstance(metrics1, PromptMetrics) or not isinstance(metrics2, PromptMetrics):
            raise TypeError("Both arguments must be PromptMetrics instances")

        if metrics1.score > metrics2.score:
            return 1
        elif metrics1.score < metrics2.score:
            return -1
        else:
            return 0

    def aggregate_metrics(self, metrics_list: List[PromptMetrics]) -> Dict[str, float]:
        """Aggregate multiple prompt metrics.

        Args:
            metrics_list: List of PromptMetrics

        Returns:
            Dictionary with aggregated statistics

        Raises:
            ValueError: If metrics_list is empty
        """
        if not metrics_list:
            raise ValueError("metrics_list cannot be empty")

        if not all(isinstance(m, PromptMetrics) for m in metrics_list):
            raise TypeError("All items must be PromptMetrics instances")

        scores = [m.score for m in metrics_list]
        clarities = [m.clarity for m in metrics_list]
        specificities = [m.specificity for m in metrics_list]

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "mean_clarity": float(np.mean(clarities)),
            "mean_specificity": float(np.mean(specificities)),
        }
