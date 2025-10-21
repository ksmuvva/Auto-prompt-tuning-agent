"""Core agent functionality."""

from typing import List, Dict, Any, Optional, Tuple
from .config import AgentConfig, LLMConfig
from .llm_interface import LLMInterface
from .metrics import MetricsCalculator
from .prompt_tuner import PromptTuner


class PromptTuningAgent:
    """Main agent for automatic prompt tuning."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent.

        Args:
            config: Optional agent configuration
        """
        self.config = config or AgentConfig()

        if not isinstance(self.config, AgentConfig):
            raise TypeError("config must be an AgentConfig instance")

        self.llm = LLMInterface(self.config.llm_config)
        self.metrics = MetricsCalculator()
        self.tuner = PromptTuner(self.llm, self.metrics, self.config)
        self._session_history: List[Dict[str, Any]] = []

    def optimize_prompt(
        self,
        prompt: str,
        expected_output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Optimize a single prompt.

        Args:
            prompt: Initial prompt to optimize
            expected_output: Optional expected output for evaluation

        Returns:
            Dictionary with optimization results

        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        if not prompt.strip():
            raise ValueError("prompt cannot be only whitespace")

        best_prompt, best_score = self.tuner.tune(prompt, expected_output)

        result = {
            "original_prompt": prompt,
            "optimized_prompt": best_prompt,
            "score": best_score,
            "improvement": best_score - (
                self.tuner.history[0]["score"] if self.tuner.history else 0.0
            ),
            "iterations": self.tuner.get_iteration_count(),
            "history": self.tuner.get_history(),
        }

        self._session_history.append(result)
        return result

    def optimize_batch(
        self,
        prompts: List[str],
        expected_outputs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Optimize multiple prompts.

        Args:
            prompts: List of prompts to optimize
            expected_outputs: Optional list of expected outputs

        Returns:
            List of optimization results

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
            result = self.optimize_prompt(prompt, expected)
            results.append(result)

        return results

    def evaluate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Evaluate a prompt without optimization.

        Args:
            prompt: Prompt to evaluate

        Returns:
            Dictionary with evaluation results

        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        metrics = self.tuner.evaluate_prompt(prompt)

        return {
            "prompt": prompt,
            "metrics": metrics.to_dict(),
        }

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get session optimization history.

        Returns:
            List of optimization results from this session
        """
        return self._session_history.copy()

    def reset_session(self) -> None:
        """Reset the session history."""
        self._session_history = []
        self.llm.reset_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        return {
            "llm_stats": self.llm.get_stats(),
            "session_optimizations": len(self._session_history),
            "config": self.config.to_dict(),
        }

    def update_config(self, new_config: AgentConfig) -> None:
        """Update agent configuration.

        Args:
            new_config: New configuration

        Raises:
            TypeError: If new_config is invalid type
        """
        if not isinstance(new_config, AgentConfig):
            raise TypeError("new_config must be an AgentConfig instance")

        self.config = new_config
        self.llm = LLMInterface(self.config.llm_config)
        self.tuner = PromptTuner(self.llm, self.metrics, self.config)
