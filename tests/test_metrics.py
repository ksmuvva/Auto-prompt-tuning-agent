"""Tests for metrics module."""

import pytest
import numpy as np
from src.agent.metrics import MetricsCalculator, PromptMetrics


class TestPromptMetrics:
    """Tests for PromptMetrics dataclass."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test PromptMetrics initialization."""
        metrics = PromptMetrics(
            score=0.8,
            length=100,
            clarity=0.9,
            specificity=0.7,
            token_efficiency=0.85,
            metadata={"test": True}
        )
        assert metrics.score == 0.8
        assert metrics.length == 100
        assert metrics.clarity == 0.9

    @pytest.mark.edge_case
    def test_score_validation(self):
        """Test score must be between 0 and 1."""
        # Valid scores
        PromptMetrics(0.0, 10, 0.5, 0.5, 0.5, {})
        PromptMetrics(1.0, 10, 0.5, 0.5, 0.5, {})

        # Invalid scores
        with pytest.raises(ValueError, match="score must be between"):
            PromptMetrics(-0.1, 10, 0.5, 0.5, 0.5, {})

        with pytest.raises(ValueError, match="score must be between"):
            PromptMetrics(1.1, 10, 0.5, 0.5, 0.5, {})

    @pytest.mark.edge_case
    def test_length_validation(self):
        """Test length must be non-negative."""
        PromptMetrics(0.5, 0, 0.5, 0.5, 0.5, {})  # Zero is valid

        with pytest.raises(ValueError, match="length must be non-negative"):
            PromptMetrics(0.5, -1, 0.5, 0.5, 0.5, {})

    @pytest.mark.edge_case
    def test_clarity_validation(self):
        """Test clarity must be between 0 and 1."""
        with pytest.raises(ValueError, match="clarity must be between"):
            PromptMetrics(0.5, 10, -0.1, 0.5, 0.5, {})

        with pytest.raises(ValueError, match="clarity must be between"):
            PromptMetrics(0.5, 10, 1.1, 0.5, 0.5, {})

    @pytest.mark.edge_case
    def test_specificity_validation(self):
        """Test specificity must be between 0 and 1."""
        with pytest.raises(ValueError, match="specificity must be between"):
            PromptMetrics(0.5, 10, 0.5, -0.1, 0.5, {})

        with pytest.raises(ValueError, match="specificity must be between"):
            PromptMetrics(0.5, 10, 0.5, 1.1, 0.5, {})

    @pytest.mark.edge_case
    def test_token_efficiency_validation(self):
        """Test token_efficiency must be between 0 and 1."""
        with pytest.raises(ValueError, match="token_efficiency must be between"):
            PromptMetrics(0.5, 10, 0.5, 0.5, -0.1, {})

        with pytest.raises(ValueError, match="token_efficiency must be between"):
            PromptMetrics(0.5, 10, 0.5, 0.5, 1.1, {})

    @pytest.mark.micro
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PromptMetrics(0.8, 100, 0.9, 0.7, 0.85, {"key": "value"})
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result["score"] == 0.8
        assert result["length"] == 100
        assert result["metadata"]["key"] == "value"


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    @pytest.mark.unit
    def test_initialization_default(self):
        """Test default initialization."""
        calc = MetricsCalculator()
        assert "clarity" in calc.weights
        assert "specificity" in calc.weights
        assert sum(calc.weights.values()) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_initialization_custom_weights(self):
        """Test initialization with custom weights."""
        weights = {
            "clarity": 0.4,
            "specificity": 0.3,
            "token_efficiency": 0.2,
            "response_quality": 0.1,
        }
        calc = MetricsCalculator(weights)
        assert calc.weights == weights

    @pytest.mark.edge_case
    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        invalid_weights = {
            "clarity": 0.4,
            "specificity": 0.3,
            "token_efficiency": 0.2,
            "response_quality": 0.2,  # Sum = 1.1
        }
        with pytest.raises(ValueError, match="must sum to 1.0"):
            MetricsCalculator(invalid_weights)

    @pytest.mark.edge_case
    def test_weights_in_valid_range(self):
        """Test that individual weights must be between 0 and 1."""
        invalid_weights = {
            "clarity": 1.5,  # Invalid
            "specificity": -0.5,  # Invalid
            "token_efficiency": 0.0,
            "response_quality": 0.0,
        }
        with pytest.raises(ValueError):
            MetricsCalculator(invalid_weights)

    @pytest.mark.unit
    def test_calculate_metrics_basic(self, metrics_calculator):
        """Test basic metrics calculation."""
        metrics = metrics_calculator.calculate_metrics("Write a hello world program")
        assert isinstance(metrics, PromptMetrics)
        assert 0.0 <= metrics.score <= 1.0
        assert metrics.length > 0

    @pytest.mark.edge_case
    def test_calculate_metrics_empty_prompt(self, metrics_calculator):
        """Test metrics calculation with empty prompt."""
        with pytest.raises(ValueError, match="non-empty string"):
            metrics_calculator.calculate_metrics("")

    @pytest.mark.edge_case
    def test_calculate_metrics_whitespace_prompt(self, metrics_calculator):
        """Test metrics calculation with whitespace-only prompt."""
        with pytest.raises(ValueError, match="only whitespace"):
            metrics_calculator.calculate_metrics("   ")

    @pytest.mark.edge_case
    def test_calculate_metrics_none_prompt(self, metrics_calculator):
        """Test metrics calculation with None prompt."""
        with pytest.raises(ValueError):
            metrics_calculator.calculate_metrics(None)

    @pytest.mark.micro
    def test_calculate_metrics_with_response(self, metrics_calculator):
        """Test metrics calculation with response."""
        metrics = metrics_calculator.calculate_metrics(
            "Write hello world",
            response="print('Hello, World!')",
        )
        assert isinstance(metrics, PromptMetrics)

    @pytest.mark.micro
    def test_calculate_metrics_with_expected_output(self, metrics_calculator):
        """Test metrics calculation with expected output."""
        metrics = metrics_calculator.calculate_metrics(
            "Write hello world",
            response="print('Hello, World!')",
            expected_output="print('Hello, World!')",
        )
        assert metrics.metadata["has_expected"] is True

    @pytest.mark.edge_case
    def test_calculate_clarity_empty(self, metrics_calculator):
        """Test clarity calculation with edge cases."""
        # Empty should return 0
        clarity = metrics_calculator._calculate_clarity("")
        assert clarity == 0.0

    @pytest.mark.micro
    def test_calculate_clarity_with_question(self, metrics_calculator):
        """Test clarity improves with question mark."""
        clarity_without = metrics_calculator._calculate_clarity("Write code")
        clarity_with = metrics_calculator._calculate_clarity("Write code?")
        assert clarity_with >= clarity_without

    @pytest.mark.micro
    def test_calculate_clarity_with_instructions(self, metrics_calculator):
        """Test clarity improves with instruction words."""
        clarity = metrics_calculator._calculate_clarity("Please explain the concept")
        assert clarity > 0.3

    @pytest.mark.edge_case
    def test_calculate_clarity_very_short(self, metrics_calculator):
        """Test clarity penalty for very short prompts."""
        clarity_short = metrics_calculator._calculate_clarity("Hi")
        clarity_normal = metrics_calculator._calculate_clarity("Please write a function")
        assert clarity_normal > clarity_short

    @pytest.mark.edge_case
    def test_calculate_clarity_very_long(self, metrics_calculator):
        """Test clarity penalty for very long prompts."""
        long_prompt = "word " * 150
        clarity = metrics_calculator._calculate_clarity(long_prompt)
        assert clarity < 1.0

    @pytest.mark.micro
    def test_calculate_specificity_with_numbers(self, metrics_calculator):
        """Test specificity increases with numbers."""
        spec_without = metrics_calculator._calculate_specificity("Write functions")
        spec_with = metrics_calculator._calculate_specificity("Write 5 functions")
        assert spec_with > spec_without

    @pytest.mark.micro
    def test_calculate_specificity_with_examples(self, metrics_calculator):
        """Test specificity increases with examples."""
        spec = metrics_calculator._calculate_specificity("Write code, e.g., hello world")
        assert spec >= 0.6

    @pytest.mark.micro
    def test_calculate_specificity_with_constraints(self, metrics_calculator):
        """Test specificity increases with constraints."""
        spec = metrics_calculator._calculate_specificity("You must use only Python")
        assert spec >= 0.5

    @pytest.mark.edge_case
    def test_calculate_specificity_empty(self, metrics_calculator):
        """Test specificity with empty prompt."""
        spec = metrics_calculator._calculate_specificity("")
        assert spec == 0.0

    @pytest.mark.micro
    def test_calculate_token_efficiency_optimal(self, metrics_calculator):
        """Test token efficiency at optimal length."""
        prompt = " ".join(["word"] * 30)  # Optimal range
        efficiency = metrics_calculator._calculate_token_efficiency(prompt)
        assert efficiency == 1.0

    @pytest.mark.edge_case
    def test_calculate_token_efficiency_too_short(self, metrics_calculator):
        """Test token efficiency penalty for very short prompts."""
        efficiency = metrics_calculator._calculate_token_efficiency("Hi")
        assert efficiency < 1.0

    @pytest.mark.edge_case
    def test_calculate_token_efficiency_too_long(self, metrics_calculator):
        """Test token efficiency penalty for very long prompts."""
        long_prompt = " ".join(["word"] * 200)
        efficiency = metrics_calculator._calculate_token_efficiency(long_prompt)
        assert efficiency < 1.0

    @pytest.mark.edge_case
    def test_calculate_token_efficiency_empty(self, metrics_calculator):
        """Test token efficiency with empty prompt."""
        efficiency = metrics_calculator._calculate_token_efficiency("")
        assert efficiency == 0.0

    @pytest.mark.micro
    def test_calculate_response_quality(self, metrics_calculator):
        """Test response quality calculation."""
        quality = metrics_calculator._calculate_response_quality(
            "hello world program",
            "hello world program"
        )
        assert quality > 0.5

    @pytest.mark.edge_case
    def test_calculate_response_quality_no_match(self, metrics_calculator):
        """Test response quality with no overlap."""
        quality = metrics_calculator._calculate_response_quality(
            "completely different",
            "nothing matches here"
        )
        assert quality >= 0.0

    @pytest.mark.edge_case
    def test_calculate_response_quality_empty(self, metrics_calculator):
        """Test response quality with empty strings."""
        assert metrics_calculator._calculate_response_quality("", "test") == 0.0
        assert metrics_calculator._calculate_response_quality("test", "") == 0.0
        assert metrics_calculator._calculate_response_quality("", "") == 0.0

    @pytest.mark.unit
    def test_compare_prompts(self, metrics_calculator):
        """Test comparing two prompts."""
        metrics1 = PromptMetrics(0.8, 10, 0.8, 0.8, 0.8, {})
        metrics2 = PromptMetrics(0.6, 10, 0.6, 0.6, 0.6, {})

        result = metrics_calculator.compare_prompts(metrics1, metrics2)
        assert result == 1

    @pytest.mark.edge_case
    def test_compare_prompts_equal(self, metrics_calculator):
        """Test comparing equal prompts."""
        metrics1 = PromptMetrics(0.7, 10, 0.7, 0.7, 0.7, {})
        metrics2 = PromptMetrics(0.7, 10, 0.8, 0.6, 0.7, {})

        result = metrics_calculator.compare_prompts(metrics1, metrics2)
        assert result == 0

    @pytest.mark.edge_case
    def test_compare_prompts_invalid_type(self, metrics_calculator):
        """Test comparing with invalid types."""
        metrics1 = PromptMetrics(0.7, 10, 0.7, 0.7, 0.7, {})

        with pytest.raises(TypeError):
            metrics_calculator.compare_prompts(metrics1, "not metrics")

    @pytest.mark.unit
    def test_aggregate_metrics(self, metrics_calculator):
        """Test aggregating multiple metrics."""
        metrics_list = [
            PromptMetrics(0.8, 10, 0.8, 0.8, 0.8, {}),
            PromptMetrics(0.6, 20, 0.6, 0.6, 0.6, {}),
            PromptMetrics(0.7, 15, 0.7, 0.7, 0.7, {}),
        ]

        agg = metrics_calculator.aggregate_metrics(metrics_list)
        assert "mean_score" in agg
        assert "std_score" in agg
        assert "min_score" in agg
        assert "max_score" in agg
        assert agg["mean_score"] == pytest.approx(0.7, abs=0.01)

    @pytest.mark.edge_case
    def test_aggregate_metrics_empty_list(self, metrics_calculator):
        """Test aggregating empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            metrics_calculator.aggregate_metrics([])

    @pytest.mark.edge_case
    def test_aggregate_metrics_invalid_types(self, metrics_calculator):
        """Test aggregating with invalid types."""
        with pytest.raises(TypeError):
            metrics_calculator.aggregate_metrics([0.8, 0.6, 0.7])

    @pytest.mark.edge_case
    def test_aggregate_metrics_single_item(self, metrics_calculator):
        """Test aggregating single metric."""
        metrics_list = [PromptMetrics(0.8, 10, 0.8, 0.8, 0.8, {})]
        agg = metrics_calculator.aggregate_metrics(metrics_list)
        assert agg["mean_score"] == 0.8
        assert agg["std_score"] == 0.0
        assert agg["min_score"] == 0.8
        assert agg["max_score"] == 0.8
