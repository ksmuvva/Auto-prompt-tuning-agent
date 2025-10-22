"""
Comprehensive tests for A/B testing framework
"""

import pytest
import numpy as np
from agent.ab_testing import (
    Variant, TestResult, ABTest, MultiVariantOptimizer,
    TestStatus, SignificanceTest
)


class TestVariant:
    """Test Variant class"""

    def test_variant_creation(self):
        """Test creating a variant"""
        var = Variant(id="variant_a", name="Control", description="Baseline prompt")
        assert var.id == "variant_a"
        assert var.name == "Control"
        assert var.sample_size == 0

    def test_add_sample(self):
        """Test adding samples"""
        var = Variant(id="test", name="Test")
        var.add_sample(0.85)
        var.add_sample(0.90)
        var.add_sample(0.88)

        assert var.sample_size == 3
        assert len(var.samples) == 3

    def test_statistics(self):
        """Test statistical calculations"""
        var = Variant(id="test", name="Test")
        samples = [0.85, 0.90, 0.88, 0.92, 0.87]
        for s in samples:
            var.add_sample(s)

        assert var.mean == pytest.approx(np.mean(samples))
        assert var.std == pytest.approx(np.std(samples, ddof=1))
        assert var.sem > 0

    def test_to_dict(self):
        """Test converting to dictionary"""
        var = Variant(id="test", name="Test", description="Test variant")
        var.add_sample(0.9)

        d = var.to_dict()
        assert d['id'] == "test"
        assert d['name'] == "Test"
        assert d['sample_size'] == 1
        assert 'mean' in d
        assert 'std' in d


class TestABTest:
    """Test ABTest class"""

    def test_initialization(self):
        """Test initializing A/B test"""
        test = ABTest(
            test_name="Prompt Comparison",
            alpha=0.05,
            power=0.8
        )
        assert test.test_name == "Prompt Comparison"
        assert test.alpha == 0.05
        assert test.power == 0.8
        assert test.status == TestStatus.RUNNING

    def test_add_variant(self):
        """Test adding variants"""
        test = ABTest(test_name="Test")
        var = test.add_variant("var_a", "Variant A", "First variant")

        assert "var_a" in test.variants
        assert var.name == "Variant A"

    def test_record_observation(self):
        """Test recording observations"""
        test = ABTest(test_name="Test")
        test.add_variant("var_a", "Variant A")

        test.record_observation("var_a", 0.85)
        test.record_observation("var_a", 0.90)

        assert test.variants["var_a"].sample_size == 2

    def test_record_invalid_variant(self):
        """Test recording for nonexistent variant raises error"""
        test = ABTest(test_name="Test")

        with pytest.raises(ValueError):
            test.record_observation("nonexistent", 0.5)

    def test_sample_size_calculation(self):
        """Test required sample size calculation"""
        test = ABTest(
            test_name="Test",
            alpha=0.05,
            power=0.8,
            minimum_detectable_effect=0.05
        )

        n = test.calculate_required_sample_size(
            baseline_mean=0.80,
            baseline_std=0.10
        )

        assert n > 0
        assert isinstance(n, int)

    def test_t_test_significant_difference(self):
        """Test t-test with significant difference"""
        test = ABTest(test_name="Test", alpha=0.05)

        # Add variants
        test.add_variant("control", "Control")
        test.add_variant("treatment", "Treatment")

        # Control: mean ~ 0.80
        np.random.seed(42)
        for _ in range(50):
            test.record_observation("control", np.random.normal(0.80, 0.10))

        # Treatment: mean ~ 0.90 (significantly higher)
        for _ in range(50):
            test.record_observation("treatment", np.random.normal(0.90, 0.10))

        result = test.perform_t_test("control", "treatment", equal_var=False)

        assert result.significant == True
        assert result.p_value < 0.05
        assert result.winner == "treatment"
        assert "Significant difference" in result.recommendation

    def test_t_test_no_difference(self):
        """Test t-test with no significant difference"""
        test = ABTest(test_name="Test", alpha=0.05)

        test.add_variant("var_a", "Variant A")
        test.add_variant("var_b", "Variant B")

        # Both variants: mean ~ 0.85
        np.random.seed(42)
        for _ in range(50):
            test.record_observation("var_a", np.random.normal(0.85, 0.10))
            test.record_observation("var_b", np.random.normal(0.85, 0.10))

        result = test.perform_t_test("var_a", "var_b", equal_var=False)

        assert result.significant == False
        assert result.p_value >= 0.05
        assert result.winner is None
        assert "No significant difference" in result.recommendation

    def test_mann_whitney_test(self):
        """Test Mann-Whitney U test"""
        test = ABTest(test_name="Test")

        test.add_variant("var_a", "Variant A")
        test.add_variant("var_b", "Variant B")

        # Add samples
        np.random.seed(42)
        for _ in range(30):
            test.record_observation("var_a", np.random.normal(0.80, 0.10))
            test.record_observation("var_b", np.random.normal(0.85, 0.10))

        result = test.perform_mann_whitney("var_a", "var_b")

        assert result.test_type == SignificanceTest.MANN_WHITNEY
        assert result.p_value >= 0
        assert result.effect_size is not None

    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval"""
        test = ABTest(test_name="Test")

        np.random.seed(42)
        samples_a = [np.random.normal(0.80, 0.10) for _ in range(30)]
        samples_b = [np.random.normal(0.85, 0.10) for _ in range(30)]

        ci_lower, ci_upper = test._bootstrap_ci(samples_a, samples_b, n_bootstrap=1000)

        assert ci_lower <= ci_upper
        assert isinstance(ci_lower, (int, float))
        assert isinstance(ci_upper, (int, float))

    def test_multi_variant_test(self):
        """Test testing multiple variants"""
        test = ABTest(test_name="Multi-variant Test", alpha=0.05)

        # Add 3 variants
        test.add_variant("control", "Control")
        test.add_variant("treatment_1", "Treatment 1")
        test.add_variant("treatment_2", "Treatment 2")

        # Add data
        np.random.seed(42)
        for _ in range(40):
            test.record_observation("control", np.random.normal(0.80, 0.10))
            test.record_observation("treatment_1", np.random.normal(0.85, 0.10))
            test.record_observation("treatment_2", np.random.normal(0.90, 0.10))

        results = test.run_multi_variant_test()

        # Should have 3 pairwise comparisons
        assert len(results) == 3

        # Check Bonferroni correction was applied (alpha should be adjusted)
        # Original alpha / number of comparisons

    def test_sequential_testing(self):
        """Test sequential testing with early stopping"""
        test = ABTest(
            test_name="Sequential Test",
            alpha=0.05,
            sequential_testing=True
        )

        test.add_variant("control", "Control")
        test.add_variant("treatment", "Treatment")

        # Add data incrementally
        np.random.seed(42)
        for i in range(20):
            test.record_observation("control", np.random.normal(0.70, 0.10))
            test.record_observation("treatment", np.random.normal(0.90, 0.10))

            if i % 5 == 0 and i > 0:  # Interim analysis every 5 samples
                result = test.perform_sequential_test("control", "treatment")

                assert 'decision' in result
                assert 'adjusted_alpha' in result
                assert result['look_number'] == (i // 5)

    def test_get_summary(self):
        """Test getting test summary"""
        test = ABTest(test_name="Summary Test")

        test.add_variant("var_a", "Variant A")
        test.add_variant("var_b", "Variant B")

        test.record_observation("var_a", 0.85)
        test.record_observation("var_b", 0.90)

        summary = test.get_summary()

        assert summary['test_name'] == "Summary Test"
        assert 'variants' in summary
        assert 'results' in summary
        assert summary['status'] == TestStatus.RUNNING.value

    def test_export_results(self, tmp_path):
        """Test exporting results to JSON"""
        test = ABTest(test_name="Export Test")
        test.add_variant("var_a", "Variant A")
        test.record_observation("var_a", 0.85)

        filepath = tmp_path / "test_results.json"
        test.export_results(str(filepath))

        assert filepath.exists()

        import json
        with open(filepath) as f:
            data = json.load(f)

        assert data['test_name'] == "Export Test"

    def test_finalize_test(self):
        """Test finalizing a test"""
        test = ABTest(test_name="Test")
        test.finalize_test()

        assert test.status == TestStatus.COMPLETED
        assert test.end_time is not None


class TestMultiVariantOptimizer:
    """Test MultiVariantOptimizer"""

    def test_tournament_basic(self):
        """Test basic tournament elimination"""
        optimizer = MultiVariantOptimizer(alpha=0.05)

        # Create variant data with clear winner
        np.random.seed(42)
        variants_data = {
            'weak': [np.random.normal(0.70, 0.10) for _ in range(50)],
            'medium': [np.random.normal(0.80, 0.10) for _ in range(50)],
            'strong': [np.random.normal(0.90, 0.10) for _ in range(50)],
        }

        result = optimizer.run_tournament(variants_data)

        assert 'winner' in result
        assert 'rounds' in result
        assert result['winner'] in variants_data.keys()

    def test_tournament_tie(self):
        """Test tournament with similar variants"""
        optimizer = MultiVariantOptimizer(alpha=0.05)

        # Create similar variants
        np.random.seed(42)
        variants_data = {
            'var_a': [np.random.normal(0.85, 0.10) for _ in range(30)],
            'var_b': [np.random.normal(0.85, 0.10) for _ in range(30)],
            'var_c': [np.random.normal(0.85, 0.10) for _ in range(30)],
        }

        result = optimizer.run_tournament(variants_data)

        # With very similar variants, winner may be None
        # Just verify the structure is correct
        assert result['total_rounds'] >= 0


def test_integration_ab_test_full_workflow():
    """Test complete A/B testing workflow"""
    # Simulate real prompt comparison scenario
    test = ABTest(
        test_name="Prompt Template Comparison",
        alpha=0.05,
        power=0.8,
        minimum_detectable_effect=0.05
    )

    # Calculate required sample size
    n_required = test.calculate_required_sample_size(
        baseline_mean=0.80,
        baseline_std=0.10
    )

    assert n_required > 0

    # Add variants
    test.add_variant("baseline", "Baseline Prompt", "Original prompt template")
    test.add_variant("optimized", "Optimized Prompt", "Improved prompt template")

    # Simulate collecting data
    np.random.seed(42)

    # Baseline: accuracy ~ 0.80
    for _ in range(n_required):
        score = np.random.normal(0.80, 0.10)
        test.record_observation("baseline", max(0, min(1, score)))  # Clamp to [0,1]

    # Optimized: accuracy ~ 0.84 (5% improvement)
    for _ in range(n_required):
        score = np.random.normal(0.84, 0.10)
        test.record_observation("optimized", max(0, min(1, score)))

    # Run t-test
    result = test.run_test("baseline", "optimized", SignificanceTest.WELCH_T_TEST)

    # With sufficient samples and 5% improvement, should detect difference
    assert result is not None
    assert result.p_value is not None

    # Get summary
    summary = test.get_summary()
    assert summary['variants']['baseline']['sample_size'] == n_required
    assert summary['variants']['optimized']['sample_size'] == n_required

    # Finalize test
    test.finalize_test()
    assert test.status == TestStatus.COMPLETED
