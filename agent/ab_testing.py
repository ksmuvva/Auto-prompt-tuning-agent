"""
A/B Testing Framework for Prompt Optimization

This module implements rigorous A/B testing with:
1. Statistical significance testing (t-tests, chi-square)
2. Multi-variant testing (A/B/C/D...)
3. Sequential testing with early stopping
4. Confidence intervals and p-values
5. Sample size calculation
6. Test result reporting

Program of Thoughts:
1. Define test variants (different prompts)
2. Collect performance data for each variant
3. Apply statistical tests to determine significance
4. Calculate confidence intervals
5. Determine winner or declare no significant difference
6. Provide actionable recommendations
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum


class TestStatus(Enum):
    """Status of an A/B test"""
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"
    INCONCLUSIVE = "inconclusive"


class SignificanceTest(Enum):
    """Available statistical tests"""
    T_TEST = "t_test"  # For continuous metrics (accuracy, latency)
    WELCH_T_TEST = "welch_t_test"  # T-test without equal variance assumption
    MANN_WHITNEY = "mann_whitney"  # Non-parametric alternative
    CHI_SQUARE = "chi_square"  # For categorical outcomes
    BOOTSTRAP = "bootstrap"  # Bootstrap confidence intervals


@dataclass
class Variant:
    """Represents a test variant (e.g., different prompt template)"""
    id: str
    name: str
    description: str = ""
    samples: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_sample(self, value: float):
        """Add a measurement sample"""
        self.samples.append(value)

    @property
    def sample_size(self) -> int:
        """Number of samples collected"""
        return len(self.samples)

    @property
    def mean(self) -> float:
        """Sample mean"""
        return np.mean(self.samples) if self.samples else 0.0

    @property
    def std(self) -> float:
        """Sample standard deviation"""
        return np.std(self.samples, ddof=1) if len(self.samples) > 1 else 0.0

    @property
    def sem(self) -> float:
        """Standard error of the mean"""
        return self.std / np.sqrt(self.sample_size) if self.sample_size > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'sample_size': self.sample_size,
            'mean': self.mean,
            'std': self.std,
            'sem': self.sem,
            'samples': self.samples,
            'metadata': self.metadata
        }


@dataclass
class TestResult:
    """Result of a statistical comparison"""
    variant_a: str
    variant_b: str
    test_type: SignificanceTest
    p_value: float
    statistic: float
    significant: bool
    confidence_level: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    winner: Optional[str] = None
    recommendation: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'variant_a': self.variant_a,
            'variant_b': self.variant_b,
            'test_type': self.test_type.value,
            'p_value': self.p_value,
            'statistic': self.statistic,
            'significant': self.significant,
            'confidence_level': self.confidence_level,
            'effect_size': self.effect_size,
            'confidence_interval': list(self.confidence_interval),
            'winner': self.winner,
            'recommendation': self.recommendation
        }


class ABTest:
    """
    A/B Testing framework with statistical rigor

    Implements proper statistical testing for comparing prompt variants.
    """

    def __init__(self,
                 test_name: str,
                 alpha: float = 0.05,
                 power: float = 0.8,
                 minimum_detectable_effect: float = 0.05,
                 sequential_testing: bool = False):
        """
        Initialize A/B test

        Args:
            test_name: Name of the test
            alpha: Significance level (default 0.05 for 95% confidence)
            power: Statistical power (probability of detecting true effect)
            minimum_detectable_effect: Smallest effect size worth detecting
            sequential_testing: Enable early stopping based on sequential analysis
        """
        self.test_name = test_name
        self.alpha = alpha
        self.power = power
        self.minimum_detectable_effect = minimum_detectable_effect
        self.sequential_testing = sequential_testing

        self.variants: Dict[str, Variant] = {}
        self.status = TestStatus.RUNNING
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.results: List[TestResult] = []

        # Sequential testing parameters (O'Brien-Fleming boundaries)
        self.looks = 0  # Number of interim analyses
        self.max_looks = 5  # Maximum interim analyses

    def add_variant(self, variant_id: str, name: str, description: str = "") -> Variant:
        """Add a test variant"""
        variant = Variant(id=variant_id, name=name, description=description)
        self.variants[variant_id] = variant
        return variant

    def record_observation(self, variant_id: str, value: float):
        """Record an observation for a variant"""
        if variant_id not in self.variants:
            raise ValueError(f"Variant '{variant_id}' not found")

        self.variants[variant_id].add_sample(value)

    def calculate_required_sample_size(self,
                                       baseline_mean: float,
                                       baseline_std: float) -> int:
        """
        Calculate required sample size per variant using power analysis

        Program of Thoughts:
        1. Effect size = minimum_detectable_effect
        2. Use standard formula: n = 2 * ((Z_α/2 + Z_β) * σ / δ)^2
        3. Z_α/2: critical value for two-tailed test
        4. Z_β: critical value for power
        5. σ: standard deviation
        6. δ: minimum detectable effect

        Args:
            baseline_mean: Expected mean of baseline variant
            baseline_std: Expected standard deviation

        Returns:
            Required sample size per variant
        """
        # Critical values
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)

        # Effect size in absolute units
        delta = baseline_mean * self.minimum_detectable_effect

        # Sample size formula
        n = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2

        return int(np.ceil(n))

    def perform_t_test(self, variant_a_id: str, variant_b_id: str,
                      equal_var: bool = False) -> TestResult:
        """
        Perform independent samples t-test

        Args:
            variant_a_id: First variant ID
            variant_b_id: Second variant ID
            equal_var: Assume equal variances (False uses Welch's t-test)

        Returns:
            TestResult object
        """
        var_a = self.variants[variant_a_id]
        var_b = self.variants[variant_b_id]

        if var_a.sample_size < 2 or var_b.sample_size < 2:
            raise ValueError("Need at least 2 samples per variant for t-test")

        # Perform t-test
        statistic, p_value = stats.ttest_ind(
            var_a.samples,
            var_b.samples,
            equal_var=equal_var
        )

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((var_a.sample_size - 1) * var_a.std**2 +
             (var_b.sample_size - 1) * var_b.std**2) /
            (var_a.sample_size + var_b.sample_size - 2)
        )
        effect_size = (var_a.mean - var_b.mean) / pooled_std if pooled_std > 0 else 0.0

        # Confidence interval for difference in means
        se_diff = np.sqrt(var_a.sem**2 + var_b.sem**2)
        df = var_a.sample_size + var_b.sample_size - 2
        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        mean_diff = var_a.mean - var_b.mean
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        # Determine significance and winner
        significant = p_value < self.alpha
        winner = None
        if significant:
            winner = variant_a_id if var_a.mean > var_b.mean else variant_b_id

        # Generate recommendation
        if significant:
            improvement = abs((var_a.mean - var_b.mean) / var_b.mean * 100)
            recommendation = (
                f"Significant difference detected (p={p_value:.4f}). "
                f"{winner} performs {improvement:.1f}% better. "
                f"Recommend using {winner}."
            )
        else:
            recommendation = (
                f"No significant difference (p={p_value:.4f}). "
                f"Variants perform similarly. Consider other factors."
            )

        test_type = SignificanceTest.WELCH_T_TEST if not equal_var else SignificanceTest.T_TEST

        return TestResult(
            variant_a=variant_a_id,
            variant_b=variant_b_id,
            test_type=test_type,
            p_value=p_value,
            statistic=statistic,
            significant=significant,
            confidence_level=1 - self.alpha,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            winner=winner,
            recommendation=recommendation
        )

    def perform_mann_whitney(self, variant_a_id: str, variant_b_id: str) -> TestResult:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test)

        Useful when data is not normally distributed.
        """
        var_a = self.variants[variant_a_id]
        var_b = self.variants[variant_b_id]

        if var_a.sample_size < 2 or var_b.sample_size < 2:
            raise ValueError("Need at least 2 samples per variant")

        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            var_a.samples,
            var_b.samples,
            alternative='two-sided'
        )

        # Effect size (rank-biserial correlation)
        n1, n2 = var_a.sample_size, var_b.sample_size
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        # Bootstrap confidence interval for median difference
        ci_lower, ci_upper = self._bootstrap_ci(var_a.samples, var_b.samples)

        significant = p_value < self.alpha
        winner = None
        if significant:
            winner = variant_a_id if var_a.mean > var_b.mean else variant_b_id

        recommendation = (
            f"Mann-Whitney U test: {'Significant' if significant else 'No significant'} "
            f"difference (p={p_value:.4f}). "
            f"{'Recommend ' + winner if winner else 'No clear winner'}."
        )

        return TestResult(
            variant_a=variant_a_id,
            variant_b=variant_b_id,
            test_type=SignificanceTest.MANN_WHITNEY,
            p_value=p_value,
            statistic=statistic,
            significant=significant,
            confidence_level=1 - self.alpha,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            winner=winner,
            recommendation=recommendation
        )

    def _bootstrap_ci(self, samples_a: List[float], samples_b: List[float],
                     n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for difference in means

        Args:
            samples_a: Samples from variant A
            samples_b: Samples from variant B
            n_bootstrap: Number of bootstrap iterations

        Returns:
            (lower_bound, upper_bound) confidence interval
        """
        differences = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            resample_a = np.random.choice(samples_a, size=len(samples_a), replace=True)
            resample_b = np.random.choice(samples_b, size=len(samples_b), replace=True)

            # Calculate difference in means
            diff = np.mean(resample_a) - np.mean(resample_b)
            differences.append(diff)

        # Calculate percentile-based confidence interval
        alpha_half = self.alpha / 2
        ci_lower = np.percentile(differences, alpha_half * 100)
        ci_upper = np.percentile(differences, (1 - alpha_half) * 100)

        return ci_lower, ci_upper

    def perform_sequential_test(self, variant_a_id: str, variant_b_id: str) -> Dict:
        """
        Perform sequential analysis for early stopping

        Uses O'Brien-Fleming spending function to control Type I error.

        Returns:
            Dictionary with decision and adjusted alpha
        """
        self.looks += 1

        # O'Brien-Fleming alpha spending
        # More conservative early, less conservative late
        t = self.looks / self.max_looks
        if t >= 1.0:
            adjusted_alpha = self.alpha
        else:
            # Simplified O'Brien-Fleming boundary
            adjusted_alpha = self.alpha * (1 - np.exp(-t * 3))

        # Perform test with adjusted alpha
        original_alpha = self.alpha
        self.alpha = adjusted_alpha

        try:
            result = self.perform_t_test(variant_a_id, variant_b_id, equal_var=False)

            decision = "continue"
            if result.significant:
                decision = "stop_winner_found"
                self.status = TestStatus.STOPPED_EARLY
            elif self.looks >= self.max_looks:
                decision = "stop_max_looks"
                self.status = TestStatus.COMPLETED

            return {
                'look_number': self.looks,
                'adjusted_alpha': adjusted_alpha,
                'decision': decision,
                'result': result.to_dict()
            }
        finally:
            self.alpha = original_alpha

    def run_test(self, variant_a_id: str, variant_b_id: str,
                test_type: SignificanceTest = SignificanceTest.WELCH_T_TEST) -> TestResult:
        """
        Run appropriate statistical test

        Args:
            variant_a_id: First variant ID
            variant_b_id: Second variant ID
            test_type: Type of statistical test to perform

        Returns:
            TestResult object
        """
        if test_type == SignificanceTest.T_TEST:
            result = self.perform_t_test(variant_a_id, variant_b_id, equal_var=True)
        elif test_type == SignificanceTest.WELCH_T_TEST:
            result = self.perform_t_test(variant_a_id, variant_b_id, equal_var=False)
        elif test_type == SignificanceTest.MANN_WHITNEY:
            result = self.perform_mann_whitney(variant_a_id, variant_b_id)
        else:
            raise ValueError(f"Test type {test_type} not implemented")

        self.results.append(result)
        return result

    def run_multi_variant_test(self,
                               test_type: SignificanceTest = SignificanceTest.WELCH_T_TEST) -> List[TestResult]:
        """
        Run pairwise tests for all variant combinations

        Applies Bonferroni correction for multiple comparisons.

        Returns:
            List of TestResult objects for all pairwise comparisons
        """
        variant_ids = list(self.variants.keys())
        n_comparisons = len(variant_ids) * (len(variant_ids) - 1) // 2

        # Bonferroni correction
        original_alpha = self.alpha
        self.alpha = original_alpha / n_comparisons

        results = []
        try:
            for i, var_a in enumerate(variant_ids):
                for var_b in variant_ids[i+1:]:
                    result = self.run_test(var_a, var_b, test_type)
                    results.append(result)
        finally:
            self.alpha = original_alpha

        return results

    def get_summary(self) -> Dict:
        """Get comprehensive test summary"""
        return {
            'test_name': self.test_name,
            'status': self.status.value,
            'alpha': self.alpha,
            'power': self.power,
            'minimum_detectable_effect': self.minimum_detectable_effect,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'variants': {
                vid: var.to_dict() for vid, var in self.variants.items()
            },
            'results': [r.to_dict() for r in self.results],
            'num_comparisons': len(self.results)
        }

    def export_results(self, filepath: str):
        """Export test results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

    def finalize_test(self):
        """Mark test as completed"""
        self.status = TestStatus.COMPLETED
        self.end_time = datetime.now()


class MultiVariantOptimizer:
    """
    Optimize across multiple variants using tournament-style testing

    Program of Thoughts:
    1. Start with all variants
    2. Run pairwise comparisons
    3. Eliminate statistically inferior variants
    4. Repeat until convergence or single winner
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.rounds: List[Dict] = []

    def run_tournament(self, variants_data: Dict[str, List[float]]) -> Dict:
        """
        Run tournament-style elimination

        Args:
            variants_data: Dictionary mapping variant IDs to sample lists

        Returns:
            Tournament results with winner
        """
        remaining_variants = set(variants_data.keys())
        round_num = 0

        while len(remaining_variants) > 1:
            round_num += 1
            round_tests = []

            # Test all pairs
            variant_list = list(remaining_variants)
            for i, var_a in enumerate(variant_list):
                for var_b in variant_list[i+1:]:
                    # Quick t-test
                    samples_a = variants_data[var_a]
                    samples_b = variants_data[var_b]

                    if len(samples_a) > 1 and len(samples_b) > 1:
                        stat, p_val = stats.ttest_ind(samples_a, samples_b)
                        mean_a = np.mean(samples_a)
                        mean_b = np.mean(samples_b)

                        round_tests.append({
                            'variant_a': var_a,
                            'variant_b': var_b,
                            'p_value': p_val,
                            'mean_a': mean_a,
                            'mean_b': mean_b,
                            'significant': p_val < self.alpha,
                            'winner': var_a if mean_a > mean_b else var_b
                        })

            # Find variants that never win
            win_counts = {v: 0 for v in remaining_variants}
            for test in round_tests:
                if test['significant']:
                    win_counts[test['winner']] += 1

            # Eliminate variants that never win significantly
            variants_to_eliminate = {
                v for v, wins in win_counts.items() if wins == 0 and len(win_counts) > 1
            }

            if not variants_to_eliminate:
                # No clear eliminations, keep best performer
                best_variant = max(remaining_variants,
                                 key=lambda v: np.mean(variants_data[v]))
                break

            remaining_variants -= variants_to_eliminate

            self.rounds.append({
                'round': round_num,
                'tests': round_tests,
                'eliminated': list(variants_to_eliminate),
                'remaining': list(remaining_variants)
            })

        winner = list(remaining_variants)[0] if remaining_variants else None

        return {
            'winner': winner,
            'rounds': self.rounds,
            'total_rounds': round_num
        }
