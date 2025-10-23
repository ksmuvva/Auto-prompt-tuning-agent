"""
Quality Assurance Layer

Comprehensive quality checks for generated data:
- Validation (format, type, cross-field)
- Constraint satisfaction
- Quality metrics (completeness, consistency, diversity)
- Quality reports
- Automated fixing
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import statistics

from .validators import DataValidator, ValidationResult
from .constraint_system import ConstraintSatisfactionSystem


@dataclass
class QualityMetrics:
    """Quality metrics for generated data"""
    total_records: int
    valid_records: int
    completeness_score: float  # % of non-null fields
    consistency_score: float   # % passing cross-field validation
    format_validity_score: float  # % with valid formats
    diversity_score: float     # Shannon entropy / uniqueness
    constraint_satisfaction_score: float  # % satisfying constraints
    overall_quality_score: float

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_quality: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"""
Quality Metrics:
  Total Records: {self.total_records}
  Valid Records: {self.valid_records} ({self.valid_records/self.total_records*100:.1f}%)

  Scores:
    Overall Quality:      {self.overall_quality_score:.1%}
    Completeness:         {self.completeness_score:.1%}
    Consistency:          {self.consistency_score:.1%}
    Format Validity:      {self.format_validity_score:.1%}
    Diversity:            {self.diversity_score:.1%}
    Constraint Satisfaction: {self.constraint_satisfaction_score:.1%}

  Errors: {len(self.errors)}
  Warnings: {len(self.warnings)}
"""


class QualityAssuranceLayer:
    """Comprehensive quality assurance for generated data"""

    def __init__(self):
        self.validator = DataValidator()
        self.constraint_system = ConstraintSatisfactionSystem()
        self.quality_threshold = 0.80  # 80% minimum quality

    def add_constraints_from_schema(self, schema: Dict[str, str]):
        """Add common constraints based on schema"""
        for field, field_type in schema.items():
            field_lower = field.lower()

            # Add type-specific constraints
            if field_lower in ['age', 'years']:
                self.constraint_system.add_range_constraint(field, min_val=0, max_val=120)

            elif 'price' in field_lower or 'amount' in field_lower or 'salary' in field_lower:
                self.constraint_system.add_range_constraint(field, min_val=0)

            elif 'email' in field_lower:
                self.constraint_system.add_format_constraint(
                    field,
                    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                )

            elif 'postcode' in field_lower:
                self.constraint_system.add_format_constraint(
                    field,
                    r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$'
                )

    def assess_quality(
        self,
        data: List[Dict[str, Any]],
        schema: Dict[str, str] = None
    ) -> QualityMetrics:
        """
        Comprehensive quality assessment

        Args:
            data: Generated data records
            schema: Optional schema for validation

        Returns:
            QualityMetrics with detailed scores
        """
        if not data:
            return QualityMetrics(
                total_records=0,
                valid_records=0,
                completeness_score=0.0,
                consistency_score=0.0,
                format_validity_score=0.0,
                diversity_score=0.0,
                constraint_satisfaction_score=0.0,
                overall_quality_score=0.0,
                errors=["No data provided"]
            )

        # Run all quality checks
        validation_results = self._validate_all(data, schema)
        completeness = self._calculate_completeness(data)
        consistency = self._calculate_consistency(validation_results)
        format_validity = self._calculate_format_validity(validation_results)
        diversity = self._calculate_diversity(data)
        constraint_satisfaction = self._check_constraint_satisfaction(data)

        # Calculate overall quality score
        overall = (
            completeness * 0.20 +
            consistency * 0.25 +
            format_validity * 0.20 +
            diversity * 0.15 +
            constraint_satisfaction * 0.20
        )

        # Count valid records
        valid_count = sum(1 for result in validation_results if result.is_valid)

        # Collect errors and warnings
        all_errors = []
        all_warnings = []
        for result in validation_results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        # Field-level quality
        field_quality = self._calculate_field_quality(data, validation_results)

        return QualityMetrics(
            total_records=len(data),
            valid_records=valid_count,
            completeness_score=completeness,
            consistency_score=consistency,
            format_validity_score=format_validity,
            diversity_score=diversity,
            constraint_satisfaction_score=constraint_satisfaction,
            overall_quality_score=overall,
            errors=all_errors[:10],  # First 10 errors
            warnings=all_warnings[:10],  # First 10 warnings
            field_quality=field_quality
        )

    def validate_and_fix(
        self,
        data: List[Dict[str, Any]],
        schema: Dict[str, str] = None,
        fix_violations: bool = True
    ) -> Tuple[List[Dict[str, Any]], QualityMetrics]:
        """
        Validate data and optionally fix violations

        Args:
            data: Generated data
            schema: Optional schema
            fix_violations: Whether to attempt fixing violations

        Returns:
            (fixed_data, quality_metrics)
        """
        if fix_violations:
            fixed_data = []
            for record in data:
                # Fix constraint violations
                fixed_record = self.constraint_system.fix_violations(record)
                fixed_data.append(fixed_record)
        else:
            fixed_data = data

        # Assess quality
        metrics = self.assess_quality(fixed_data, schema)

        return fixed_data, metrics

    def generate_quality_report(
        self,
        data: List[Dict[str, Any]],
        schema: Dict[str, str] = None
    ) -> str:
        """Generate detailed quality report"""

        metrics = self.assess_quality(data, schema)

        report = []
        report.append("="*70)
        report.append("DATA QUALITY ASSURANCE REPORT")
        report.append("="*70)
        report.append(str(metrics))

        # Field-level quality
        if metrics.field_quality:
            report.append("\nField-Level Quality:")
            for field, quality in sorted(metrics.field_quality.items(), key=lambda x: x[1]):
                status = "✓" if quality >= 0.8 else ("⚠️" if quality >= 0.6 else "✗")
                report.append(f"  {status} {field}: {quality:.1%}")

        # Top errors
        if metrics.errors:
            report.append("\nTop Errors:")
            error_counts = Counter(metrics.errors)
            for error, count in error_counts.most_common(5):
                report.append(f"  • {error} ({count} occurrences)")

        # Top warnings
        if metrics.warnings:
            report.append("\nTop Warnings:")
            warning_counts = Counter(metrics.warnings)
            for warning, count in warning_counts.most_common(5):
                report.append(f"  • {warning} ({count} occurrences)")

        # Recommendations
        report.append("\nRecommendations:")
        if metrics.overall_quality_score < 0.6:
            report.append("  ⚠️  CRITICAL: Overall quality below 60%. Regenerate data.")
        elif metrics.overall_quality_score < 0.8:
            report.append("  ⚠️  WARNING: Quality below 80%. Review and fix issues.")
        else:
            report.append("  ✓ Quality acceptable for use.")

        if metrics.completeness_score < 0.9:
            report.append("  • Improve completeness: Too many null/missing values")

        if metrics.diversity_score < 0.5:
            report.append("  • Improve diversity: Data is too similar")

        if metrics.constraint_satisfaction_score < 0.9:
            report.append("  • Fix constraint violations: Use validate_and_fix()")

        report.append("="*70)

        return "\n".join(report)

    # ==================== Private Methods ====================

    def _validate_all(self, data: List[Dict[str, Any]], schema: Dict[str, str]) -> List[ValidationResult]:
        """Validate all records"""
        return [self.validator.validate_record(record, schema) for record in data]

    def _calculate_completeness(self, data: List[Dict[str, Any]]) -> float:
        """Calculate completeness score (% of non-null fields)"""
        if not data:
            return 0.0

        total_fields = 0
        non_null_fields = 0

        for record in data:
            for value in record.values():
                total_fields += 1
                if value is not None and value != '' and str(value).lower() not in ['null', 'none', 'nan']:
                    non_null_fields += 1

        return non_null_fields / total_fields if total_fields > 0 else 0.0

    def _calculate_consistency(self, validation_results: List[ValidationResult]) -> float:
        """Calculate consistency score (% passing cross-field checks)"""
        if not validation_results:
            return 0.0

        # Count records with no consistency errors
        consistent_count = sum(1 for result in validation_results if len(result.errors) == 0)
        return consistent_count / len(validation_results)

    def _calculate_format_validity(self, validation_results: List[ValidationResult]) -> float:
        """Calculate format validity score"""
        if not validation_results:
            return 0.0

        # Count records with no format errors (check error messages)
        valid_format_count = 0
        for result in validation_results:
            has_format_error = any('format' in error.lower() or 'invalid' in error.lower()
                                 for error in result.errors)
            if not has_format_error:
                valid_format_count += 1

        return valid_format_count / len(validation_results)

    def _calculate_diversity(self, data: List[Dict[str, Any]]) -> float:
        """Calculate diversity score (Shannon entropy of field values)"""
        if not data or len(data) < 2:
            return 0.0

        # Calculate diversity for each field
        field_diversities = []

        # Get all fields
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())

        for field in all_fields:
            if field.startswith('_'):  # Skip metadata fields
                continue

            # Get all values for this field
            values = [record.get(field) for record in data if field in record]
            if not values:
                continue

            # Calculate uniqueness ratio
            unique_count = len(set(str(v) for v in values))
            diversity_ratio = unique_count / len(values)
            field_diversities.append(diversity_ratio)

        return statistics.mean(field_diversities) if field_diversities else 0.0

    def _check_constraint_satisfaction(self, data: List[Dict[str, Any]]) -> float:
        """Check constraint satisfaction rate"""
        if not data or not self.constraint_system.constraints:
            return 1.0  # No constraints = all satisfied

        satisfied_count = 0
        total_count = 0

        for record in data:
            is_valid, errors = self.constraint_system.validate_record(record)
            total_count += 1
            if is_valid:
                satisfied_count += 1

        return satisfied_count / total_count if total_count > 0 else 1.0

    def _calculate_field_quality(
        self,
        data: List[Dict[str, Any]],
        validation_results: List[ValidationResult]
    ) -> Dict[str, float]:
        """Calculate quality score for each field"""
        field_quality = {}

        # Get all fields
        all_fields = set()
        for record in data:
            all_fields.update(k for k in record.keys() if not k.startswith('_'))

        for field in all_fields:
            # Completeness for this field
            non_null_count = sum(1 for record in data
                                if field in record and record[field] is not None and record[field] != '')
            completeness = non_null_count / len(data)

            # Error rate for this field
            error_count = sum(1 for result in validation_results
                            if field in result.field_errors)
            error_rate = error_count / len(data)

            # Diversity for this field
            values = [str(record.get(field)) for record in data if field in record]
            diversity = len(set(values)) / len(values) if values else 0.0

            # Combine scores
            quality = (completeness * 0.4 + (1 - error_rate) * 0.4 + diversity * 0.2)
            field_quality[field] = quality

        return field_quality


class QualityEnforcer:
    """Enforces quality standards during generation"""

    def __init__(self, qa_layer: QualityAssuranceLayer, min_quality: float = 0.80):
        self.qa_layer = qa_layer
        self.min_quality = min_quality

    def ensure_quality(
        self,
        data: List[Dict[str, Any]],
        schema: Dict[str, str] = None,
        max_attempts: int = 3
    ) -> Tuple[List[Dict[str, Any]], QualityMetrics, int]:
        """
        Ensure data meets quality standards, regenerating if needed

        Args:
            data: Generated data
            schema: Optional schema
            max_attempts: Maximum fix attempts

        Returns:
            (final_data, quality_metrics, attempts_made)
        """
        current_data = data
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Fix violations
            fixed_data, metrics = self.qa_layer.validate_and_fix(current_data, schema, fix_violations=True)

            # Check if quality is acceptable
            if metrics.overall_quality_score >= self.min_quality:
                return fixed_data, metrics, attempts

            # Quality too low, would need to regenerate
            # For now, return best effort
            current_data = fixed_data

        # Return best effort after max attempts
        return current_data, metrics, attempts
