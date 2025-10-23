"""
Tests for Quality Assurance Layer, Validators, and Constraint System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.quality_assurance import QualityAssuranceLayer, QualityEnforcer
from core.constraint_system import ConstraintSatisfactionSystem, ConstraintType, CommonConstraints
from core.validators import DataValidator, FormatValidator, TypeValidator, CrossFieldValidator


class TestQualityAssurance:
    """Test quality assurance components"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("QUALITY ASSURANCE LAYER - TESTS")
        print("="*70 + "\n")

        test_groups = [
            ("Format Validators", self.test_format_validators),
            ("Type Validators", self.test_type_validators),
            ("Cross-Field Validators", self.test_cross_field_validators),
            ("Constraint System", self.test_constraint_system),
            ("Quality Metrics", self.test_quality_metrics),
            ("Quality Assurance Layer", self.test_qa_layer),
        ]

        for group_name, test_func in test_groups:
            print(f"\n{'='*70}")
            print(f"Testing: {group_name}")
            print('='*70)
            try:
                test_func()
            except Exception as e:
                print(f"❌ Test group failed with exception: {e}")
                import traceback
                traceback.print_exc()
                self.failed += 1

        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"✓ Passed: {self.passed}")
        print(f"✗ Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        print("="*70 + "\n")

        return self.failed == 0

    def assert_test(self, condition: bool, test_name: str):
        """Assert a test condition"""
        if condition:
            print(f"  ✓ {test_name}")
            self.passed += 1
        else:
            print(f"  ✗ {test_name}")
            self.failed += 1

    # ==================== Format Validators ====================

    def test_format_validators(self):
        """Test format validators"""
        validator = FormatValidator()

        # Email validation
        is_valid, _ = validator.validate_email("user@example.com")
        self.assert_test(is_valid, "Valid email accepted")

        is_valid, _ = validator.validate_email("invalid-email")
        self.assert_test(not is_valid, "Invalid email rejected")

        # UK postcode validation
        is_valid, _ = validator.validate_uk_postcode("SW1A 1AA")
        self.assert_test(is_valid, "Valid UK postcode accepted")

        is_valid, _ = validator.validate_uk_postcode("INVALID")
        self.assert_test(not is_valid, "Invalid postcode rejected")

        # UK phone validation
        is_valid, _ = validator.validate_uk_phone("07700 900 123")
        self.assert_test(is_valid, "Valid UK mobile accepted")

        is_valid, _ = validator.validate_uk_phone("020 7123 4567")
        self.assert_test(is_valid, "Valid UK landline accepted")

        is_valid, _ = validator.validate_uk_phone("123456")
        self.assert_test(not is_valid, "Invalid phone rejected")

        # Currency validation
        is_valid, _ = validator.validate_currency("£1,234.56")
        self.assert_test(is_valid, "Valid currency accepted")

        is_valid, _ = validator.validate_currency("1234.56")
        self.assert_test(is_valid, "Numeric currency accepted")

    # ==================== Type Validators ====================

    def test_type_validators(self):
        """Test type validators"""
        validator = TypeValidator()

        # Type validation
        is_valid, _ = validator.validate_type("hello", "string")
        self.assert_test(is_valid, "String type validated")

        is_valid, _ = validator.validate_type(42, "number")
        self.assert_test(is_valid, "Number type validated")

        is_valid, _ = validator.validate_type(42, "string")
        self.assert_test(not is_valid, "Wrong type rejected")

        # Schema validation
        record = {
            'name': 'John',
            'age': 30,
            'email': 'john@example.com'
        }
        schema = {
            'name': 'string',
            'age': 'number',
            'email': 'string'
        }
        is_valid, errors = validator.validate_schema(record, schema)
        self.assert_test(is_valid, "Valid schema accepted")
        self.assert_test(len(errors) == 0, "No schema errors")

        # Missing field
        incomplete_record = {'name': 'John'}
        is_valid, errors = validator.validate_schema(incomplete_record, schema)
        self.assert_test(not is_valid, "Incomplete record rejected")
        self.assert_test(len(errors) > 0, "Schema errors detected")

    # ==================== Cross-Field Validators ====================

    def test_cross_field_validators(self):
        """Test cross-field validators"""
        validator = CrossFieldValidator()

        # Age vs birth date
        is_valid, _ = validator.validate_age_vs_birth_date(30, "1995-01-01", 2025)
        self.assert_test(is_valid, "Age matches birth date")

        is_valid, _ = validator.validate_age_vs_birth_date(50, "1995-01-01", 2025)
        self.assert_test(not is_valid, "Age mismatch detected")

        # Date order
        record = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
        is_valid, _ = validator.validate_dates_order('start_date', 'end_date', record)
        self.assert_test(is_valid, "Correct date order accepted")

        bad_record = {
            'start_date': '2023-12-31',
            'end_date': '2023-01-01'
        }
        is_valid, _ = validator.validate_dates_order('start_date', 'end_date', bad_record)
        self.assert_test(not is_valid, "Wrong date order rejected")

        # Geography consistency
        uk_record = {
            'country': 'UK',
            'postcode': 'SW1A 1AA',
            'state': None
        }
        is_valid, errors = validator.validate_geography_consistency(uk_record)
        self.assert_test(is_valid, "UK geography consistent")

        bad_uk_record = {
            'country': 'UK',
            'postcode': 'INVALID',
            'state': 'California'
        }
        is_valid, errors = validator.validate_geography_consistency(bad_uk_record)
        self.assert_test(not is_valid, "UK geography inconsistency detected")
        self.assert_test(len(errors) >= 2, "Multiple geography errors detected")

    # ==================== Constraint System ====================

    def test_constraint_system(self):
        """Test constraint satisfaction system"""
        system = ConstraintSatisfactionSystem()

        # Range constraint
        system.add_range_constraint('age', min_val=0, max_val=120)

        is_valid, errors = system.validate_field('age', 30)
        self.assert_test(is_valid, "Valid age accepted")

        is_valid, errors = system.validate_field('age', 150)
        self.assert_test(not is_valid, "Age over max rejected")
        self.assert_test(len(errors) > 0, "Range error reported")

        # Format constraint
        system.add_format_constraint('email', r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

        is_valid, errors = system.validate_field('email', 'user@example.com')
        self.assert_test(is_valid, "Valid email format accepted")

        is_valid, errors = system.validate_field('email', 'invalid')
        self.assert_test(not is_valid, "Invalid email format rejected")

        # Record validation
        record = {
            'age': 30,
            'email': 'user@example.com'
        }
        is_valid, all_errors = system.validate_record(record)
        self.assert_test(is_valid, "Valid record passes all constraints")

        bad_record = {
            'age': 150,
            'email': 'invalid'
        }
        is_valid, all_errors = system.validate_record(bad_record)
        self.assert_test(not is_valid, "Invalid record fails constraints")
        self.assert_test(len(all_errors) == 2, "All constraint violations detected")

        # Fix violations
        fixed_record = system.fix_violations(bad_record)
        self.assert_test(fixed_record['age'] == 120, "Age clamped to max")

    # ==================== Quality Metrics ====================

    def test_quality_metrics(self):
        """Test quality metrics calculation"""
        qa_layer = QualityAssuranceLayer()

        # Good quality data
        good_data = [
            {'name': 'John', 'age': 30, 'email': 'john@example.com'},
            {'name': 'Jane', 'age': 25, 'email': 'jane@example.com'},
            {'name': 'Bob', 'age': 35, 'email': 'bob@example.com'},
        ]
        schema = {'name': 'string', 'age': 'number', 'email': 'string'}

        metrics = qa_layer.assess_quality(good_data, schema)
        self.assert_test(metrics.total_records == 3, "Correct record count")
        self.assert_test(metrics.completeness_score > 0.9, "High completeness")
        self.assert_test(metrics.diversity_score > 0.5, "Good diversity")

        # Poor quality data
        poor_data = [
            {'name': 'John', 'age': 30, 'email': 'invalid'},  # Bad email
            {'name': None, 'age': 200, 'email': 'test@example.com'},  # Missing name, age too high
            {'name': 'Bob', 'age': 35, 'email': None},  # Missing email
        ]

        metrics = qa_layer.assess_quality(poor_data, schema)
        self.assert_test(metrics.completeness_score < 0.9, "Low completeness detected")
        self.assert_test(len(metrics.errors) > 0, "Errors detected")

    # ==================== Quality Assurance Layer ====================

    def test_qa_layer(self):
        """Test full QA layer"""
        qa_layer = QualityAssuranceLayer()

        # Add constraints
        schema = {'name': 'string', 'age': 'number', 'email': 'string'}
        qa_layer.add_constraints_from_schema(schema)

        # Test data
        data = [
            {'name': 'John Smith', 'age': 30, 'email': 'john@example.com'},
            {'name': 'Jane Doe', 'age': 150, 'email': 'jane@example.com'},  # Age too high
            {'name': 'Bob Jones', 'age': 35, 'email': 'invalid-email'},  # Invalid email
        ]

        # Validate and fix
        fixed_data, metrics = qa_layer.validate_and_fix(data, schema, fix_violations=True)

        self.assert_test(len(fixed_data) == 3, "All records processed")
        self.assert_test(fixed_data[1]['age'] == 120, "Age violation fixed")

        # Generate report
        report = qa_layer.generate_quality_report(data, schema)
        self.assert_test('Quality Metrics' in report, "Report generated")
        self.assert_test('Overall Quality' in report, "Report includes quality score")

        # Quality enforcer
        enforcer = QualityEnforcer(qa_layer, min_quality=0.70)
        final_data, final_metrics, attempts = enforcer.ensure_quality(data, schema)

        self.assert_test(len(final_data) == 3, "Quality enforcer processed data")
        self.assert_test(attempts >= 1, "Quality enforcement attempted")


def main():
    """Run all tests"""
    tester = TestQualityAssurance()
    success = tester.run_all_tests()

    if success:
        print("🎉 All Quality Assurance tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
