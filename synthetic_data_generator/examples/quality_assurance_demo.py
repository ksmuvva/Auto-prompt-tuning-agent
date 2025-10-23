"""
Quality Assurance Layer - Demonstration

Shows how to use the QA layer to:
- Validate generated data
- Enforce constraints
- Calculate quality metrics
- Generate quality reports
- Fix violations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_providers import LLMFactory
from core.intent_engine import IntentEngine
from core.reasoning_engines import ReasoningEngineFactory
from core.quality_assurance import QualityAssuranceLayer, QualityEnforcer
from core.constraint_system import CommonConstraints


def demo_basic_validation():
    """Demo 1: Basic data validation"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Data Validation")
    print("="*70 + "\n")

    qa_layer = QualityAssuranceLayer()

    # Sample data with issues
    data = [
        {'name': 'John Smith', 'age': 30, 'email': 'john@example.com'},
        {'name': 'Jane Doe', 'age': 150, 'email': 'jane@example.com'},  # Age too high!
        {'name': 'Bob Jones', 'age': 35, 'email': 'invalid-email'},  # Invalid email!
        {'name': None, 'age': 25, 'email': 'alice@example.com'},  # Missing name!
    ]

    schema = {'name': 'string', 'age': 'number', 'email': 'string'}

    print("Original Data:")
    for i, record in enumerate(data, 1):
        print(f"  {i}. {record}")

    # Assess quality
    metrics = qa_layer.assess_quality(data, schema)

    print(f"\n{metrics}")

    print("\nIssues Found:")
    if metrics.errors:
        for error in metrics.errors[:5]:
            print(f"  ❌ {error}")

    print(f"\nOverall Quality: {metrics.overall_quality_score:.1%}")


def demo_constraint_enforcement():
    """Demo 2: Constraint enforcement and fixing"""
    print("\n" + "="*70)
    print("DEMO 2: Constraint Enforcement")
    print("="*70 + "\n")

    qa_layer = QualityAssuranceLayer()

    # Add constraints
    qa_layer.constraint_system.add_range_constraint('age', min_val=18, max_val=100)
    qa_layer.constraint_system.add_format_constraint(
        'email',
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    qa_layer.constraint_system.add_range_constraint('salary', min_val=15000, max_val=200000)

    # Data with constraint violations
    data = [
        {'name': 'Young Person', 'age': 15, 'email': 'young@example.com', 'salary': 50000},  # Age below min
        {'name': 'Old Person', 'age': 150, 'email': 'old@example.com', 'salary': 300000},  # Age & salary too high
        {'name': 'Valid Person', 'age': 30, 'email': 'valid@example.com', 'salary': 50000},  # All valid
    ]

    schema = {'name': 'string', 'age': 'number', 'email': 'string', 'salary': 'number'}

    print("Before Fixing:")
    for i, record in enumerate(data, 1):
        print(f"  {i}. Age: {record['age']}, Salary: {record['salary']}")

    # Fix violations
    fixed_data, metrics = qa_layer.validate_and_fix(data, schema, fix_violations=True)

    print("\nAfter Fixing:")
    for i, record in enumerate(fixed_data, 1):
        print(f"  {i}. Age: {record['age']}, Salary: {record['salary']}")

    print(f"\nConstraint Satisfaction: {metrics.constraint_satisfaction_score:.1%}")


def demo_quality_report():
    """Demo 3: Generate comprehensive quality report"""
    print("\n" + "="*70)
    print("DEMO 3: Quality Report Generation")
    print("="*70 + "\n")

    # Generate some data using reasoning engine
    llm = LLMFactory.create('mock')
    intent_engine = IntentEngine(llm)
    intent = intent_engine.parse_intent("Generate UK customer data for testing")

    schema = {
        'customer_id': 'string',
        'name': 'string',
        'email': 'email',
        'age': 'number',
        'postcode': 'string'
    }

    engine = ReasoningEngineFactory.create('monte_carlo', llm)
    results = engine.generate(intent, schema, count=10)
    data = [result.data for result in results]

    # Quality assessment
    qa_layer = QualityAssuranceLayer()
    qa_layer.add_constraints_from_schema(schema)

    report = qa_layer.generate_quality_report(data, schema)
    print(report)


def demo_quality_enforcer():
    """Demo 4: Quality enforcement with automatic fixing"""
    print("\n" + "="*70)
    print("DEMO 4: Quality Enforcement")
    print("="*70 + "\n")

    qa_layer = QualityAssuranceLayer()

    # Add constraints
    schema = {'name': 'string', 'age': 'number', 'email': 'string', 'salary': 'number'}
    qa_layer.add_constraints_from_schema(schema)
    qa_layer.constraint_system.add_range_constraint('salary', min_val=20000, max_val=150000)

    # Problematic data
    data = [
        {'name': 'Person 1', 'age': 200, 'email': 'p1@example.com', 'salary': 500000},
        {'name': 'Person 2', 'age': 30, 'email': 'p2@example.com', 'salary': 50000},
        {'name': 'Person 3', 'age': -5, 'email': 'invalid', 'salary': 1000},
    ]

    # Use quality enforcer
    enforcer = QualityEnforcer(qa_layer, min_quality=0.70)
    final_data, final_metrics, attempts = enforcer.ensure_quality(data, schema, max_attempts=3)

    print(f"Quality Enforcement Results:")
    print(f"  Attempts: {attempts}")
    print(f"  Final Quality: {final_metrics.overall_quality_score:.1%}")
    print(f"  Valid Records: {final_metrics.valid_records}/{final_metrics.total_records}")

    print("\nFixed Data:")
    for i, record in enumerate(final_data, 1):
        print(f"  {i}. Age: {record['age']}, Salary: {record['salary']}")


def demo_cross_field_validation():
    """Demo 5: Cross-field validation"""
    print("\n" + "="*70)
    print("DEMO 5: Cross-Field Validation")
    print("="*70 + "\n")

    qa_layer = QualityAssuranceLayer()

    # Data with cross-field issues
    data = [
        {
            'name': 'John Smith',
            'age': 30,
            'birth_date': '1995-01-01',  # Consistent
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',  # Consistent
            'country': 'UK',
            'postcode': 'SW1A 1AA',
            'state': None  # Consistent with UK
        },
        {
            'name': 'Jane Doe',
            'age': 50,
            'birth_date': '1995-01-01',  # Inconsistent! Should be ~30
            'start_date': '2023-12-31',
            'end_date': '2020-01-01',  # Inconsistent! End before start
            'country': 'UK',
            'postcode': 'INVALID',  # Invalid format
            'state': 'California'  # Inconsistent! UK doesn't have states
        },
    ]

    schema = {
        'name': 'string',
        'age': 'number',
        'birth_date': 'string',
        'start_date': 'string',
        'end_date': 'string',
        'country': 'string',
        'postcode': 'string',
        'state': 'string'
    }

    print("Validating Cross-Field Consistency:\n")

    for i, record in enumerate(data, 1):
        metrics = qa_layer.assess_quality([record], schema)

        print(f"Record {i}:")
        if metrics.errors:
            for error in metrics.errors:
                print(f"  ❌ {error}")
        else:
            print(f"  ✓ All cross-field checks passed")

        if metrics.warnings:
            for warning in metrics.warnings:
                print(f"  ⚠️  {warning}")

        print()


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("QUALITY ASSURANCE LAYER - DEMONSTRATIONS")
    print("="*70)

    demos = [
        demo_basic_validation,
        demo_constraint_enforcement,
        demo_quality_report,
        demo_quality_enforcer,
        demo_cross_field_validation,
    ]

    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("DEMONSTRATIONS COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
