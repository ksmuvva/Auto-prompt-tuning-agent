"""
Basic Usage Examples for Synthetic Data Generator

Demonstrates core functionality with different LLM providers and reasoning engines
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_providers import LLMFactory
from core.intent_engine import IntentEngine
from core.ambiguity_detector import AmbiguityDetector
from core.reasoning_engines import ReasoningEngineFactory
from core.uk_standards import UKStandardsEnforcer
from core.output_engine import OutputEngine


def example_1_basic_generation():
    """Example 1: Basic data generation with mock provider"""

    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Generation (Mock Provider)")
    print("="*70 + "\n")

    # Initialize LLM (using mock for demo - no API key needed)
    llm = LLMFactory.create('mock')

    # Create intent engine
    intent_engine = IntentEngine(llm)

    # Parse user request
    prompt = "Generate 100 UK customer records for e-commerce testing"
    print(f"User Request: {prompt}\n")

    intent = intent_engine.parse_intent(prompt)
    print(f"Parsed Intent:")
    print(f"  Data Type: {intent.data_type}")
    print(f"  Count: {intent.count}")
    print(f"  Geography: {intent.geography}")
    print(f"  Purpose: {intent.purpose}")
    print(f"  Domain: {intent.domain}")

    # Get schema
    schema = intent_engine.get_schema_suggestion(intent)
    print(f"\nSuggested Schema: {schema}")

    # Generate data using Monte Carlo
    engine = ReasoningEngineFactory.create('monte_carlo', llm)
    results = engine.generate(intent, schema, count=10)  # Generate 10 for demo

    print(f"\n✓ Generated {len(results)} records using Monte Carlo sampling")
    print(f"Sample record: {results[0].data}")


def example_2_multi_reasoning():
    """Example 2: Compare different reasoning engines"""

    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Reasoning Comparison")
    print("="*70 + "\n")

    llm = LLMFactory.create('mock')
    intent_engine = IntentEngine(llm)

    prompt = "Generate customer data for UK retail"
    intent = intent_engine.parse_intent(prompt)
    schema = {'customer_id': 'string', 'name': 'string', 'email': 'email', 'age': 'number'}

    # Test different reasoning engines
    engines = ['monte_carlo', 'beam_search', 'chain_of_thought', 'tree_of_thoughts']

    for engine_type in engines:
        print(f"\n{engine_type.upper()}:")
        engine = ReasoningEngineFactory.create(engine_type, llm)
        results = engine.generate(intent, schema, count=3)
        print(f"  ✓ Generated {len(results)} records")
        if results:
            print(f"  Sample: {results[0].data}")
            if results[0].reasoning:
                print(f"  Reasoning: {results[0].reasoning[:100]}...")


def example_3_uk_standards():
    """Example 3: UK Standards Compliance"""

    print("\n" + "="*70)
    print("EXAMPLE 3: UK Standards Compliance")
    print("="*70 + "\n")

    from core.uk_standards import UKStandardsGenerator

    uk_gen = UKStandardsGenerator()

    print("UK Postcodes:")
    for region in ['London', 'Manchester', 'Edinburgh']:
        postcode = uk_gen.generate_postcode(region)
        print(f"  {region}: {postcode}")

    print("\nUK Phone Numbers:")
    print(f"  Mobile: {uk_gen.generate_phone('mobile')}")
    print(f"  Landline: {uk_gen.generate_phone('landline')}")

    print("\nUK Names (Diverse Demographics):")
    for i in range(5):
        first, last = uk_gen.generate_name()
        email = uk_gen.generate_email(first, last)
        print(f"  {first} {last} - {email}")

    print("\nUK Addresses:")
    for region in ['London', 'Birmingham']:
        address = uk_gen.generate_address(region)
        print(f"  {address['street_address']}, {address['city']}, {address['postcode']}")


def example_4_multi_format_output():
    """Example 4: Export to multiple formats"""

    print("\n" + "="*70)
    print("EXAMPLE 4: Multi-Format Output")
    print("="*70 + "\n")

    # Generate sample data
    data = [
        {
            'customer_id': f'CUST{i:04d}',
            'name': f'Customer {i}',
            'email': f'customer{i}@example.com',
            'age': 25 + i,
            'postcode': 'SW1A 1AA'
        }
        for i in range(1, 11)
    ]

    metadata = {
        'title': 'Sample Customer Data',
        'geography': 'UK',
        'purpose': 'testing'
    }

    output_engine = OutputEngine()

    # Export to different formats
    formats = ['csv', 'json', 'markdown']

    for fmt in formats:
        output_path = f'examples/sample_output.{fmt}'
        try:
            path = output_engine.export(data, output_path, fmt, metadata)
            print(f"  ✓ Exported to {fmt.upper()}: {path}")
        except Exception as e:
            print(f"  ✗ Failed to export to {fmt.upper()}: {e}")


def example_5_ambiguity_detection():
    """Example 5: Ambiguity Detection"""

    print("\n" + "="*70)
    print("EXAMPLE 5: Ambiguity Detection")
    print("="*70 + "\n")

    llm = LLMFactory.create('mock')
    intent_engine = IntentEngine(llm)
    ambiguity_detector = AmbiguityDetector(llm)

    # Ambiguous prompt
    prompt = "Generate patient data"
    print(f"User Request: {prompt}\n")

    intent = intent_engine.parse_intent(prompt)

    clarifications = ambiguity_detector.detect_ambiguities(intent)

    if clarifications:
        print(f"Detected {len(clarifications)} ambiguities:\n")
        print(ambiguity_detector.format_clarifications(clarifications))
    else:
        print("No ambiguities detected!")


def example_6_pattern_learning():
    """Example 6: Pattern Learning"""

    print("\n" + "="*70)
    print("EXAMPLE 6: Pattern Learning")
    print("="*70 + "\n")

    from core.pattern_learner import PatternLearner

    llm = LLMFactory.create('mock')
    learner = PatternLearner(llm)

    # Learn email pattern from examples
    email_examples = [
        'john.smith@company.co.uk',
        'j.doe@company.co.uk',
        'alice.jones@company.co.uk'
    ]

    print("Learning from examples:")
    for email in email_examples:
        print(f"  - {email}")

    pattern = learner.learn_from_examples(email_examples, 'email')

    print(f"\nLearned Pattern:")
    print(f"  Type: {pattern.pattern_type}")
    print(f"  Template: {pattern.template}")
    print(f"  Metadata: {pattern.metadata}")


def main():
    """Run all examples"""

    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATOR - USAGE EXAMPLES")
    print("="*70)

    examples = [
        ("Basic Generation", example_1_basic_generation),
        ("Multi-Reasoning", example_2_multi_reasoning),
        ("UK Standards", example_3_uk_standards),
        ("Multi-Format Output", example_4_multi_format_output),
        ("Ambiguity Detection", example_5_ambiguity_detection),
        ("Pattern Learning", example_6_pattern_learning),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
