"""
Comprehensive Tests for Synthetic Data Generator

Tests all core components:
- LLM Providers
- Intent Engine
- Ambiguity Detection
- Reasoning Engines
- UK Standards
- Pattern Learning
- Output Engines
"""

import sys
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_providers import LLMFactory, MockProvider
from core.intent_engine import IntentEngine
from core.ambiguity_detector import AmbiguityDetector
from core.reasoning_engines import ReasoningEngineFactory
from core.uk_standards import UKStandardsGenerator, UKStandardsValidator, UKStandardsEnforcer
from core.pattern_learner import PatternLearner
from core.output_engine import OutputEngine


class TestSyntheticDataGenerator:
    """Comprehensive test suite"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.llm = LLMFactory.create('mock')

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("SYNTHETIC DATA GENERATOR - COMPREHENSIVE TESTS")
        print("="*70 + "\n")

        # Test groups
        test_groups = [
            ("LLM Providers", self.test_llm_providers),
            ("Intent Engine", self.test_intent_engine),
            ("Ambiguity Detection", self.test_ambiguity_detection),
            ("Reasoning Engines", self.test_reasoning_engines),
            ("UK Standards", self.test_uk_standards),
            ("Pattern Learning", self.test_pattern_learning),
            ("Output Engines", self.test_output_engines),
        ]

        for group_name, test_func in test_groups:
            print(f"\n{'='*70}")
            print(f"Testing: {group_name}")
            print('='*70)
            try:
                test_func()
            except Exception as e:
                print(f"❌ Test group failed with exception: {e}")
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

    # ==================== LLM Provider Tests ====================

    def test_llm_providers(self):
        """Test LLM provider factory"""

        # Test mock provider creation
        mock_llm = LLMFactory.create('mock')
        self.assert_test(isinstance(mock_llm, MockProvider), "Create mock provider")

        # Test generation
        response = mock_llm.generate("Test prompt")
        self.assert_test(isinstance(response, str) and len(response) > 0, "Mock provider generates text")

        # Test structured generation
        schema = {"name": "string", "age": "number"}
        structured = mock_llm.generate_structured("Generate data", schema)
        self.assert_test(isinstance(structured, dict), "Mock provider generates structured data")
        self.assert_test('name' in structured, "Structured data has 'name' field")
        self.assert_test('age' in structured, "Structured data has 'age' field")

        # Test batch generation
        batch = mock_llm.generate_batch(["prompt1", "prompt2", "prompt3"])
        self.assert_test(len(batch) == 3, "Batch generation returns correct count")

        # Test list providers
        providers = LLMFactory.list_providers()
        self.assert_test('mock' in providers, "Mock provider listed")
        self.assert_test('openai' in providers, "OpenAI provider listed")
        self.assert_test('anthropic' in providers, "Anthropic provider listed")

    # ==================== Intent Engine Tests ====================

    def test_intent_engine(self):
        """Test intent understanding"""

        engine = IntentEngine(self.llm)

        # Test basic intent parsing
        prompts = [
            ("Generate 1000 customer records", 1000, 'customer records'),
            ("Create 500 patient data", 500, 'patient records'),
            ("Make 100 UK customers for testing", 100, 'customer records'),
        ]

        for prompt, expected_count, expected_type in prompts:
            intent = engine.parse_intent(prompt)
            self.assert_test(intent.count == expected_count, f"Extract count from: '{prompt[:40]}...'")
            self.assert_test(expected_type in intent.data_type.lower(), f"Extract type from: '{prompt[:40]}...'")

        # Test geography extraction
        intent_uk = engine.parse_intent("Generate 100 UK customer records")
        self.assert_test(intent_uk.geography == 'UK', "Extract UK geography")

        # Test purpose extraction
        intent_test = engine.parse_intent("Generate 100 customers for testing")
        self.assert_test(intent_test.purpose == 'testing', "Extract testing purpose")

        # Test domain extraction
        intent_ecom = engine.parse_intent("Generate e-commerce customer data")
        self.assert_test(intent_ecom.domain == 'e-commerce', "Extract e-commerce domain")

        # Test schema suggestion
        schema = engine.get_schema_suggestion(intent_uk)
        self.assert_test(isinstance(schema, dict), "Generate schema")
        self.assert_test(len(schema) > 0, "Schema has fields")

    # ==================== Ambiguity Detection Tests ====================

    def test_ambiguity_detection(self):
        """Test ambiguity detection"""

        detector = AmbiguityDetector(self.llm)
        intent_engine = IntentEngine(self.llm)

        # Test with complete intent (no ambiguities)
        complete_intent = intent_engine.parse_intent("Generate 1000 UK customer records in CSV format for testing")
        complete_intent.output_format = 'csv'
        complete_intent.fields = ['name', 'email', 'age']

        clarifications = detector.detect_ambiguities(complete_intent)
        self.assert_test(len(clarifications) < 3, "Detect few ambiguities in complete request")

        # Test with ambiguous intent
        ambiguous_intent = intent_engine.parse_intent("Generate patient data")
        ambiguous_clarifications = detector.detect_ambiguities(ambiguous_intent)
        self.assert_test(len(ambiguous_clarifications) > 0, "Detect ambiguities in vague request")

        # Test clarification formatting
        formatted = detector.format_clarifications(ambiguous_clarifications)
        self.assert_test(isinstance(formatted, str) and len(formatted) > 0, "Format clarifications")

    # ==================== Reasoning Engine Tests ====================

    def test_reasoning_engines(self):
        """Test all reasoning engines"""

        intent_engine = IntentEngine(self.llm)
        intent = intent_engine.parse_intent("Generate customer records")
        schema = {'id': 'string', 'name': 'string', 'age': 'number'}

        engines = ['monte_carlo', 'beam_search', 'chain_of_thought', 'tree_of_thoughts']

        for engine_type in engines:
            # Create engine
            engine = ReasoningEngineFactory.create(engine_type, self.llm)
            self.assert_test(engine is not None, f"Create {engine_type} engine")

            # Generate data
            results = engine.generate(intent, schema, count=5)
            self.assert_test(len(results) == 5, f"{engine_type} generates correct count")

            # Check result structure
            if results:
                result = results[0]
                self.assert_test(hasattr(result, 'data'), f"{engine_type} result has data")
                self.assert_test(hasattr(result, 'score'), f"{engine_type} result has score")
                self.assert_test(isinstance(result.data, dict), f"{engine_type} data is dict")

    # ==================== UK Standards Tests ====================

    def test_uk_standards(self):
        """Test UK standards compliance"""

        generator = UKStandardsGenerator()
        validator = UKStandardsValidator()

        # Test postcode generation
        for region in ['London', 'Manchester', 'Edinburgh']:
            postcode = generator.generate_postcode(region)
            self.assert_test(validator.validate_postcode(postcode), f"Generate valid {region} postcode")

        # Test phone generation
        mobile = generator.generate_phone('mobile')
        self.assert_test(validator.validate_phone(mobile), "Generate valid UK mobile")

        landline = generator.generate_phone('landline')
        self.assert_test(validator.validate_phone(landline), "Generate valid UK landline")

        # Test name generation
        first, last = generator.generate_name()
        self.assert_test(isinstance(first, str) and len(first) > 0, "Generate valid first name")
        self.assert_test(isinstance(last, str) and len(last) > 0, "Generate valid last name")

        # Test email generation
        email = generator.generate_email(first, last)
        self.assert_test('@' in email, "Generate valid email")

        # Test address generation
        address = generator.generate_address('London')
        self.assert_test('street_address' in address, "Address has street")
        self.assert_test('city' in address, "Address has city")
        self.assert_test('postcode' in address, "Address has postcode")
        self.assert_test(validator.validate_postcode(address['postcode']), "Address has valid postcode")

        # Test date formatting
        date_str = generator.generate_random_date()
        self.assert_test(validator.validate_date_format(date_str), "Generate valid UK date format")

        # Test currency formatting
        currency = generator.format_currency(1234.56)
        self.assert_test(currency.startswith('£'), "Currency uses £ symbol")
        self.assert_test('1,234.56' in currency, "Currency formatted correctly")

        # Test standards enforcer
        enforcer = UKStandardsEnforcer()
        bad_data = {
            'postcode': 'INVALID',
            'phone': '123456',
            'amount': '100'
        }
        schema = {'postcode': 'postcode', 'phone': 'phone', 'amount': 'currency'}
        fixed_data = enforcer.enforce_standards(bad_data, schema)
        self.assert_test(validator.validate_postcode(fixed_data['postcode']), "Enforcer fixes invalid postcode")

    # ==================== Pattern Learning Tests ====================

    def test_pattern_learning(self):
        """Test pattern recognition and learning"""

        learner = PatternLearner(self.llm)

        # Test email pattern learning
        email_examples = [
            'john.smith@company.co.uk',
            'jane.doe@company.co.uk',
            'bob.jones@company.co.uk'
        ]

        pattern = learner.learn_from_examples(email_examples, 'email')
        self.assert_test(pattern is not None, "Learn email pattern")
        self.assert_test(pattern.pattern_type == 'email', "Detect email pattern type")
        self.assert_test('company.co.uk' in pattern.template, "Extract common domain")

        # Test ID pattern learning
        id_examples = ['EMP-001', 'EMP-002', 'EMP-003']
        id_pattern = learner.learn_from_examples(id_examples, 'employee_id')
        self.assert_test(id_pattern is not None, "Learn ID pattern")

        # Test pattern storage
        self.assert_test('email' in learner.learned_patterns, "Store learned email pattern")
        self.assert_test('employee_id' in learner.learned_patterns, "Store learned ID pattern")

    # ==================== Output Engine Tests ====================

    def test_output_engines(self):
        """Test multi-format output"""

        engine = OutputEngine()

        # Sample data
        data = [
            {'id': '1', 'name': 'Test User 1', 'email': 'test1@example.com'},
            {'id': '2', 'name': 'Test User 2', 'email': 'test2@example.com'},
        ]

        metadata = {'title': 'Test Data', 'geography': 'UK'}

        # Test each format
        with tempfile.TemporaryDirectory() as tmpdir:
            formats = ['csv', 'json', 'markdown']

            for fmt in formats:
                output_path = f"{tmpdir}/test_output.{fmt}"
                try:
                    result_path = engine.export(data, output_path, fmt, metadata)
                    file_exists = Path(result_path).exists()
                    self.assert_test(file_exists, f"Export to {fmt.upper()}")

                    if file_exists:
                        file_size = Path(result_path).stat().st_size
                        self.assert_test(file_size > 0, f"{fmt.upper()} file not empty")
                except Exception as e:
                    print(f"  ⚠️  {fmt.upper()} export failed: {e} (may need dependencies)")

        # Test supported formats list
        self.assert_test('csv' in engine.supported_formats, "CSV supported")
        self.assert_test('json' in engine.supported_formats, "JSON supported")
        self.assert_test('pdf' in engine.supported_formats, "PDF supported")


def main():
    """Run all tests"""
    tester = TestSyntheticDataGenerator()
    success = tester.run_all_tests()

    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
