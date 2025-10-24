"""
Comprehensive Test Suite for Explainability Features

Tests all explainability components:
- Feature importance analysis
- Decision rule extraction
- SHAP explanations
- LIME explanations
- MCTS reasoning engine
- Hybrid reasoning engine
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_providers import LLMFactory
from core.intent_engine import IntentEngine, Intent
from core.reasoning_engines import (
    ReasoningEngineFactory,
    MCTSEngine,
    HybridReasoningEngine
)
from core.explainability import (
    ExplainabilityEngine,
    FeatureImportanceAnalyzer,
    DecisionTreeExtractor,
    SHAPExplainer,
    LIMEExplainer
)
from core.explainable_generator import ExplainableSyntheticGenerator


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance analysis"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')
        self.analyzer = FeatureImportanceAnalyzer(self.llm)

    def test_analyze_importance(self):
        """Test basic feature importance calculation"""
        data = [
            {'age': 25, 'salary': 50000, 'name': 'Alice'},
            {'age': 35, 'salary': 75000, 'name': 'Bob'},
            {'age': 45, 'salary': 100000, 'name': 'Charlie'}
        ]
        schema = {'age': 'number', 'salary': 'number', 'name': 'string'}

        importances = self.analyzer.analyze_importance(data, schema, None, [])

        self.assertGreater(len(importances), 0)
        self.assertTrue(all(0 <= f.importance_score <= 1 for f in importances))
        self.assertTrue(all(f.feature_name in schema for f in importances))

    def test_contribution_types(self):
        """Test contribution type determination"""
        data = [
            {'x': i, 'y': i * 2, 'z': i ** 2} for i in range(10)
        ]
        schema = {'x': 'number', 'y': 'number', 'z': 'number'}

        importances = self.analyzer.analyze_importance(data, schema, None, [])

        # Should detect interactive relationships
        contribution_types = {f.feature_name: f.contribution_type for f in importances}
        self.assertIn('interactive', contribution_types.values())

    def test_empty_data(self):
        """Test with empty data"""
        data = []
        schema = {'field1': 'string', 'field2': 'number'}

        importances = self.analyzer.analyze_importance(data, schema, None, [])
        self.assertEqual(len(importances), 0)


class TestDecisionTreeExtractor(unittest.TestCase):
    """Test decision rule extraction"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')
        self.extractor = DecisionTreeExtractor(self.llm)

    def test_extract_rules(self):
        """Test rule extraction from data"""
        data = [
            {'age': 25, 'income': 50000, 'approved': 'yes'},
            {'age': 35, 'income': 75000, 'approved': 'yes'},
            {'age': 45, 'income': 100000, 'approved': 'yes'}
        ]
        schema = {'age': 'number', 'income': 'number', 'approved': 'string'}

        rules = self.extractor.extract_rules(data, schema, [])

        self.assertGreater(len(rules), 0)
        self.assertTrue(all(0 <= r.confidence <= 1 for r in rules))
        self.assertTrue(all(r.support >= 0 for r in rules))

    def test_numeric_rules(self):
        """Test rules for numeric features"""
        data = [{'value': i} for i in range(100)]
        schema = {'value': 'number'}

        rules = self.extractor.extract_rules(data, schema, [])

        # Should extract statistical rules
        self.assertGreater(len(rules), 0)
        self.assertTrue(any('Normal' in r.action for r in rules))

    def test_categorical_rules(self):
        """Test rules for categorical features"""
        data = [
            {'category': 'A'} for _ in range(50)
        ] + [
            {'category': 'B'} for _ in range(30)
        ]
        schema = {'category': 'string'}

        rules = self.extractor.extract_rules(data, schema, [])

        # Should extract probability rules
        self.assertGreater(len(rules), 0)


class TestSHAPExplainer(unittest.TestCase):
    """Test SHAP explanations"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')
        self.explainer = SHAPExplainer(self.llm)

    def test_explain(self):
        """Test SHAP explanation generation"""
        data = [
            {'age': 25, 'salary': 50000},
            {'age': 35, 'salary': 75000},
            {'age': 45, 'salary': 100000}
        ]
        schema = {'age': 'number', 'salary': 'number'}

        explanations = self.explainer.explain(data, schema, None, num_samples=3)

        self.assertEqual(len(explanations), 3)
        for exp in explanations:
            self.assertIsNotNone(exp.shap_values)
            self.assertIsNotNone(exp.feature_contributions)
            self.assertEqual(len(exp.shap_values), len(schema))

    def test_shap_values_sum(self):
        """Test SHAP values consistency"""
        data = [
            {'x': 10, 'y': 20, 'z': 30}
        ]
        schema = {'x': 'number', 'y': 'number', 'z': 'number'}

        explanations = self.explainer.explain(data, schema, None, num_samples=1)

        exp = explanations[0]
        # SHAP values should sum to total prediction
        self.assertAlmostEqual(
            sum(exp.shap_values.values()),
            exp.total_prediction,
            places=5
        )


class TestLIMEExplainer(unittest.TestCase):
    """Test LIME explanations"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')
        self.explainer = LIMEExplainer(self.llm)

    def test_explain(self):
        """Test LIME explanation generation"""
        data = [
            {'age': 25, 'salary': 50000, 'city': 'London'},
            {'age': 35, 'salary': 75000, 'city': 'Manchester'},
            {'age': 45, 'salary': 100000, 'city': 'Edinburgh'}
        ]
        schema = {'age': 'number', 'salary': 'number', 'city': 'string'}

        explanations = self.explainer.explain(data, schema, None, num_samples=3)

        self.assertEqual(len(explanations), 3)
        for exp in explanations:
            self.assertIsNotNone(exp.feature_weights)
            self.assertIsNotNone(exp.interpretations)
            self.assertGreater(exp.model_accuracy, 0)

    def test_local_model_fidelity(self):
        """Test local model fidelity scores"""
        data = [{'value': i} for i in range(20)]
        schema = {'value': 'number'}

        explanations = self.explainer.explain(data, schema, None, num_samples=5)

        for exp in explanations:
            self.assertGreaterEqual(exp.model_accuracy, 0)
            self.assertLessEqual(exp.model_accuracy, 1)


class TestMCTSEngine(unittest.TestCase):
    """Test MCTS reasoning engine"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')
        self.engine = MCTSEngine(self.llm, num_simulations=10)

    def test_generate(self):
        """Test MCTS data generation"""
        intent_engine = IntentEngine(self.llm)
        intent = intent_engine.parse_intent("Generate 5 customer records")
        schema = {'name': 'string', 'age': 'number', 'email': 'email'}

        results = self.engine.generate(intent, schema, count=5)

        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsNotNone(result.data)
            self.assertGreater(result.score, 0)
            self.assertIn('MCTS', result.reasoning)

    def test_mcts_metadata(self):
        """Test MCTS metadata"""
        intent_engine = IntentEngine(self.llm)
        intent = intent_engine.parse_intent("Generate 3 records")
        schema = {'field1': 'string'}

        results = self.engine.generate(intent, schema, count=3)

        for result in results:
            self.assertIn('method', result.metadata)
            self.assertEqual(result.metadata['method'], 'mcts')
            self.assertIn('simulations', result.metadata)

    def test_mcts_with_different_simulations(self):
        """Test MCTS with different simulation counts"""
        intent_engine = IntentEngine(self.llm)
        intent = intent_engine.parse_intent("Generate 2 records")
        schema = {'value': 'number'}

        # Test with few simulations
        engine_few = MCTSEngine(self.llm, num_simulations=5)
        results_few = engine_few.generate(intent, schema, count=2)

        # Test with many simulations
        engine_many = MCTSEngine(self.llm, num_simulations=50)
        results_many = engine_many.generate(intent, schema, count=2)

        self.assertEqual(len(results_few), 2)
        self.assertEqual(len(results_many), 2)


class TestHybridEngine(unittest.TestCase):
    """Test hybrid reasoning engine"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')

    def test_generate_default_weights(self):
        """Test hybrid engine with default weights"""
        engine = HybridReasoningEngine(self.llm, adaptive=False)

        intent_engine = IntentEngine(self.llm)
        intent = intent_engine.parse_intent("Generate 20 records")
        schema = {'field1': 'string', 'field2': 'number'}

        results = engine.generate(intent, schema, count=20)

        self.assertEqual(len(results), 20)
        for result in results:
            self.assertIn('hybrid_strategy', result.metadata)

    def test_custom_weights(self):
        """Test hybrid engine with custom weights"""
        custom_weights = {
            'monte_carlo': 0.5,
            'beam_search': 0.3,
            'chain_of_thought': 0.1,
            'tree_of_thoughts': 0.05,
            'mcts': 0.05
        }

        engine = HybridReasoningEngine(
            self.llm,
            strategy_weights=custom_weights,
            adaptive=False
        )

        intent_engine = IntentEngine(self.llm)
        intent = intent_engine.parse_intent("Generate 30 records")
        schema = {'value': 'number'}

        results = engine.generate(intent, schema, count=30)

        # Count strategy usage
        strategies_used = {}
        for result in results:
            strategy = result.metadata['hybrid_strategy']
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

        # Monte Carlo should be used more (has highest weight)
        self.assertGreater(
            strategies_used.get('monte_carlo', 0),
            strategies_used.get('mcts', 0)
        )

    def test_adaptive_weights(self):
        """Test adaptive weight adjustment"""
        engine = HybridReasoningEngine(self.llm, adaptive=True)

        intent_engine = IntentEngine(self.llm)
        intent = intent_engine.parse_intent("Generate 50 records")
        schema = {'value': 'number'}

        # Generate enough data to trigger adaptation
        results = engine.generate(intent, schema, count=50)

        # Check that performance history is being tracked
        self.assertGreater(len(engine.performance_history), 0)

        # Check performance summary
        summary = engine.get_performance_summary()
        self.assertGreater(len(summary), 0)


class TestExplainabilityEngine(unittest.TestCase):
    """Test complete explainability engine"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')
        self.engine = ExplainabilityEngine(self.llm)

    def test_generate_report(self):
        """Test complete explainability report generation"""
        data = [
            {'age': 25, 'salary': 50000, 'city': 'London'},
            {'age': 35, 'salary': 75000, 'city': 'Manchester'},
            {'age': 45, 'salary': 100000, 'city': 'Edinburgh'}
        ]
        schema = {'age': 'number', 'salary': 'number', 'city': 'string'}

        report = self.engine.generate_report(
            data, schema, None, [],
            include_shap=True,
            include_lime=True
        )

        self.assertIsNotNone(report.feature_importances)
        self.assertIsNotNone(report.decision_rules)
        self.assertIsNotNone(report.shap_explanations)
        self.assertIsNotNone(report.lime_explanations)
        self.assertGreater(len(report.summary), 0)

    def test_report_export_json(self):
        """Test JSON export of explainability report"""
        data = [{'value': i} for i in range(10)]
        schema = {'value': 'number'}

        report = self.engine.generate_report(data, schema, None, [])

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            self.engine.export_report_json(report, output_path)
            self.assertTrue(os.path.exists(output_path))

            # Verify JSON is valid
            import json
            with open(output_path, 'r') as f:
                data = json.load(f)
            self.assertIn('metadata', data)
            self.assertIn('feature_importances', data)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_report_export_markdown(self):
        """Test Markdown export of explainability report"""
        data = [{'value': i} for i in range(10)]
        schema = {'value': 'number'}

        report = self.engine.generate_report(data, schema, None, [])

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            self.engine.export_report_markdown(report, output_path)
            self.assertTrue(os.path.exists(output_path))

            # Verify markdown content
            with open(output_path, 'r') as f:
                content = f.read()
            self.assertIn('# Explainability Report', content)
            self.assertIn('Feature Importances', content)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestExplainableGenerator(unittest.TestCase):
    """Test explainable synthetic data generator"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')

    def test_generate_from_prompt(self):
        """Test generation from natural language prompt"""
        generator = ExplainableSyntheticGenerator(
            self.llm,
            reasoning_engine='monte_carlo',
            enable_explainability=True
        )

        result = generator.generate_from_prompt(
            "Generate 10 customer records",
            include_shap=True,
            include_lime=True,
            export_explanation=False
        )

        self.assertEqual(len(result.data), 10)
        self.assertIsNotNone(result.explanation_report)

    def test_generate_with_schema(self):
        """Test generation with explicit schema"""
        generator = ExplainableSyntheticGenerator(
            self.llm,
            reasoning_engine='hybrid',
            enable_explainability=True
        )

        intent = Intent(
            data_type='employee',
            entity='Employee',
            count=15,
            output_format='json',
            purpose='testing',
            domain=None,
            geography='UK'
        )

        schema = {'name': 'string', 'age': 'number', 'salary': 'number'}

        result = generator.generate_with_schema(intent, schema)

        self.assertEqual(len(result.data), 15)
        self.assertIsNotNone(result.explanation_report)

    def test_feature_importance_summary(self):
        """Test feature importance summary generation"""
        generator = ExplainableSyntheticGenerator(
            self.llm,
            reasoning_engine='monte_carlo',
            enable_explainability=True
        )

        result = generator.generate_from_prompt(
            "Generate 10 records",
            export_explanation=False
        )

        summary = generator.get_feature_importance_summary(result, top_k=3)

        self.assertGreater(len(summary), 0)
        self.assertIn('Most Important Features', summary)

    def test_decision_rules_summary(self):
        """Test decision rules summary generation"""
        generator = ExplainableSyntheticGenerator(
            self.llm,
            reasoning_engine='mcts',
            enable_explainability=True
        )

        result = generator.generate_from_prompt(
            "Generate 10 records",
            export_explanation=False
        )

        summary = generator.get_decision_rules_summary(result, top_k=3)

        self.assertGreater(len(summary), 0)
        self.assertIn('Decision Rules', summary)


class TestReasoningEngineFactory(unittest.TestCase):
    """Test reasoning engine factory"""

    def setUp(self):
        self.llm = LLMFactory.create('mock')

    def test_create_all_engines(self):
        """Test creation of all reasoning engines"""
        engines = [
            'monte_carlo',
            'beam_search',
            'chain_of_thought',
            'tree_of_thoughts',
            'mcts',
            'hybrid'
        ]

        for engine_type in engines:
            engine = ReasoningEngineFactory.create(engine_type, self.llm)
            self.assertIsNotNone(engine)

    def test_invalid_engine_type(self):
        """Test error on invalid engine type"""
        with self.assertRaises(ValueError):
            ReasoningEngineFactory.create('invalid_engine', self.llm)


def run_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("EXPLAINABILITY TEST SUITE")
    print("="*80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureImportance))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionTreeExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestSHAPExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestLIMEExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestMCTSEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestExplainabilityEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestExplainableGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestReasoningEngineFactory))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
