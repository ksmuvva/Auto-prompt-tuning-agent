"""
Explainable Synthetic Data Generator

Integrates explainability into the data generation process
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .llm_providers import LLMProvider
from .intent_engine import IntentEngine, Intent
from .reasoning_engines import ReasoningEngineFactory, GenerationResult
from .explainability import (
    ExplainabilityEngine,
    ExplanationReport,
    FeatureImportance,
    DecisionRule
)
from .output_engine import OutputEngine


@dataclass
class ExplainableGenerationResult:
    """Result from explainable data generation"""
    data: List[Dict[str, Any]]
    generation_metadata: Dict[str, Any]
    explanation_report: ExplanationReport
    reasoning_engine: str
    intent: Intent


class ExplainableSyntheticGenerator:
    """
    Explainable Synthetic Data Generator

    Generates synthetic data with comprehensive explainability
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        reasoning_engine: str = 'hybrid',
        enable_explainability: bool = True,
        **engine_kwargs
    ):
        """
        Initialize explainable generator

        Args:
            llm_provider: LLM provider instance
            reasoning_engine: Type of reasoning engine to use
            enable_explainability: Whether to generate explanations
            **engine_kwargs: Additional arguments for reasoning engine
        """
        self.llm = llm_provider
        self.intent_engine = IntentEngine(llm_provider)
        self.reasoning_engine_type = reasoning_engine
        self.engine_kwargs = engine_kwargs
        self.enable_explainability = enable_explainability

        # Initialize explainability engine
        if enable_explainability:
            self.explainability_engine = ExplainabilityEngine(llm_provider)
        else:
            self.explainability_engine = None

        # Initialize output engine
        self.output_engine = OutputEngine()

    def generate_from_prompt(
        self,
        prompt: str,
        include_shap: bool = True,
        include_lime: bool = True,
        export_explanation: bool = True,
        output_dir: str = './output'
    ) -> ExplainableGenerationResult:
        """
        Generate synthetic data from natural language prompt

        Args:
            prompt: Natural language description of data to generate
            include_shap: Include SHAP explanations
            include_lime: Include LIME explanations
            export_explanation: Export explanation report
            output_dir: Directory for output files

        Returns:
            ExplainableGenerationResult with data and explanations
        """

        # Step 1: Parse intent
        print("🧠 Parsing intent from prompt...")
        intent = self.intent_engine.parse_intent(prompt)

        # Step 2: Get schema
        print("📐 Generating schema...")
        schema = self.intent_engine.get_schema_suggestion(intent)

        # Step 3: Generate data
        print(f"⚙️  Generating data using {self.reasoning_engine_type} reasoning...")
        engine = ReasoningEngineFactory.create(
            self.reasoning_engine_type,
            self.llm,
            **self.engine_kwargs
        )

        results = engine.generate(intent, schema, count=intent.count)
        generated_data = [r.data for r in results]
        reasoning_metadata = [r.metadata for r in results]

        # Step 4: Generate explanations
        explanation_report = None
        if self.enable_explainability and self.explainability_engine:
            print("🔍 Generating explainability report...")
            explanation_report = self.explainability_engine.generate_report(
                generated_data=generated_data,
                schema=schema,
                intent=intent,
                reasoning_metadata=reasoning_metadata,
                include_shap=include_shap,
                include_lime=include_lime
            )

            # Export explanation if requested
            if export_explanation:
                self._export_explanations(explanation_report, output_dir)

        # Step 5: Create result
        result = ExplainableGenerationResult(
            data=generated_data,
            generation_metadata={
                'intent': intent.__dict__,
                'schema': schema,
                'reasoning_engine': self.reasoning_engine_type,
                'num_records': len(generated_data)
            },
            explanation_report=explanation_report,
            reasoning_engine=self.reasoning_engine_type,
            intent=intent
        )

        print(f"✅ Successfully generated {len(generated_data)} records with explainability!")

        return result

    def generate_with_schema(
        self,
        intent: Intent,
        schema: Dict[str, str],
        include_shap: bool = True,
        include_lime: bool = True
    ) -> ExplainableGenerationResult:
        """
        Generate data with explicit schema

        Args:
            intent: Generation intent
            schema: Data schema
            include_shap: Include SHAP explanations
            include_lime: Include LIME explanations

        Returns:
            ExplainableGenerationResult
        """

        # Generate data
        engine = ReasoningEngineFactory.create(
            self.reasoning_engine_type,
            self.llm,
            **self.engine_kwargs
        )

        results = engine.generate(intent, schema, count=intent.count)
        generated_data = [r.data for r in results]
        reasoning_metadata = [r.metadata for r in results]

        # Generate explanations
        explanation_report = None
        if self.enable_explainability and self.explainability_engine:
            explanation_report = self.explainability_engine.generate_report(
                generated_data=generated_data,
                schema=schema,
                intent=intent,
                reasoning_metadata=reasoning_metadata,
                include_shap=include_shap,
                include_lime=include_lime
            )

        return ExplainableGenerationResult(
            data=generated_data,
            generation_metadata={
                'intent': intent.__dict__,
                'schema': schema,
                'reasoning_engine': self.reasoning_engine_type,
                'num_records': len(generated_data)
            },
            explanation_report=explanation_report,
            reasoning_engine=self.reasoning_engine_type,
            intent=intent
        )

    def _export_explanations(self, report: ExplanationReport, output_dir: str):
        """Export explanation reports"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Export JSON
        json_path = os.path.join(output_dir, 'explainability_report.json')
        self.explainability_engine.export_report_json(report, json_path)
        print(f"  📄 Exported JSON report: {json_path}")

        # Export Markdown
        md_path = os.path.join(output_dir, 'explainability_report.md')
        self.explainability_engine.export_report_markdown(report, md_path)
        print(f"  📝 Exported Markdown report: {md_path}")

    def export_data(
        self,
        result: ExplainableGenerationResult,
        output_path: str,
        format: str = 'csv'
    ):
        """
        Export generated data

        Args:
            result: Generation result
            output_path: Output file path
            format: Output format (csv, json, excel, etc.)
        """
        self.output_engine.export(result.data, output_path, format)
        print(f"💾 Exported data to: {output_path}")

    def get_feature_importance_summary(
        self,
        result: ExplainableGenerationResult,
        top_k: int = 5
    ) -> str:
        """
        Get summary of top feature importances

        Args:
            result: Generation result
            top_k: Number of top features to include

        Returns:
            Human-readable summary
        """
        if not result.explanation_report:
            return "Explainability not enabled"

        importances = result.explanation_report.feature_importances[:top_k]

        summary = f"Top {top_k} Most Important Features:\n\n"
        for i, feat in enumerate(importances, 1):
            summary += f"{i}. {feat.feature_name}\n"
            summary += f"   Importance: {feat.importance_score:.2f}\n"
            summary += f"   Type: {feat.contribution_type}\n"
            summary += f"   {feat.explanation}\n\n"

        return summary

    def get_decision_rules_summary(
        self,
        result: ExplainableGenerationResult,
        top_k: int = 5
    ) -> str:
        """
        Get summary of decision rules

        Args:
            result: Generation result
            top_k: Number of rules to include

        Returns:
            Human-readable summary
        """
        if not result.explanation_report:
            return "Explainability not enabled"

        rules = result.explanation_report.decision_rules[:top_k]

        summary = f"Top {top_k} Decision Rules:\n\n"
        for i, rule in enumerate(rules, 1):
            summary += f"{i}. IF {rule.condition}\n"
            summary += f"   THEN {rule.action}\n"
            summary += f"   Confidence: {rule.confidence:.2f}\n"
            summary += f"   Support: {rule.support} records\n\n"

        return summary

    def compare_reasoning_engines(
        self,
        prompt: str,
        engines: List[str] = None,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Compare different reasoning engines

        Args:
            prompt: Generation prompt
            engines: List of engines to compare
            num_samples: Number of samples to generate

        Returns:
            Comparison results
        """
        if engines is None:
            engines = ['monte_carlo', 'beam_search', 'chain_of_thought', 'mcts', 'hybrid']

        # Parse intent
        intent = self.intent_engine.parse_intent(prompt)
        intent.count = num_samples
        schema = self.intent_engine.get_schema_suggestion(intent)

        comparison = {}

        for engine_type in engines:
            print(f"\n🔄 Testing {engine_type}...")

            try:
                # Create engine
                engine = ReasoningEngineFactory.create(engine_type, self.llm)

                # Generate data
                results = engine.generate(intent, schema, count=num_samples)
                generated_data = [r.data for r in results]

                # Calculate metrics
                comparison[engine_type] = {
                    'success': True,
                    'num_records': len(generated_data),
                    'avg_score': sum(r.score for r in results) / len(results) if results else 0,
                    'sample_reasoning': results[0].reasoning if results else None
                }

            except Exception as e:
                comparison[engine_type] = {
                    'success': False,
                    'error': str(e)
                }

        return comparison


class ExplainabilityDashboard:
    """
    Dashboard for visualizing explainability results
    """

    @staticmethod
    def print_report(result: ExplainableGenerationResult):
        """Print comprehensive explainability report"""

        print("\n" + "="*80)
        print("EXPLAINABILITY REPORT")
        print("="*80)

        if not result.explanation_report:
            print("Explainability not enabled for this generation")
            return

        report = result.explanation_report

        # Summary
        print("\n" + report.summary)

        # Feature Importances
        print("\n" + "-"*80)
        print("FEATURE IMPORTANCES")
        print("-"*80)

        for feat in report.feature_importances[:10]:
            bar_length = int(feat.importance_score * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"\n{feat.feature_name:20} [{bar}] {feat.importance_score:.2f}")
            print(f"  Type: {feat.contribution_type}")
            print(f"  {feat.explanation}")

        # Decision Rules
        print("\n" + "-"*80)
        print("DECISION RULES")
        print("-"*80)

        for i, rule in enumerate(report.decision_rules[:10], 1):
            print(f"\n{i}. IF {rule.condition}")
            print(f"   THEN {rule.action}")
            print(f"   Confidence: {rule.confidence:.2f} | Support: {rule.support}")

        # SHAP Summary
        if report.shap_explanations:
            print("\n" + "-"*80)
            print(f"SHAP EXPLANATIONS (Sample of {len(report.shap_explanations)})")
            print("-"*80)

            for shap in report.shap_explanations[:3]:
                print(f"\n{shap.record_id}:")
                for feature, value in sorted(
                    shap.shap_values.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]:
                    print(f"  {feature:20} {value:+8.2f}  {shap.feature_contributions.get(feature, '')}")

        # LIME Summary
        if report.lime_explanations:
            print("\n" + "-"*80)
            print(f"LIME EXPLANATIONS (Sample of {len(report.lime_explanations)})")
            print("-"*80)

            for lime in report.lime_explanations[:3]:
                print(f"\n{lime.record_id} (accuracy: {lime.model_accuracy:.2f}):")
                for interp in lime.interpretations[:3]:
                    print(f"  • {interp}")

        print("\n" + "="*80)
