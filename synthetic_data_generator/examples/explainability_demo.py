"""
Comprehensive Explainability Demonstration

Demonstrates all explainability features of the synthetic data generator
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_providers import LLMFactory
from core.explainable_generator import ExplainableSyntheticGenerator, ExplainabilityDashboard


def demo_basic_explainability():
    """Demo 1: Basic explainability with Monte Carlo"""
    print("\n" + "="*100)
    print("DEMO 1: Basic Explainability - Monte Carlo Reasoning")
    print("="*100)

    # Initialize with mock LLM
    llm = LLMFactory.create('mock')

    # Create explainable generator with Monte Carlo
    generator = ExplainableSyntheticGenerator(
        llm_provider=llm,
        reasoning_engine='monte_carlo',
        enable_explainability=True
    )

    # Generate data
    result = generator.generate_from_prompt(
        prompt="Generate 100 UK customer records for e-commerce",
        include_shap=True,
        include_lime=True,
        export_explanation=True,
        output_dir='./output/demo1'
    )

    # Display report
    ExplainabilityDashboard.print_report(result)

    # Export data
    generator.export_data(result, './output/demo1/customers.csv', format='csv')

    print("\n✅ Demo 1 complete! Check ./output/demo1/ for results")


def demo_mcts_explainability():
    """Demo 2: MCTS reasoning with explainability"""
    print("\n" + "="*100)
    print("DEMO 2: MCTS Reasoning with Explainability")
    print("="*100)

    llm = LLMFactory.create('mock')

    # Create generator with MCTS
    generator = ExplainableSyntheticGenerator(
        llm_provider=llm,
        reasoning_engine='mcts',
        enable_explainability=True,
        num_simulations=50  # MCTS parameter
    )

    # Generate patient data
    result = generator.generate_from_prompt(
        prompt="Generate 50 patient records for healthcare ML training",
        include_shap=True,
        include_lime=True,
        export_explanation=True,
        output_dir='./output/demo2'
    )

    # Show feature importance
    print("\n" + "-"*100)
    print("FEATURE IMPORTANCE SUMMARY")
    print("-"*100)
    print(generator.get_feature_importance_summary(result, top_k=5))

    # Show decision rules
    print("\n" + "-"*100)
    print("DECISION RULES SUMMARY")
    print("-"*100)
    print(generator.get_decision_rules_summary(result, top_k=5))

    print("\n✅ Demo 2 complete! Check ./output/demo2/ for results")


def demo_hybrid_explainability():
    """Demo 3: Hybrid reasoning with adaptive explainability"""
    print("\n" + "="*100)
    print("DEMO 3: Hybrid Reasoning with Adaptive Explainability")
    print("="*100)

    llm = LLMFactory.create('mock')

    # Create generator with hybrid reasoning
    generator = ExplainableSyntheticGenerator(
        llm_provider=llm,
        reasoning_engine='hybrid',
        enable_explainability=True,
        adaptive=True  # Adaptive strategy selection
    )

    # Generate financial data
    result = generator.generate_from_prompt(
        prompt="Generate 200 banking transaction records for fraud detection training",
        include_shap=True,
        include_lime=True,
        export_explanation=True,
        output_dir='./output/demo3'
    )

    # Display comprehensive report
    ExplainabilityDashboard.print_report(result)

    # Export in multiple formats
    generator.export_data(result, './output/demo3/transactions.csv', format='csv')
    generator.export_data(result, './output/demo3/transactions.json', format='json')

    print("\n✅ Demo 3 complete! Check ./output/demo3/ for results")


def demo_compare_reasoning_engines():
    """Demo 4: Compare reasoning engines with explainability"""
    print("\n" + "="*100)
    print("DEMO 4: Reasoning Engine Comparison with Explainability")
    print("="*100)

    llm = LLMFactory.create('mock')

    generator = ExplainableSyntheticGenerator(
        llm_provider=llm,
        reasoning_engine='hybrid',
        enable_explainability=True
    )

    # Compare engines
    comparison = generator.compare_reasoning_engines(
        prompt="Generate employee records",
        engines=['monte_carlo', 'beam_search', 'chain_of_thought', 'mcts', 'hybrid'],
        num_samples=50
    )

    # Display comparison
    print("\n" + "-"*100)
    print("REASONING ENGINE COMPARISON")
    print("-"*100)

    for engine, results in comparison.items():
        print(f"\n{engine.upper()}:")
        if results['success']:
            print(f"  ✅ Success")
            print(f"  Records: {results['num_records']}")
            print(f"  Avg Score: {results['avg_score']:.2f}")
            print(f"  Reasoning: {results['sample_reasoning'][:80]}...")
        else:
            print(f"  ❌ Failed: {results['error']}")

    print("\n✅ Demo 4 complete!")


def demo_shap_lime_deep_dive():
    """Demo 5: Deep dive into SHAP and LIME explanations"""
    print("\n" + "="*100)
    print("DEMO 5: SHAP and LIME Deep Dive")
    print("="*100)

    llm = LLMFactory.create('mock')

    generator = ExplainableSyntheticGenerator(
        llm_provider=llm,
        reasoning_engine='hybrid',
        enable_explainability=True
    )

    # Generate data
    result = generator.generate_from_prompt(
        prompt="Generate 100 insurance claim records",
        include_shap=True,
        include_lime=True,
        export_explanation=False,
        output_dir='./output/demo5'
    )

    report = result.explanation_report

    # Detailed SHAP analysis
    print("\n" + "-"*100)
    print("DETAILED SHAP ANALYSIS")
    print("-"*100)

    if report.shap_explanations:
        for i, shap in enumerate(report.shap_explanations, 1):
            print(f"\nRecord {i} ({shap.record_id}):")
            print(f"  Base Value: {shap.base_value:.2f}")
            print(f"  Total Prediction: {shap.total_prediction:.2f}")
            print(f"\n  Feature Contributions:")

            for feature, value in sorted(
                shap.shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ):
                contribution = shap.feature_contributions.get(feature, '')
                print(f"    {feature:20} SHAP: {value:+8.2f}  → {contribution}")

    # Detailed LIME analysis
    print("\n" + "-"*100)
    print("DETAILED LIME ANALYSIS")
    print("-"*100)

    if report.lime_explanations:
        for i, lime in enumerate(report.lime_explanations, 1):
            print(f"\nRecord {i} ({lime.record_id}):")
            print(f"  Local Model: {lime.local_model}")
            print(f"  Fidelity: {lime.model_accuracy:.2f}")
            print(f"\n  Feature Weights:")

            for feature, weight in sorted(
                lime.feature_weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ):
                print(f"    {feature:20} Weight: {weight:+6.2f}")

            print(f"\n  Interpretations:")
            for interp in lime.interpretations:
                print(f"    • {interp}")

    print("\n✅ Demo 5 complete!")


def demo_all():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "="*98 + "╗")
    print("║" + " "*25 + "EXPLAINABILITY DEMONSTRATION SUITE" + " "*38 + "║")
    print("║" + " "*98 + "║")
    print("║" + "  Comprehensive demonstrations of explainability features:             " + " "*27 + "║")
    print("║" + "  • Feature Importance                                                  " + " "*27 + "║")
    print("║" + "  • Decision Rules                                                      " + " "*27 + "║")
    print("║" + "  • SHAP Explanations                                                   " + " "*27 + "║")
    print("║" + "  • LIME Explanations                                                   " + " "*27 + "║")
    print("║" + "  • Reasoning Engine Comparison                                         " + " "*27 + "║")
    print("╚" + "="*98 + "╝")

    try:
        # Run all demos
        demo_basic_explainability()
        demo_mcts_explainability()
        demo_hybrid_explainability()
        demo_compare_reasoning_engines()
        demo_shap_lime_deep_dive()

        # Final summary
        print("\n" + "="*100)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*100)
        print("\n📁 Output files generated in:")
        print("  • ./output/demo1/ - Basic explainability with Monte Carlo")
        print("  • ./output/demo2/ - MCTS reasoning")
        print("  • ./output/demo3/ - Hybrid reasoning")
        print("  • ./output/demo5/ - SHAP/LIME deep dive")
        print("\n📄 Each directory contains:")
        print("  • Generated data (CSV/JSON)")
        print("  • Explainability report (JSON)")
        print("  • Explainability report (Markdown)")
        print("\n✅ Review the reports to understand how the data was generated!")

    except Exception as e:
        print(f"\n❌ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    demo_all()
