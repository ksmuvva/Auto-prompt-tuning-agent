"""
Example: TRUE AI Agent with Adaptive Learning
Demonstrates real dynamic prompt tuning with true metrics
"""

import os
import logging
from agent.true_ai_agent import TrueAIAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_adaptive_tuning():
    """
    Example 1: Adaptive Tuning for Single Requirement

    Demonstrates:
    - TRUE metrics (ground truth comparison)
    - Dynamic prompt generation based on failures
    - Iterative optimization until 98% targets met
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: ADAPTIVE TUNING FOR FW15")
    print("="*70 + "\n")

    # Initialize agent with your LLM provider
    # For OpenAI:
    # agent = TrueAIAgent(
    #     llm_provider='openai',
    #     model='gpt-4',
    #     api_key=os.getenv('OPENAI_API_KEY')
    # )

    # For Anthropic:
    # agent = TrueAIAgent(
    #     llm_provider='anthropic',
    #     model='claude-3-opus-20240229',
    #     api_key=os.getenv('ANTHROPIC_API_KEY')
    # )

    # For demo, using mock (you should use real LLM)
    agent = TrueAIAgent(
        llm_provider='mock',  # Change to 'openai', 'anthropic', etc.
        data_dir='data',
        output_dir='results',
        max_tuning_iterations=10
    )

    # Load data
    print("Loading data...")
    load_result = agent.load_data()

    if not load_result.get('success'):
        print("Failed to load data!")
        return

    print(f"✓ Loaded {load_result['transactions']} transactions\n")

    # Run adaptive tuning for FW15
    print("Starting adaptive tuning for FW15 (high-value transactions)...")
    print("This will:")
    print("  1. Test initial prompt")
    print("  2. Calculate TRUE metrics (ground truth comparison)")
    print("  3. Use LLM to improve prompt based on failures")
    print("  4. Test improved prompt")
    print("  5. Repeat until 98% precision & accuracy achieved\n")

    result = agent.analyze_with_dynamic_tuning(
        requirement='fw15',
        requirement_description='Identify high-value transactions over £250',
        target_precision=0.98,
        target_accuracy=0.98,
        test_formats=False  # Set True to test multiple output formats
    )

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    if result.get('target_achieved'):
        print("🎯 TARGET ACHIEVED!")
    else:
        print("⚠ Target not achieved (may need more iterations)")

    metrics = result.get('best_metrics', {})
    print(f"\nBest Metrics:")
    print(f"  Precision: {metrics.get('precision', 0):.2%}")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
    print(f"  Recall:    {metrics.get('recall', 0):.2%}")
    print(f"  F1 Score:  {metrics.get('f1_score', 0):.2%}")
    print(f"\nIterations: {result.get('iterations', 0)}")

    print(f"\nBest Prompt:")
    print("-" * 70)
    print(result.get('best_prompt', 'N/A')[:500] + "...")
    print("-" * 70)


def example_dynamic_vs_template():
    """
    Example 2: Compare Dynamic Tuning vs Template-Based

    Demonstrates why dynamic tuning is superior
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: DYNAMIC vs TEMPLATE COMPARISON")
    print("="*70 + "\n")

    agent = TrueAIAgent(
        llm_provider='mock',
        data_dir='data',
        output_dir='results',
        max_tuning_iterations=5
    )

    agent.load_data()

    print("Comparing dynamic tuning vs predefined templates...")
    print("Testing:")
    print("  - Dynamic: AI-generated prompts optimized iteratively")
    print("  - Templates: Predefined prompt templates\n")

    result = agent.compare_dynamic_vs_template(
        requirement='fw15',
        requirement_description='Identify high-value transactions over £250',
        template_names=['direct_concise', 'role_based_expert', 'chain_of_thought']
    )

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\n🏆 Winner: {result['winner'].upper()}")
    print(f"\nDynamic Tuning Score: {result['dynamic_score']:.2%}")
    print(f"Best Template Score:  {result['best_template_score']:.2%} ({result['best_template']})")

    print(f"\nImprovement: {(result['dynamic_score'] - result['best_template_score']):.2%}")


def example_all_requirements():
    """
    Example 3: Analyze All FW Requirements

    Demonstrates complete workflow for all 8 requirements
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: ANALYZE ALL FW REQUIREMENTS")
    print("="*70 + "\n")

    agent = TrueAIAgent(
        llm_provider='mock',
        data_dir='data',
        output_dir='results',
        max_tuning_iterations=5
    )

    agent.load_data()

    print("Analyzing all 8 FW requirements with adaptive tuning...")
    print("This may take several minutes...\n")

    results = agent.analyze_all_requirements(
        use_dynamic=True,
        test_formats=False
    )

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    summary = results['summary']
    print(f"\nRequirements meeting 98% targets: {summary['targets_met']}/{results['total_requirements']}")
    print(f"Average Precision: {summary['average_precision']:.2%}")
    print(f"Average Accuracy:  {summary['average_accuracy']:.2%}")

    print("\nIndividual Results:")
    for req_code, req_result in results['results'].items():
        metrics = req_result.get('best_metrics', {})
        status = "✓" if metrics.get('meets_98_percent_target') else "✗"
        print(f"  {status} {req_code}: P={metrics.get('precision', 0):.2%}, A={metrics.get('accuracy', 0):.2%}")


def example_nlp_cli():
    """
    Example 4: Natural Language CLI

    Demonstrates NLP interface
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: NATURAL LANGUAGE CLI")
    print("="*70 + "\n")

    print("The NLP CLI allows natural language interaction:")
    print("\nExample commands:")
    print("  'use openai'")
    print("  'analyze fw15'")
    print("  'show me the metrics'")
    print("  'tune the prompts'")
    print("  'why is accuracy low?'")

    print("\nTo launch NLP CLI:")
    print("  python -m agent.nlp_cli")


def example_metrics_calculation():
    """
    Example 5: True Metrics Calculation

    Demonstrates precise mathematical metrics
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: TRUE METRICS CALCULATION")
    print("="*70 + "\n")

    from agent.true_metrics import TrueMetricsCalculator

    calculator = TrueMetricsCalculator()

    # Example LLM response
    llm_response = """
    High-value transactions found:
    - Transaction TXN_001: £500
    - Transaction TXN_003: £750
    - Transaction TXN_005: £1000
    """

    # Example ground truth
    ground_truth = {
        'high_value_transactions': ['TXN_001', 'TXN_003', 'TXN_004', 'TXN_005']
    }

    # Calculate TRUE metrics
    metrics = calculator.calculate_metrics(
        llm_response=llm_response,
        ground_truth=ground_truth,
        total_transactions=100
    )

    print("LLM Response IDs: TXN_001, TXN_003, TXN_005")
    print("Ground Truth IDs: TXN_001, TXN_003, TXN_004, TXN_005")
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['confusion_matrix']['true_positives']} (correct predictions)")
    print(f"  False Positives: {metrics['confusion_matrix']['false_positives']} (wrong predictions)")
    print(f"  False Negatives: {metrics['confusion_matrix']['false_negatives']} (missed)")
    print(f"  True Negatives:  {metrics['confusion_matrix']['true_negatives']}")

    print("\nCalculated Metrics:")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  F1 Score:  {metrics['f1_score']:.2%}")

    print(f"\nMeets 98% target? {metrics['meets_98_percent_target']}")


if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  TRUE AI AGENT EXAMPLES                                       ║
║  Adaptive Learning with Real Metrics                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")

    print("\nWhich example would you like to run?")
    print("1. Adaptive Tuning for Single Requirement")
    print("2. Dynamic vs Template Comparison")
    print("3. Analyze All FW Requirements")
    print("4. Natural Language CLI Demo")
    print("5. True Metrics Calculation")
    print("6. Run All Examples")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '1':
        example_adaptive_tuning()
    elif choice == '2':
        example_dynamic_vs_template()
    elif choice == '3':
        example_all_requirements()
    elif choice == '4':
        example_nlp_cli()
    elif choice == '5':
        example_metrics_calculation()
    elif choice == '6':
        example_metrics_calculation()
        example_adaptive_tuning()
        example_dynamic_vs_template()
        example_all_requirements()
        example_nlp_cli()
    else:
        print("Invalid choice")

    print("\n✓ Examples complete!")
