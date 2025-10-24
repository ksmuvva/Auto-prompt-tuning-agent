"""
Hybrid Reasoning Engine Demonstration

Shows how the hybrid reasoning engine combines multiple strategies
for optimal synthetic data generation
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_providers import LLMFactory
from core.intent_engine import IntentEngine, Intent
from core.reasoning_engines import ReasoningEngineFactory, HybridReasoningEngine


def demo_hybrid_basic():
    """Demo 1: Basic hybrid reasoning"""
    print("\n" + "="*80)
    print("DEMO 1: Basic Hybrid Reasoning")
    print("="*80)

    llm = LLMFactory.create('mock')

    # Create hybrid engine with default weights
    hybrid = ReasoningEngineFactory.create('hybrid', llm, adaptive=True)

    # Parse intent
    intent_engine = IntentEngine(llm)
    intent = intent_engine.parse_intent("Generate 50 customer records")
    schema = intent_engine.get_schema_suggestion(intent)

    print(f"\n📊 Generating {intent.count} records using hybrid reasoning...")
    print(f"Schema: {list(schema.keys())}")

    # Generate data
    results = hybrid.generate(intent, schema, count=50)

    # Show strategy distribution
    strategies_used = {}
    for result in results:
        strategy = result.metadata.get('hybrid_strategy', 'unknown')
        strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

    print("\n📈 Strategy Distribution:")
    for strategy, count in sorted(strategies_used.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {strategy:20} {bar} {count:3d} ({percentage:5.1f}%)")

    # Show performance summary
    if isinstance(hybrid, HybridReasoningEngine):
        perf_summary = hybrid.get_performance_summary()
        print("\n🎯 Performance Summary:")
        for strategy, metrics in perf_summary.items():
            print(f"  {strategy:20} Avg Score: {metrics['avg_score']:.3f}  "
                  f"Uses: {metrics['num_uses']:3d}  Weight: {metrics['current_weight']:.3f}")

    print("\n✅ Demo 1 complete!")


def demo_hybrid_custom_weights():
    """Demo 2: Hybrid reasoning with custom weights"""
    print("\n" + "="*80)
    print("DEMO 2: Hybrid Reasoning with Custom Weights")
    print("="*80)

    llm = LLMFactory.create('mock')

    # Create hybrid engine with custom strategy weights
    custom_weights = {
        'monte_carlo': 0.1,      # Low weight for Monte Carlo
        'beam_search': 0.1,      # Low weight for Beam Search
        'chain_of_thought': 0.2, # Medium weight for CoT
        'tree_of_thoughts': 0.2, # Medium weight for ToT
        'mcts': 0.4              # High weight for MCTS
    }

    hybrid = ReasoningEngineFactory.create(
        'hybrid',
        llm,
        strategy_weights=custom_weights,
        adaptive=False  # Disable adaptation to use fixed weights
    )

    # Parse intent
    intent_engine = IntentEngine(llm)
    intent = intent_engine.parse_intent("Generate 100 employee records for HR analysis")
    schema = intent_engine.get_schema_suggestion(intent)

    print(f"\n📊 Generating {intent.count} records with custom weights...")
    print("\nCustom Strategy Weights:")
    for strategy, weight in custom_weights.items():
        bar = "█" * int(weight * 50)
        print(f"  {strategy:20} {bar} {weight:.2f}")

    # Generate data
    results = hybrid.generate(intent, schema, count=100)

    # Show actual strategy distribution
    strategies_used = {}
    for result in results:
        strategy = result.metadata.get('hybrid_strategy', 'unknown')
        strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

    print("\n📈 Actual Strategy Distribution:")
    for strategy, count in sorted(strategies_used.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {strategy:20} {bar} {count:3d} ({percentage:5.1f}%)")

    print("\n✅ Demo 2 complete!")


def demo_hybrid_adaptive():
    """Demo 3: Adaptive hybrid reasoning"""
    print("\n" + "="*80)
    print("DEMO 3: Adaptive Hybrid Reasoning")
    print("="*80)

    llm = LLMFactory.create('mock')

    # Create adaptive hybrid engine
    hybrid = ReasoningEngineFactory.create('hybrid', llm, adaptive=True)

    # Parse intent
    intent_engine = IntentEngine(llm)
    intent = intent_engine.parse_intent("Generate 200 transaction records")
    schema = intent_engine.get_schema_suggestion(intent)

    print(f"\n📊 Generating {intent.count} records with adaptive strategy selection...")
    print("The engine will learn which strategies work best and adapt weights accordingly.\n")

    # Generate data in batches to show adaptation
    batch_size = 50
    num_batches = 4

    for batch_num in range(num_batches):
        print(f"\n--- Batch {batch_num + 1}/{num_batches} ---")

        batch_results = hybrid.generate(intent, schema, count=batch_size)

        # Show strategy distribution for this batch
        strategies_used = {}
        for result in batch_results:
            strategy = result.metadata.get('hybrid_strategy', 'unknown')
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

        print("Strategy usage in this batch:")
        for strategy, count in sorted(strategies_used.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy:20} {count:3d}")

        # Show current weights (after adaptation)
        if isinstance(hybrid, HybridReasoningEngine) and batch_num < num_batches - 1:
            print("\nCurrent strategy weights (will adapt for next batch):")
            for strategy, weight in hybrid.strategy_weights.items():
                bar = "█" * int(weight * 50)
                print(f"  {strategy:20} {bar} {weight:.3f}")

    # Final performance summary
    if isinstance(hybrid, HybridReasoningEngine):
        print("\n" + "-"*80)
        print("FINAL PERFORMANCE SUMMARY")
        print("-"*80)
        perf_summary = hybrid.get_performance_summary()
        for strategy, metrics in sorted(
            perf_summary.items(),
            key=lambda x: x[1]['avg_score'],
            reverse=True
        ):
            print(f"\n{strategy}:")
            print(f"  Average Score:  {metrics['avg_score']:.3f}")
            print(f"  Times Used:     {metrics['num_uses']}")
            print(f"  Final Weight:   {metrics['current_weight']:.3f}")

    print("\n✅ Demo 3 complete!")


def demo_compare_all_engines():
    """Demo 4: Compare all reasoning engines"""
    print("\n" + "="*80)
    print("DEMO 4: Comprehensive Reasoning Engine Comparison")
    print("="*80)

    llm = LLMFactory.create('mock')

    # Parse intent
    intent_engine = IntentEngine(llm)
    intent = intent_engine.parse_intent("Generate 30 product records")
    schema = intent_engine.get_schema_suggestion(intent)

    # All available engines
    engines = [
        'monte_carlo',
        'beam_search',
        'chain_of_thought',
        'tree_of_thoughts',
        'mcts',
        'hybrid'
    ]

    print(f"\n📊 Comparing {len(engines)} reasoning engines...")
    print(f"Generating {intent.count} records with each engine\n")

    results_comparison = {}

    for engine_type in engines:
        print(f"Testing {engine_type}...")

        try:
            # Create engine
            if engine_type == 'hybrid':
                engine = ReasoningEngineFactory.create(engine_type, llm, adaptive=False)
            else:
                engine = ReasoningEngineFactory.create(engine_type, llm)

            # Generate data
            results = engine.generate(intent, schema, count=30)

            # Calculate metrics
            avg_score = sum(r.score for r in results) / len(results)
            sample_reasoning = results[0].reasoning if results else "N/A"

            results_comparison[engine_type] = {
                'success': True,
                'avg_score': avg_score,
                'num_records': len(results),
                'sample_reasoning': sample_reasoning
            }

        except Exception as e:
            results_comparison[engine_type] = {
                'success': False,
                'error': str(e)
            }

    # Display comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    # Sort by average score
    sorted_results = sorted(
        [(k, v) for k, v in results_comparison.items() if v.get('success', False)],
        key=lambda x: x[1].get('avg_score', 0),
        reverse=True
    )

    for i, (engine_type, metrics) in enumerate(sorted_results, 1):
        print(f"\n{i}. {engine_type.upper()}")
        print(f"   Average Score: {metrics['avg_score']:.3f}")
        print(f"   Records:       {metrics['num_records']}")
        print(f"   Reasoning:     {metrics['sample_reasoning'][:60]}...")

    # Show failures
    failures = [(k, v) for k, v in results_comparison.items() if not v.get('success', False)]
    if failures:
        print("\n" + "-"*80)
        print("FAILURES:")
        for engine_type, metrics in failures:
            print(f"  ❌ {engine_type}: {metrics['error']}")

    print("\n✅ Demo 4 complete!")


def demo_all():
    """Run all hybrid reasoning demonstrations"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "HYBRID REASONING ENGINE DEMONSTRATIONS" + " "*20 + "║")
    print("╚" + "="*78 + "╝")

    try:
        demo_hybrid_basic()
        demo_hybrid_custom_weights()
        demo_hybrid_adaptive()
        demo_compare_all_engines()

        print("\n" + "="*80)
        print("ALL HYBRID REASONING DEMONSTRATIONS COMPLETED")
        print("="*80)
        print("\n✅ Key Takeaways:")
        print("  • Hybrid engine combines multiple reasoning strategies")
        print("  • Can use fixed or adaptive strategy weights")
        print("  • Learns which strategies work best over time")
        print("  • Provides optimal balance of quality and diversity")

    except Exception as e:
        print(f"\n❌ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    demo_all()
