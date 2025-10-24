#!/usr/bin/env python3
"""
COMPREHENSIVE TESTING SUITE
Tests TRUE AI Agent with Real OpenAI API
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  COMPREHENSIVE TRUE AI AGENT TEST SUITE                      ║
║  Testing with Real OpenAI API                                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")

# Test results storage
test_results = {
    'timestamp': datetime.now().isoformat(),
    'tests': [],
    'summary': {
        'total': 0,
        'passed': 0,
        'failed': 0
    }
}

def log_test(name, status, details=""):
    """Log test result"""
    test_results['tests'].append({
        'name': name,
        'status': status,
        'details': details,
        'timestamp': datetime.now().isoformat()
    })
    test_results['summary']['total'] += 1
    if status == 'PASS':
        test_results['summary']['passed'] += 1
        print(f"✓ {name}: PASS")
    else:
        test_results['summary']['failed'] += 1
        print(f"✗ {name}: FAIL - {details}")
    if details and status == 'PASS':
        print(f"  {details}")


print("\n" + "="*70)
print("TEST 1: DATA LOADING")
print("="*70)

try:
    from agent.data_processor import TransactionDataProcessor

    processor = TransactionDataProcessor(data_dir='data')
    data_result = processor.process_all()

    if data_result and 'full_data' in data_result:
        count = len(data_result['full_data'])
        log_test("Data Loading", "PASS", f"Loaded {count} transactions")
    else:
        log_test("Data Loading", "FAIL", "No data loaded")

except Exception as e:
    log_test("Data Loading", "FAIL", str(e))


print("\n" + "="*70)
print("TEST 2: GROUND TRUTH LOADING")
print("="*70)

try:
    from agent.ground_truth import GroundTruthManager

    gt_manager = GroundTruthManager()
    loaded = gt_manager.load_master_file()

    if loaded:
        # Get FW15 ground truth
        fw15_gt = gt_manager.get_requirement_ground_truth('fw15')
        if fw15_gt and 'high_value_transactions' in fw15_gt:
            count = len(fw15_gt['high_value_transactions'])
            log_test("Ground Truth Loading", "PASS",
                    f"Loaded ground truth for FW15: {count} high-value transactions")
        else:
            log_test("Ground Truth Loading", "FAIL", "FW15 ground truth not found")
    else:
        log_test("Ground Truth Loading", "FAIL", "Failed to load ground truth file")

except Exception as e:
    log_test("Ground Truth Loading", "FAIL", str(e))


print("\n" + "="*70)
print("TEST 3: TRUE METRICS CALCULATOR")
print("="*70)

try:
    from agent.true_metrics import TrueMetricsCalculator

    calculator = TrueMetricsCalculator()

    # Test with sample data
    llm_response = """
    High-value transactions found:
    - Transaction TXN_001: £500.00
    - Transaction TXN_003: £750.00
    - Transaction TXN_005: £1000.00
    """

    ground_truth = {
        'high_value_transactions': ['TXN_001', 'TXN_003', 'TXN_004', 'TXN_005']
    }

    metrics = calculator.calculate_metrics(llm_response, ground_truth, 100)

    # Verify metrics
    if all(k in metrics for k in ['precision', 'recall', 'accuracy', 'f1_score']):
        details = f"P={metrics['precision']:.2%}, R={metrics['recall']:.2%}, A={metrics['accuracy']:.2%}, F1={metrics['f1_score']:.2%}"
        log_test("TRUE Metrics Calculator", "PASS", details)
    else:
        log_test("TRUE Metrics Calculator", "FAIL", "Missing metric fields")

except Exception as e:
    log_test("TRUE Metrics Calculator", "FAIL", str(e))


print("\n" + "="*70)
print("TEST 4: OPENAI LLM INTEGRATION")
print("="*70)

try:
    from agent.llm_service import LLMService

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        log_test("OpenAI API Key", "FAIL", "API key not set")
    else:
        log_test("OpenAI API Key", "PASS", f"API key present: {api_key[:20]}...")

    # Initialize LLM service with OpenAI
    llm = LLMService(provider='openai', model='gpt-3.5-turbo')

    # Test simple generation
    test_prompt = "Say 'Hello from OpenAI' and nothing else."
    result = llm.generate(test_prompt)

    if result.get('success'):
        response = result['response']
        log_test("OpenAI Generation", "PASS", f"Response: {response[:100]}")
    else:
        log_test("OpenAI Generation", "FAIL", f"Error: {result.get('error', 'Unknown')}")

except Exception as e:
    log_test("OpenAI Integration", "FAIL", str(e))


print("\n" + "="*70)
print("TEST 5: ADAPTIVE PROMPT TUNER (INITIALIZATION)")
print("="*70)

try:
    from agent.adaptive_tuner import AdaptivePromptTuner
    from agent.llm_service import LLMService

    llm = LLMService(provider='openai', model='gpt-3.5-turbo')
    tuner = AdaptivePromptTuner(llm, max_iterations=3)

    log_test("Adaptive Tuner Init", "PASS", "Tuner initialized with OpenAI")

except Exception as e:
    log_test("Adaptive Tuner Init", "FAIL", str(e))


print("\n" + "="*70)
print("TEST 6: DYNAMIC PROMPT GENERATION")
print("="*70)

try:
    from agent.adaptive_tuner import AdaptivePromptTuner
    from agent.llm_service import LLMService

    llm = LLMService(provider='openai', model='gpt-3.5-turbo')
    tuner = AdaptivePromptTuner(llm, max_iterations=3)

    # Generate initial prompt
    initial_prompt = tuner.generate_initial_prompt(
        requirement='Identify high-value transactions over £250',
        context={'total_transactions': 3000, 'threshold': 250}
    )

    if initial_prompt and len(initial_prompt) > 50:
        log_test("Dynamic Prompt Generation", "PASS",
                f"Generated {len(initial_prompt)} char prompt")
        print(f"\n  Generated Prompt Preview:")
        print(f"  {initial_prompt[:200]}...")
    else:
        log_test("Dynamic Prompt Generation", "FAIL", "Prompt too short or empty")

except Exception as e:
    log_test("Dynamic Prompt Generation", "FAIL", str(e))


print("\n" + "="*70)
print("TEST 7: TRUE AI AGENT INITIALIZATION")
print("="*70)

try:
    from agent.true_ai_agent import TrueAIAgent

    agent = TrueAIAgent(
        llm_provider='openai',
        model='gpt-3.5-turbo',
        data_dir='data',
        output_dir='results',
        max_tuning_iterations=3
    )

    status = agent.get_status()
    log_test("TRUE AI Agent Init", "PASS",
            f"Provider: {status['provider']}, Model: {status['model']}")

    # Load data
    load_result = agent.load_data()
    if load_result.get('success'):
        log_test("Agent Data Loading", "PASS",
                f"Loaded {load_result['transactions']} transactions")
    else:
        log_test("Agent Data Loading", "FAIL", "Failed to load data")

except Exception as e:
    log_test("TRUE AI Agent Init", "FAIL", str(e))


print("\n" + "="*70)
print("TEST 8: SINGLE ITERATION PROMPT TEST (OpenAI)")
print("="*70)

try:
    from agent.true_ai_agent import TrueAIAgent

    agent = TrueAIAgent(
        llm_provider='openai',
        model='gpt-3.5-turbo',
        data_dir='data',
        max_tuning_iterations=1  # Just 1 iteration for quick test
    )

    agent.load_data()

    print("\n  Testing FW15 with single iteration...")
    print("  This will:")
    print("  1. Generate initial prompt using OpenAI")
    print("  2. Test prompt on transaction data")
    print("  3. Calculate TRUE metrics (ground truth comparison)")
    print("  4. Report results\n")

    result = agent.analyze_with_dynamic_tuning(
        requirement='fw15',
        requirement_description='Identify high-value transactions over £250',
        target_precision=0.98,
        target_accuracy=0.98,
        test_formats=False
    )

    if result.get('best_metrics'):
        metrics = result['best_metrics']
        details = (f"P={metrics['precision']:.2%}, "
                  f"A={metrics['accuracy']:.2%}, "
                  f"F1={metrics['f1_score']:.2%}, "
                  f"Iterations={result.get('iterations', 0)}")
        log_test("Single Iteration Test", "PASS", details)

        # Show confusion matrix
        cm = metrics.get('confusion_matrix', {})
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {cm.get('true_positives', 0)}")
        print(f"    TN: {cm.get('true_negatives', 0)}")
        print(f"    FP: {cm.get('false_positives', 0)}")
        print(f"    FN: {cm.get('false_negatives', 0)}")
    else:
        log_test("Single Iteration Test", "FAIL", "No metrics returned")

except Exception as e:
    log_test("Single Iteration Test", "FAIL", str(e))
    logger.error(f"Single iteration test error: {e}", exc_info=True)


print("\n" + "="*70)
print("TEST 9: ADAPTIVE TUNING (3 ITERATIONS)")
print("="*70)

try:
    from agent.true_ai_agent import TrueAIAgent

    agent = TrueAIAgent(
        llm_provider='openai',
        model='gpt-3.5-turbo',
        data_dir='data',
        max_tuning_iterations=3
    )

    agent.load_data()

    print("\n  Running adaptive tuning with up to 3 iterations...")
    print("  Agent will:")
    print("  - Test initial prompt")
    print("  - If target not met, analyze failures")
    print("  - Generate improved prompt")
    print("  - Test again")
    print("  - Repeat until 98% target or max iterations\n")

    result = agent.analyze_with_dynamic_tuning(
        requirement='fw15',
        requirement_description='Identify high-value transactions over £250',
        target_precision=0.98,
        target_accuracy=0.98,
        test_formats=False
    )

    if result.get('best_metrics'):
        metrics = result['best_metrics']
        target_met = result.get('target_achieved', False)

        details = (f"Target: {'✓ MET' if target_met else '✗ NOT MET'}, "
                  f"P={metrics['precision']:.2%}, "
                  f"A={metrics['accuracy']:.2%}, "
                  f"Iterations={result.get('iterations', 0)}")

        log_test("Adaptive Tuning (3 iter)", "PASS", details)

        # Show history
        history = result.get('tuning_result', {}).get('history', [])
        if history:
            print(f"\n  Iteration History:")
            for i, h in enumerate(history, 1):
                m = h.get('metrics', {})
                print(f"    Iter {i}: P={m.get('precision', 0):.2%}, "
                      f"A={m.get('accuracy', 0):.2%}, "
                      f"F1={m.get('f1_score', 0):.2%}")
    else:
        log_test("Adaptive Tuning (3 iter)", "FAIL", "No metrics returned")

except Exception as e:
    log_test("Adaptive Tuning (3 iter)", "FAIL", str(e))
    logger.error(f"Adaptive tuning error: {e}", exc_info=True)


print("\n" + "="*70)
print("TEST 10: NLP CLI PARSER")
print("="*70)

try:
    from agent.nlp_cli import NLPCommandParser

    parser = NLPCommandParser()

    test_commands = [
        "use openai",
        "analyze fw15",
        "show me the metrics",
        "tune the prompts",
        "switch to gpt-4"
    ]

    all_parsed = True
    for cmd in test_commands:
        cmd_type, params = parser.parse(cmd)
        if not cmd_type:
            all_parsed = False
            break

    if all_parsed:
        log_test("NLP Parser", "PASS", f"Parsed {len(test_commands)} commands")
    else:
        log_test("NLP Parser", "FAIL", "Failed to parse some commands")

except Exception as e:
    log_test("NLP Parser", "FAIL", str(e))


print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

summary = test_results['summary']
print(f"\nTotal Tests: {summary['total']}")
print(f"Passed: {summary['passed']} ✓")
print(f"Failed: {summary['failed']} ✗")
print(f"Success Rate: {(summary['passed']/summary['total']*100):.1f}%")

# Save results
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)
results_file = results_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(results_file, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")

print("\n" + "="*70)
print("DETAILED TEST RESULTS")
print("="*70)

for test in test_results['tests']:
    print(f"\n{test['name']}: {test['status']}")
    if test['details']:
        print(f"  {test['details']}")

print("\n✓ Testing Complete!")

# Exit with appropriate code
sys.exit(0 if summary['failed'] == 0 else 1)
