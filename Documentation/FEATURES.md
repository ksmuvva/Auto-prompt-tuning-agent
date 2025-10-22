# Advanced Features Guide

This document describes the advanced features added to the Prompt Tuning Agent.

## Table of Contents

1. [Multi-Objective Optimization](#multi-objective-optimization)
2. [A/B Testing Framework](#ab-testing-framework)
3. [Real-Time Monitoring Dashboard](#real-time-monitoring-dashboard)
4. [Distributed Prompt Testing](#distributed-prompt-testing)
5. [Neural Prompt Optimization](#neural-prompt-optimization)
6. [Multiple LLM Providers](#multiple-llm-providers)
7. [Multi-Format Prompt Support](#multi-format-prompt-support)

---

## Multi-Objective Optimization

Optimize prompts across multiple objectives simultaneously using Pareto optimality.

### Features

- **Pareto Frontier Calculation**: Find non-dominated solutions
- **Multi-Objective Trade-off Analysis**: Understand trade-offs between objectives
- **Hypervolume Metric**: Measure solution set quality
- **Smart Recommendations**: Get best compromise solutions

### Usage

```python
from agent.multi_objective import MultiObjectiveOptimizer, Objective, Solution

# Initialize optimizer
optimizer = MultiObjectiveOptimizer()

# Add solutions from prompt testing
objective_config = {
    'accuracy': {'maximize': True, 'weight': 0.4},
    'latency': {'maximize': False, 'weight': 0.3},
    'cost': {'maximize': False, 'weight': 0.3}
}

# Add solutions
for prompt_id, metrics in test_results.items():
    optimizer.add_from_metrics(prompt_id, metrics, objective_config)

# Calculate Pareto frontier
pareto_frontier = optimizer.calculate_pareto_frontier()

# Get best compromise
best = optimizer.get_best_compromise()

# Analyze trade-offs
analysis = optimizer.analyze_tradeoffs()
print(f"Pareto frontier size: {analysis['pareto_frontier_size']}")
print(f"Trade-offs: {analysis['trade_offs']}")

# Get recommendations
recommendations = optimizer.get_recommendations()
for rec in recommendations:
    print(f"{rec['type']}: {rec['explanation']}")
```

### Key Concepts

**Pareto Dominance**: A solution A dominates B if A is better in all objectives or equal in some and better in at least one.

**Pareto Frontier**: Set of non-dominated solutions representing optimal trade-offs.

**Hypervolume**: Measures the volume of objective space dominated by solutions. Higher is better.

---

## A/B Testing Framework

Rigorous statistical testing for comparing prompt variants.

### Features

- **Statistical Significance Testing**: t-tests, Mann-Whitney U, bootstrap
- **Multi-Variant Testing**: Compare multiple prompts with Bonferroni correction
- **Sequential Testing**: Early stopping with O'Brien-Fleming boundaries
- **Sample Size Calculation**: Power analysis for proper experiment design
- **Detailed Results**: p-values, effect sizes, confidence intervals

### Usage

```python
from agent.ab_testing import ABTest, SignificanceTest
import numpy as np

# Initialize test
test = ABTest(
    test_name="Prompt Template Comparison",
    alpha=0.05,  # 95% confidence
    power=0.8,   # 80% power
    minimum_detectable_effect=0.05  # 5% improvement
)

# Calculate required sample size
n_required = test.calculate_required_sample_size(
    baseline_mean=0.80,
    baseline_std=0.10
)
print(f"Required samples per variant: {n_required}")

# Add variants
test.add_variant("control", "Control Prompt", "Original template")
test.add_variant("treatment", "Optimized Prompt", "Improved template")

# Collect data (simulated)
for _ in range(n_required):
    test.record_observation("control", np.random.normal(0.80, 0.10))
    test.record_observation("treatment", np.random.normal(0.85, 0.10))

# Run t-test
result = test.run_test("control", "treatment", SignificanceTest.WELCH_T_TEST)

print(f"P-value: {result.p_value:.4f}")
print(f"Significant: {result.significant}")
print(f"Winner: {result.winner}")
print(f"Recommendation: {result.recommendation}")

# Export results
test.export_results("ab_test_results.json")
```

### Multi-Variant Testing

```python
# Add multiple variants
test.add_variant("variant_a", "Variant A")
test.add_variant("variant_b", "Variant B")
test.add_variant("variant_c", "Variant C")

# Collect data for all variants
# ...

# Run pairwise comparisons with Bonferroni correction
results = test.run_multi_variant_test()

for result in results:
    print(f"{result.variant_a} vs {result.variant_b}: p={result.p_value:.4f}")
```

---

## Real-Time Monitoring Dashboard

Web-based dashboard for monitoring agent performance in real-time.

### Features

- **Live Metrics**: Real-time LLM call statistics, latency, cost
- **Active Test Tracking**: Monitor running tests
- **Historical Data**: 24-hour trend analysis
- **Cost Breakdown**: Per-provider cost tracking
- **WebSocket Updates**: Live data streaming

### Starting the Dashboard

```python
from agent.dashboard_server import run_dashboard

# Start dashboard server
run_dashboard(host='0.0.0.0', port=5000)
```

Open `http://localhost:5000` in your browser.

### Programmatic Usage

```python
from agent.monitoring import get_dashboard

# Get dashboard instance
dashboard = get_dashboard()

# Record LLM call
dashboard.record_llm_call(
    latency=1.2,
    tokens=1500,
    cost=0.045,
    success=True,
    provider='openai'
)

# Record test execution
test_id = "test_123"
dashboard.record_test_start(test_id, "multi_objective", variant_count=5)

# ... run test ...

dashboard.record_test_complete(test_id, {
    'status': 'completed',
    'best_score': 0.92
})

# Get dashboard data
data = dashboard.get_dashboard_data()
print(f"Active tests: {len(data['active_tests'])}")
print(f"Total cost (1h): ${data['summary']['last_hour']['total_cost']:.4f}")

# Get cost breakdown
cost_breakdown = dashboard.get_cost_breakdown(hours=24)
for provider, stats in cost_breakdown['by_provider'].items():
    print(f"{provider}: ${stats['cost']:.4f} ({stats['tokens']} tokens)")
```

---

## Distributed Prompt Testing

Execute prompt tests in parallel across multiple workers.

### Features

- **Parallel Execution**: Multi-threaded or multi-process execution
- **Task Queue**: Priority-based task scheduling
- **Load Balancing**: Distribute work evenly across workers
- **Fault Tolerance**: Automatic retry on failures
- **Progress Tracking**: Real-time progress monitoring

### Usage

```python
from agent.distributed_testing import DistributedTestExecutor, Task, TaskPriority

# Initialize executor
executor = DistributedTestExecutor(
    num_workers=8,
    use_processes=False,  # Use threads for I/O-bound tasks
    max_retries=3
)

# Create tasks
for i, prompt_template in enumerate(prompt_templates):
    task = Task(
        task_id=f"test_{i}",
        prompt_id=prompt_template.name,
        test_function="agent.prompt_tuner.test_single_prompt",
        kwargs={
            'prompt': prompt_template.text,
            'test_data': test_data,
            'ground_truth': ground_truth
        },
        priority=TaskPriority.NORMAL
    )
    executor.submit_task(task)

# Execute all tasks
results = executor.execute_all(timeout=300)

# Get progress
progress = executor.get_progress()
print(f"Completed: {progress['completed']}/{progress['total_tasks']}")
print(f"Success rate: {progress['progress_pct']:.1f}%")

# Get summary
summary = executor.get_summary()
print(f"Average task duration: {summary['avg_task_duration']:.2f}s")

# Shutdown
executor.shutdown()
```

### Convenience Function

```python
from agent.distributed_testing import parallel_test_prompts

# Test prompts in parallel
results = parallel_test_prompts(
    prompts=prompt_list,
    test_data=data,
    ground_truth=truth,
    llm_service=llm,
    metrics_evaluator=metrics,
    num_workers=4
)
```

---

## Neural Prompt Optimization

Advanced optimization using embeddings and evolutionary algorithms.

### Features

- **Embedding-Based Search**: Find semantically similar high-performing prompts
- **Genetic Algorithm**: Evolve prompts through mutation and crossover
- **Semantic Clustering**: Group similar prompts
- **Iterative Optimization**: Continuous improvement loop

### Usage

#### Semantic Search Optimization

```python
from agent.neural_optimization import NeuralPromptOptimizer

optimizer = NeuralPromptOptimizer(embedding_model="sentence-transformers")

# Define evaluation function
def evaluate_prompt(prompt_text):
    # Test prompt and return score
    result = test_prompt(prompt_text)
    return result['accuracy']

# Optimize using semantic search
results = optimizer.optimize_via_semantic_search(
    seed_prompts=initial_prompts,
    evaluation_func=evaluate_prompt,
    iterations=5,
    candidates_per_iteration=10
)

print("Best prompts found:")
for prompt in results['best_prompts'][:3]:
    print(f"Score: {prompt['score']:.3f}")
    print(f"Text: {prompt['text'][:100]}...")
```

#### Genetic Algorithm Optimization

```python
# Optimize using genetic algorithm
results = optimizer.optimize_via_genetic_algorithm(
    seed_prompts=initial_prompts,
    evaluation_func=evaluate_prompt,
    generations=10,
    population_size=20
)

print(f"Best prompt after {len(results['history'])} generations:")
best = results['best_prompts'][0]
print(f"Score: {best['score']:.3f}")
print(f"Text: {best['text']}")

# Analyze evolution
for gen in results['history']:
    print(f"Generation {gen['generation']}: "
          f"Best={gen['best_score']:.3f}, Avg={gen['avg_score']:.3f}")
```

---

## Multiple LLM Providers

Support for 8+ LLM providers including cloud APIs and local models.

### Supported Providers

1. **OpenAI** (GPT-4, GPT-3.5)
2. **Anthropic** (Claude)
3. **Google Gemini**
4. **Cohere**
5. **Mistral AI**
6. **Ollama** (Local models)
7. **LM Studio** (Local models)
8. **Mock** (Testing)

### Usage

```python
from agent.llm_service import LLMService

# OpenAI
llm = LLMService(provider="openai", model="gpt-4", api_key="sk-...")

# Anthropic
llm = LLMService(provider="anthropic", model="claude-3-sonnet-20240229", api_key="sk-...")

# Google Gemini
llm = LLMService(provider="gemini", model="gemini-pro", api_key="...")

# Cohere
llm = LLMService(provider="cohere", model="command", api_key="...")

# Mistral
llm = LLMService(provider="mistral", model="mistral-medium", api_key="...")

# Ollama (local)
llm = LLMService(
    provider="ollama",
    model="llama2",
    base_url="http://localhost:11434"
)

# LM Studio (local)
llm = LLMService(
    provider="lmstudio",
    model="local-model",
    base_url="http://localhost:1234/v1"
)

# Generate response
response = llm.generate(
    prompt="Analyze this data...",
    temperature=0.7,
    max_tokens=2000
)

print(response['response'])
print(f"Tokens used: {response['tokens_used']}")
print(f"Latency: {response['latency']:.2f}s")
```

### Switching Providers

```python
# Start with OpenAI
llm = LLMService(provider="openai")

# Switch to Anthropic
llm.switch_provider("anthropic", api_key="sk-...")

# Check usage stats
stats = llm.get_usage_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Success rate: {stats['success_rate']:.1%}")
```

---

## Multi-Format Prompt Support

Support for XML, JSON, YAML, Markdown, POML, and plain text prompt formats.

### Supported Formats

- **XML**: Hierarchical structured prompts
- **JSON**: JSON-based prompts
- **YAML**: Human-readable YAML
- **Markdown**: Markdown-formatted prompts
- **POML**: Custom Prompt Markup Language
- **Plain Text**: Simple text prompts

### Usage

```python
from agent.prompt_formats import PromptFormatConverter, PromptFormat, PromptStructure

converter = PromptFormatConverter()

# Create structured prompt
structure = PromptStructure(
    role="financial_analyst",
    task="Analyze transaction data",
    instructions=[
        "Identify high-value transactions",
        "Detect anomalies",
        "Generate report"
    ],
    context={"threshold": "250", "currency": "GBP"},
    constraints=["Max 5 minutes processing"]
)

# Format as XML
xml_formatter = XMLPromptFormatter()
xml_text = xml_formatter.format(structure)
print(xml_text)

# Format as JSON
json_formatter = JSONPromptFormatter()
json_text = json_formatter.format(structure)
print(json_text)

# Format as Markdown
md_formatter = MarkdownPromptFormatter()
md_text = md_formatter.format(structure)
print(md_text)
```

### Format Conversion

```python
# Convert between formats
json_text = '{"role": "analyst", "task": "Analyze data", ...}'

# Convert JSON to XML
xml_text = converter.convert(json_text, PromptFormat.JSON, PromptFormat.XML)

# Convert XML to Markdown
md_text = converter.convert(xml_text, PromptFormat.XML, PromptFormat.MARKDOWN)

# Auto-detect format
detected = converter.auto_detect_format(some_text)
print(f"Detected format: {detected}")
```

### Provider Optimization

```python
# Optimize format for specific provider
json_prompt = '{"role": "analyst", ...}'

# Optimize for Anthropic (prefers XML)
optimized = converter.optimize_for_provider(
    json_prompt,
    PromptFormat.JSON,
    "anthropic"
)
# Returns XML format

# Optimize for OpenAI (prefers Markdown)
optimized = converter.optimize_for_provider(
    json_prompt,
    PromptFormat.JSON,
    "openai"
)
# Returns Markdown format
```

### Example: POML Format

```poml
[ROLE: financial_analyst]
[TASK: Analyze transaction data for fraud detection]

[INSTRUCTIONS]
- Identify transactions above £250
- Calculate statistical anomalies
- Detect suspicious patterns
[/INSTRUCTIONS]

[CONTEXT]
threshold = 250
currency = GBP
dataset = transactions.csv
[/CONTEXT]

[OUTPUT: structured_json]
```

---

## Integration Example

Combining multiple features for powerful prompt optimization:

```python
from agent.multi_objective import MultiObjectiveOptimizer
from agent.ab_testing import ABTest
from agent.distributed_testing import DistributedTestExecutor
from agent.neural_optimization import NeuralPromptOptimizer
from agent.monitoring import get_dashboard
from agent.llm_service import LLMService

# 1. Start monitoring
dashboard = get_dashboard()

# 2. Initialize LLM
llm = LLMService(provider="openai", model="gpt-4")

# 3. Generate prompt variants using neural optimization
neural_opt = NeuralPromptOptimizer()
def evaluate(prompt):
    result = llm.generate(prompt)
    # Evaluate quality
    return calculate_score(result)

neural_results = neural_opt.optimize_via_genetic_algorithm(
    seed_prompts=initial_prompts,
    evaluation_func=evaluate,
    generations=5
)

# 4. Test top variants in parallel
executor = DistributedTestExecutor(num_workers=4)
# ... submit tasks ...
test_results = executor.execute_all()

# 5. Multi-objective optimization
optimizer = MultiObjectiveOptimizer()
for prompt_id, metrics in test_results.items():
    optimizer.add_from_metrics(prompt_id, metrics, objective_config)

pareto_frontier = optimizer.calculate_pareto_frontier()

# 6. A/B test top candidates
ab_test = ABTest(test_name="Final Comparison")
ab_test.add_variant("current", "Current Prompt")
ab_test.add_variant("optimized", "Optimized Prompt")

# Collect production data...
result = ab_test.run_test("current", "optimized")

# 7. Get comprehensive metrics
dashboard_data = dashboard.get_dashboard_data()
cost_breakdown = dashboard.get_cost_breakdown()

print(f"Total cost: ${cost_breakdown['total_cost']:.2f}")
print(f"Winner: {result.winner}")
print(f"Improvement: {result.effect_size:.1%}")
```

---

## Testing

Run comprehensive tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific feature tests
pytest tests/test_multi_objective.py -v
pytest tests/test_ab_testing.py -v
pytest tests/test_prompt_formats.py -v

# Run with coverage
pytest tests/ --cov=agent --cov-report=html
```

---

## Performance Tips

1. **Distributed Testing**: Use processes for CPU-bound tasks, threads for I/O-bound
2. **Monitoring**: Regularly check dashboard for cost optimization opportunities
3. **Multi-Objective**: Start with 2-3 objectives, add more as needed
4. **A/B Testing**: Calculate required sample size before collecting data
5. **Neural Optimization**: Use smaller populations/iterations first, then scale up
6. **Format Optimization**: Let the converter choose optimal format for each provider

---

## Troubleshooting

### Issue: Tests running slowly
**Solution**: Increase number of workers in DistributedTestExecutor

### Issue: High API costs
**Solution**: Check dashboard cost breakdown, consider local models (Ollama/LM Studio)

### Issue: A/B test shows no significant difference
**Solution**: Increase sample size or reduce minimum detectable effect

### Issue: Format conversion losing information
**Solution**: Check that all required fields are populated in PromptStructure

---

## Next Steps

- Explore the [Architecture Documentation](ARCHITECTURE.md)
- Read the [Quick Start Guide](QUICKSTART.md)
- Check out [example scripts](../examples/)
- Join our community for support

---

For questions or issues, please open an issue on GitHub.
