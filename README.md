# Auto-prompt-tuning-agent

An AI agent that automatically tunes and optimizes prompts for better LLM performance.

## Features

- **Automatic Prompt Optimization**: Iteratively refines prompts to improve quality
- **Comprehensive Metrics**: Evaluates prompts based on clarity, specificity, and token efficiency
- **Batch Processing**: Optimize multiple prompts simultaneously
- **Flexible Configuration**: Customize optimization parameters and LLM settings
- **Multi-Provider Support**: Works with OpenAI, Anthropic, and local models

## Installation

```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -r requirements-dev.txt
```

## Quick Start

```python
from src.agent import PromptTuningAgent

# Create an agent
agent = PromptTuningAgent()

# Optimize a prompt
result = agent.optimize_prompt("Write a function")

print(f"Original: {result['original_prompt']}")
print(f"Optimized: {result['optimized_prompt']}")
print(f"Score: {result['score']}")
```

## Testing

This project includes comprehensive test coverage (95%) with 225+ tests covering:

- **Unit tests**: Testing individual components
- **Integration tests**: Testing component interactions
- **Edge case tests**: Boundary conditions, unicode, special characters
- **Micro tests**: Fine-grained functionality testing

Run all tests:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run specific test categories:
```bash
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m edge_case     # Edge cases only
pytest tests/ -m micro         # Micro tests only
pytest tests/ -m integration   # Integration tests only
```

## Test Coverage

The project achieves **95% code coverage** across all modules:

- `src/agent/config.py`: 89% coverage
- `src/agent/core.py`: 100% coverage
- `src/agent/llm_interface.py`: 96% coverage
- `src/agent/metrics.py`: 94% coverage
- `src/agent/prompt_tuner.py`: 96% coverage

## Project Structure

```
Auto-prompt-tuning-agent/
├── src/
│   └── agent/
│       ├── __init__.py          # Package exports
│       ├── config.py            # Configuration classes
│       ├── core.py              # Main agent class
│       ├── llm_interface.py     # LLM integration
│       ├── metrics.py           # Metrics calculation
│       └── prompt_tuner.py      # Tuning logic
├── tests/
│   ├── conftest.py              # Pytest fixtures
│   ├── test_config.py           # Config tests
│   ├── test_core.py             # Core agent tests
│   ├── test_edge_cases.py       # Edge case tests
│   ├── test_integration.py      # Integration tests
│   ├── test_llm_interface.py    # LLM interface tests
│   ├── test_metrics.py          # Metrics tests
│   └── test_prompt_tuner.py     # Tuner tests
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── pytest.ini                   # Pytest configuration
└── setup.py                     # Package setup
```

## Configuration

Customize the agent behavior:

```python
from src.agent import AgentConfig, LLMConfig, LLMProvider

llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    temperature=0.8,
    max_tokens=2000
)

agent_config = AgentConfig(
    llm_config=llm_config,
    max_iterations=20,
    convergence_threshold=0.01
)

agent = PromptTuningAgent(agent_config)
```

## Examples

### Batch Optimization

```python
prompts = [
    "Explain machine learning",
    "Write a sorting algorithm",
    "Describe quantum computing"
]

results = agent.optimize_batch(prompts)
for result in results:
    print(f"Score: {result['score']}")
```

### Evaluation Only

```python
result = agent.evaluate_prompt("Your prompt here")
metrics = result['metrics']
print(f"Clarity: {metrics['clarity']}")
print(f"Specificity: {metrics['specificity']}")
```

### Session Management

```python
# Optimize multiple prompts
agent.optimize_prompt("Prompt 1")
agent.optimize_prompt("Prompt 2")

# View history
history = agent.get_session_history()
print(f"Optimized {len(history)} prompts")

# Get statistics
stats = agent.get_stats()
print(f"Total LLM calls: {stats['llm_stats']['call_count']}")

# Reset for new session
agent.reset_session()
```

## Test Categories

### Unit Tests
Basic functionality tests for individual components:
- Configuration validation
- Metrics calculation
- LLM interface operations
- Core agent methods

### Edge Case Tests
Boundary conditions and unusual inputs:
- Unicode and emoji support
- Very long/short prompts
- Special characters
- Null/empty inputs
- Extreme configuration values

### Micro Tests
Fine-grained testing of specific behaviors:
- Parameter validation
- State management
- Error handling
- Data flow

### Integration Tests
End-to-end workflow testing:
- Complete optimization pipelines
- Component interaction
- Configuration persistence
- Real-world scenarios

## Contributing

1. Write tests for new features
2. Ensure all tests pass: `pytest tests/`
3. Maintain >90% code coverage
4. Follow existing code style

## License

See LICENSE file for details.
