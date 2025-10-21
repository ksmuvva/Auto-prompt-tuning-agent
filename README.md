# Prompt Tuning AI Agent

**Automated Prompt Optimization for Bank Transaction Analysis**

A sophisticated AI agent that autonomously tests, evaluates, and optimizes prompts for analyzing bank transaction data. The agent uses metrics-driven evaluation to find the best-performing prompts and can even generate improved prompts using AI.

## Features

### Core Capabilities
- **Automated Prompt Testing**: Test multiple prompt templates against your data
- **Metrics-Driven Evaluation**: Comprehensive scoring system with accuracy, precision, recall, F1, completeness, format quality, and specificity
- **AI-Powered Optimization**: Agent generates improved prompts based on performance feedback
- **Multiple LLM Support**: Works with OpenAI, Anthropic, or mock providers
- **Interactive CLI**: User-friendly command-line interface
- **Memory System**: Agent learns from past interactions and stores knowledge
- **Custom Prompts**: Add your own prompt templates via CLI

### Analysis Features
- Identify transactions above threshold (default: 250 GBP)
- Detect anomalies using statistical methods
- LLM-powered pattern recognition
- Ground truth validation
- Comprehensive reporting

## Architecture

```
Auto-prompt-tuning-agent/
├── agent/
│   ├── core.py              # Main AI agent with memory & reasoning
│   ├── cli.py               # Interactive CLI interface
│   ├── llm_service.py       # LLM integration (OpenAI, Anthropic, Mock)
│   ├── data_processor.py   # CSV data processing
│   ├── prompt_tuner.py      # Automated prompt optimization engine
│   └── metrics.py           # Evaluation metrics system
├── prompts/
│   └── templates.py         # 8+ built-in prompt templates
├── data/                    # CSV transaction data
├── config/
│   └── config.json          # Configuration
├── results/                 # Output files
└── logs/                    # Agent memory & logs
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Auto-prompt-tuning-agent
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys (Optional)
If using OpenAI or Anthropic:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Generate Sample Data
```bash
python generate_sample_data.py
```
This creates 30 CSV files with 3,000 realistic bank transactions.

## Quick Start

### Interactive Mode (Recommended)
```bash
python -m agent.cli
```

### Example Session
```
agent> init mock                    # Initialize with mock LLM
agent> load                         # Load CSV data
agent> analyze quick                # Quick test (3 prompts)
agent> best-prompt                  # View best performing prompt
agent> recommendations              # Get AI recommendations
agent> export                       # Export results
```

## Usage Modes

### 1. Quick Test
Test 3 high-performing prompt templates (fastest):
```
agent> analyze quick
```

### 2. Full Test
Test all 8+ available prompt templates:
```
agent> analyze full
```

### 3. Adaptive Tuning (AI-Powered)
Agent autonomously generates and tests improved prompts:
```
agent> analyze adaptive
```
This mode:
- Tests all prompts
- Identifies best performer
- Uses LLM to generate optimized prompts
- Iteratively improves until target score or max iterations

## CLI Commands

### Setup
| Command | Description |
|---------|-------------|
| `init <provider>` | Initialize agent (openai, anthropic, mock) |
| `config` | Show configuration |
| `status` | Show agent status |

### Data Operations
| Command | Description |
|---------|-------------|
| `load` | Load CSV transaction data |
| `data-info` | Show data statistics |

### Prompt Management
| Command | Description |
|---------|-------------|
| `list-prompts` | List all prompt templates |
| `show-prompt <name>` | View specific prompt |
| `add-prompt` | Add custom prompt (interactive) |

### Analysis
| Command | Description |
|---------|-------------|
| `analyze <mode>` | Run analysis (quick/full/adaptive) |
| `quick-test` | Quick test shortcut |
| `full-test` | Full test shortcut |
| `adaptive-tune` | Adaptive tuning shortcut |

### Results
| Command | Description |
|---------|-------------|
| `results` | Show latest results |
| `best-prompt` | Display best prompt |
| `recommendations` | Get AI recommendations |
| `export` | Export all results |

### Agent Interaction
| Command | Description |
|---------|-------------|
| `ask <question>` | Ask agent a question |
| `think <query>` | Agent reasoning |
| `reset` | Reset agent state |

## Programmatic Usage

### Python API
```python
from agent.core import PromptTuningAgent

# Initialize agent
agent = PromptTuningAgent(
    llm_provider="openai",
    data_dir="data",
    config={'llm': {'model': 'gpt-4'}}
)

# Load data
data_results = agent.load_and_process_data()

# Run analysis
results = agent.run_analysis(mode="adaptive")

print(f"Best Prompt: {results['best_prompt']}")
print(f"Score: {results['best_score']:.3f}")

# Export results
agent.export_results()
```

### Custom Prompt Template
```python
from prompts.templates import PromptTemplateLibrary

library = PromptTemplateLibrary()

# Add custom template
library.create_custom_template(
    name="my_custom_prompt",
    template_text="""
    Analyze these transactions:
    {data}

    Find transactions above {threshold} GBP.
    """,
    description="My custom approach"
)
```

## Metrics Explained

The agent evaluates prompts using multiple metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Accuracy** | 30% | Correct identification of high-value transactions |
| **F1 Score** | 25% | Balance of precision and recall |
| **Completeness** | 15% | All required sections present |
| **Format Quality** | 15% | Proper structure and formatting |
| **Specificity** | 15% | Details like IDs, amounts, dates |

**Composite Score**: Weighted average of all metrics (0.0 - 1.0)

## Built-in Prompt Templates

1. **direct_concise**: Direct instructions with bullet points
2. **detailed_step_by_step**: Comprehensive step-by-step analysis
3. **json_structured**: Requests JSON output
4. **role_based_expert**: Expert fraud analyst persona
5. **few_shot_examples**: Includes example outputs
6. **chain_of_thought**: Encourages reasoning process
7. **minimal**: Minimal instructions
8. **table_format**: Table-based output

Plus any AI-generated optimized prompts during adaptive tuning.

## Configuration

Edit `config/config.json`:

```json
{
  "tuning": {
    "max_iterations": 3,        // Adaptive tuning iterations
    "target_score": 0.85,       // Target composite score
    "default_mode": "quick"
  },
  "data": {
    "threshold_gbp": 250.0,     // Transaction threshold
    "max_rows_for_llm": 1000    // Max rows sent to LLM
  }
}
```

## LLM Providers

### Mock Provider (Default)
No API key needed. Perfect for testing:
```bash
agent> init mock
```

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
agent> init openai
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
agent> init anthropic
```

## Output Files

All results saved to `results/` directory:

- `tuning_results_<timestamp>.json` - Complete tuning history
- `best_prompt_<timestamp>.txt` - Best performing prompt
- `metrics_<timestamp>.json` - Detailed metrics

## Advanced Features

### Agent Memory
The agent maintains memory across sessions:
- **Short-term**: Recent interactions (last 50)
- **Long-term**: Persistent knowledge
- **Learned patterns**: Performance insights

Memory stored in `logs/agent_memory.json`

### Autonomous Learning
In adaptive mode, the agent:
1. Tests all prompts
2. Analyzes performance
3. Generates improvement suggestions
4. Creates new optimized prompts
5. Tests new prompts
6. Repeats until target achieved

### AI Reasoning
Ask the agent questions:
```
agent> ask "Why is my F1 score low?"
agent> ask "How can I improve prompt performance?"
```

## Example Results

```
=== ANALYSIS COMPLETE ===
Mode: adaptive
Best Prompt: role_based_expert
Best Score: 0.872

Metrics:
  Accuracy: 0.91
  F1 Score: 0.88
  Completeness: 0.85
  Format Quality: 0.82
  Specificity: 0.90

Recommendations:
  ✓ Excellent performance! Current prompt is highly effective.
```

## Tools & Technologies

### Core Stack
- **Python 3.8+**
- **Pandas**: Data processing
- **NumPy**: Statistical analysis

### LLM Integration
- **OpenAI Python SDK**: GPT models
- **Anthropic Python SDK**: Claude models
- **Custom abstraction**: Provider-agnostic interface

### Architecture Patterns
- **Agent Pattern**: Autonomous decision-making
- **Strategy Pattern**: Pluggable LLM providers
- **Template Pattern**: Flexible prompt system
- **Memory Pattern**: Persistent learning

## Troubleshooting

### No module named 'pandas'
```bash
pip install -r requirements.txt
```

### API key errors
Check `.env` file and ensure API keys are set correctly.

### No data files found
Run `python generate_sample_data.py` to create sample data.

### Agent not initialized
Run `init <provider>` command first.

## Development

### Run Tests
```bash
pytest
```

### Code Style
```bash
black .
flake8 .
```

### Add New Prompt Template
Edit `prompts/templates.py` and add to `_initialize_templates()`

## Real-World Applications

This agent is designed for:
- **Prompt Engineering**: Systematically find best prompts
- **LLM Evaluation**: Compare prompt performance
- **Financial Analysis**: Transaction monitoring
- **Fraud Detection**: Anomaly identification
- **Compliance**: Regulatory reporting
- **Research**: Prompt optimization studies

## Future Enhancements

- [ ] Multi-objective optimization
- [ ] A/B testing framework
- [ ] Real-time monitoring dashboard
- [ ] Distributed prompt testing
- [ ] Neural prompt optimization
- [ ] Integration with more LLM providers

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues or questions:
- Open an issue on GitHub
- Check documentation
- Review example sessions

---

**Built with AI for AI** - This is a true AI agent that learns, adapts, and optimizes autonomously.
