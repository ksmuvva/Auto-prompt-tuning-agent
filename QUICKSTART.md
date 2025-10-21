# Quick Start Guide

Get up and running with the Prompt Tuning AI Agent in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation (30 seconds)

```bash
# Clone the repository
git clone <repo-url>
cd Auto-prompt-tuning-agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install pandas numpy pyyaml
```

## Generate Sample Data (30 seconds)

```bash
python generate_sample_data.py
```

This creates 30 CSV files with 3,000 bank transactions in the `data/` directory.

## Launch the Agent (Instant)

```bash
python -m agent.cli
```

## Your First Session (3 minutes)

```
╔══════════════════════════════════════════════════════════════╗
║     PROMPT TUNING AI AGENT                                   ║
╚══════════════════════════════════════════════════════════════╝

Auto-initializing with mock provider for testing...
✓ Agent initialized successfully!

Ready! Type 'help' for commands.

agent> load
Loading transaction data...
✓ Loaded 3000 transactions
  High-value transactions (>250 GBP): 450
  Statistical anomalies detected: 150

agent> list-prompts
=== AVAILABLE PROMPTS (8) ===
1. direct_concise
   Direct, concise instructions with bullet points
2. detailed_step_by_step
   Detailed step-by-step analysis instructions
3. json_structured
   Requests structured JSON output
4. role_based_expert
   Role-based prompt with expert persona
5. few_shot_examples
   Includes examples of desired output
6. chain_of_thought
   Encourages reasoning process
7. minimal
   Minimal prompt to test LLM capabilities
8. table_format
   Requests output in table format

agent> analyze quick
Running analysis in 'quick' mode...
This may take a few moments...

Progress: 1/3
Testing prompt template: direct_concise
Evaluated direct_concise: Composite Score = 0.756

Progress: 2/3
Testing prompt template: json_structured
Evaluated json_structured: Composite Score = 0.812

Progress: 3/3
Testing prompt template: role_based_expert
Evaluated role_based_expert: Composite Score = 0.845

✓ Analysis complete!
  Mode: quick
  Best Prompt: role_based_expert
  Best Score: 0.845

agent> best-prompt
=== BEST PROMPT ===
Name: role_based_expert
Score: 0.845

Template:
You are an expert fraud detection analyst with 10+ years of experience...
[Full prompt displayed]

agent> recommendations
=== AI RECOMMENDATIONS ===
{
  "timestamp": "2024-01-15T10:30:00",
  "best_prompt": "role_based_expert",
  "best_score": 0.845,
  "suggestions": [
    "Good performance! Run adaptive mode to further optimize"
  ]
}

agent> export
Exporting results...
✓ Exported files:
  tuning_results: results/tuning_results_1705318200.json
  best_prompt: results/best_prompt_1705318200.txt
  metrics: results/metrics_1705318200.json

agent> quit
Goodbye! Agent shutting down...
```

## What Just Happened?

1. **Agent initialized** with mock LLM (no API key needed)
2. **Loaded 3,000 transactions** from 30 CSV files
3. **Tested 3 prompts** against the data
4. **Evaluated performance** using 7 different metrics
5. **Identified best prompt** (role_based_expert with 84.5% score)
6. **Exported results** for further analysis

## Next Steps

### Try Adaptive Mode (AI-Powered Optimization)

```
agent> analyze adaptive
```

The agent will:
- Test all 8 built-in prompts
- Analyze performance
- **Autonomously generate improved prompts using AI**
- Test the new prompts
- Repeat until reaching target score (85%) or max iterations (3)

### Add Your Own Prompt

```
agent> add-prompt
Prompt name: my_custom
Description: My custom approach
Enter prompt template (use {data} and {threshold} as placeholders):
Type 'END' on a new line when finished:
Analyze this data: {data}
Find all transactions over {threshold} GBP.
END
✓ Custom prompt 'my_custom' added successfully!

agent> analyze full
```

### Use Real LLM (OpenAI or Anthropic)

```bash
# Set up API key
export OPENAI_API_KEY="sk-..."

# Launch agent
python -m agent.cli

# Initialize with OpenAI
agent> init openai
✓ Agent initialized with OpenAI provider

agent> analyze adaptive
```

## Understanding the Metrics

When you run analysis, the agent evaluates prompts on:

| Metric | What It Measures |
|--------|------------------|
| **Accuracy** (30%) | Correctly identified high-value transactions |
| **F1 Score** (25%) | Balance between precision and recall |
| **Completeness** (15%) | All required information present |
| **Format Quality** (15%) | Proper structure and organization |
| **Specificity** (15%) | Level of detail (IDs, amounts, dates) |

**Composite Score** = Weighted average (0.0 to 1.0)

- **0.60-0.70**: Acceptable
- **0.70-0.85**: Good
- **0.85+**: Excellent

## Common Commands Cheat Sheet

```bash
# Setup
init mock/openai/anthropic  # Initialize agent
status                      # Check agent status

# Data
load                        # Load CSV data
data-info                   # Show data statistics

# Analysis
analyze quick               # Quick test (3 prompts)
analyze full                # Test all prompts
analyze adaptive            # AI optimization

# Results
results                     # Show detailed results
best-prompt                 # View best prompt
export                      # Save results

# Agent
ask <question>              # Ask the agent
reset                       # Reset state
help                        # Show all commands
quit                        # Exit
```

## Tips for Best Results

1. **Start with Quick Mode**: Fastest way to get initial results
2. **Use Adaptive Mode for Production**: Best performance through AI optimization
3. **Add Custom Prompts**: Test your own prompt strategies
4. **Check Metrics**: Understand which prompts perform better and why
5. **Export Results**: Save for comparison and documentation

## Programmatic Usage

Instead of CLI, use Python directly:

```python
from agent.core import PromptTuningAgent

# Initialize
agent = PromptTuningAgent(llm_provider="mock")

# Load data
agent.load_and_process_data()

# Run analysis
results = agent.run_analysis(mode="adaptive")

# Get best prompt
print(f"Best: {results['best_prompt']} ({results['best_score']:.3f})")

# Export
agent.export_results()
```

## Troubleshooting

### "No module named 'pandas'"
```bash
pip install -r requirements.txt
```

### "No data files found"
```bash
python generate_sample_data.py
```

### "Agent not initialized"
Type `init mock` in the CLI first

### "API key not found"
For real LLMs, set environment variable:
```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

## What Makes This a True AI Agent?

Unlike traditional scripts, this agent:

1. **Learns**: Stores knowledge across sessions
2. **Reasons**: Can think about queries using LLM
3. **Optimizes**: Autonomously improves prompts
4. **Adapts**: Changes strategy based on results
5. **Decides**: Chooses best approaches without human input

## Example Use Cases

- **Prompt Engineers**: Find optimal prompts systematically
- **Data Analysts**: Automate transaction analysis
- **Fraud Detection**: Identify suspicious patterns
- **Compliance Teams**: Monitor high-value transactions
- **Researchers**: Study prompt effectiveness

## Ready to Dive Deeper?

- Read [README.md](README.md) for full documentation
- Check [TOOLS.md](TOOLS.md) for technical details
- Explore `agent/` source code
- Try different prompt templates
- Experiment with adaptive mode

---

**You're now ready to use the Prompt Tuning AI Agent!** 🎉
