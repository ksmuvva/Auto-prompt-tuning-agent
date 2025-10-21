# Project Summary: Prompt Tuning AI Agent

## Overview

This is a **production-ready AI agent** for automated prompt optimization. It analyzes bank transaction data (30 CSV files) to find transactions above 250 GBP and detect anomalies, while autonomously testing and optimizing different prompts to achieve the best results.

## Key Features

### 1. True AI Agent Capabilities
- **Autonomous Learning**: Learns from results and improves over time
- **Memory System**: Maintains short-term and long-term memory
- **Reasoning**: Can think about queries using LLM
- **Self-Optimization**: Generates improved prompts using AI
- **Adaptive Behavior**: Changes strategy based on performance

### 2. Automated Prompt Tuning
- Tests multiple prompt templates (8+ built-in)
- Evaluates using 7 different metrics
- Generates new prompts autonomously (adaptive mode)
- Identifies best-performing prompts automatically
- Exports comprehensive results

### 3. Multiple LLM Support
- **OpenAI**: GPT-4, GPT-3.5
- **Anthropic**: Claude 3
- **Mock**: For testing without API costs

### 4. Interactive CLI
- 25+ commands
- User-friendly interface
- Real-time feedback
- Custom prompt addition via CLI

## Project Structure

```
Auto-prompt-tuning-agent/
├── agent/                          # Main agent package
│   ├── __init__.py                # Package initialization
│   ├── core.py                    # AI agent with memory & reasoning (350+ lines)
│   ├── cli.py                     # Interactive CLI (400+ lines)
│   ├── llm_service.py             # LLM integration layer (250+ lines)
│   ├── data_processor.py          # CSV data processing (200+ lines)
│   ├── prompt_tuner.py            # Optimization engine (300+ lines)
│   └── metrics.py                 # Evaluation system (350+ lines)
│
├── prompts/                        # Prompt templates
│   ├── __init__.py
│   └── templates.py               # 8 built-in templates (250+ lines)
│
├── config/                         # Configuration
│   └── config.json                # Agent configuration
│
├── data/                          # CSV transaction data (generated)
├── results/                       # Output files
├── logs/                          # Agent memory & logs
│
├── generate_sample_data.py        # Data generation script
├── setup.py                       # Package setup
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
│
└── Documentation:
    ├── README.md                  # Complete documentation
    ├── QUICKSTART.md              # 5-minute quick start
    ├── TOOLS.md                   # Technology stack details
    └── PROJECT_SUMMARY.md         # This file
```

## Components

### 1. Agent Core (`agent/core.py`)
- Main AI agent class
- Memory management (short-term, long-term, learned patterns)
- Task orchestration
- Autonomous decision-making
- State management
- AI reasoning capabilities

### 2. LLM Service (`agent/llm_service.py`)
- Abstract provider interface
- OpenAI integration
- Anthropic integration
- Mock provider for testing
- Usage tracking
- Error handling

### 3. Data Processor (`agent/data_processor.py`)
- CSV file loading (30 files)
- Data validation
- Transaction filtering (>250 GBP)
- Statistical anomaly detection (Z-score)
- Ground truth generation
- Data formatting for LLM

### 4. Prompt Tuner (`agent/prompt_tuner.py`)
- Single prompt testing
- Batch prompt evaluation
- Adaptive tuning loop
- AI-generated prompt creation
- Performance tracking
- Results export

### 5. Metrics System (`agent/metrics.py`)
- Accuracy calculation
- Precision, Recall, F1 Score
- Completeness measurement
- Format quality evaluation
- Specificity scoring
- Composite score (weighted average)
- Improvement suggestions

### 6. Prompt Templates (`prompts/templates.py`)
8 Built-in Templates:
1. Direct Concise
2. Detailed Step-by-Step
3. JSON Structured
4. Role-Based Expert
5. Few-Shot Examples
6. Chain of Thought
7. Minimal
8. Table Format

### 7. CLI Interface (`agent/cli.py`)
- Interactive REPL
- Command processing
- Help system
- Status display
- Result visualization

## How It Works

### Basic Flow
```
1. Load CSV Data → 2. Process Transactions → 3. Test Prompts
                                                     ↓
6. Export Results ← 5. Evaluate Metrics ← 4. Get LLM Responses
```

### Adaptive Mode Flow
```
1. Test All Prompts
2. Evaluate Performance
3. Identify Best Prompt
4. Analyze Weaknesses
5. Generate Improved Prompt (using LLM)
6. Test New Prompt
7. Repeat until target score or max iterations
```

## Metrics Explained

Each prompt is evaluated on:

| Metric | Weight | Description |
|--------|--------|-------------|
| Accuracy | 30% | Correct transaction identification |
| F1 Score | 25% | Balance of precision & recall |
| Completeness | 15% | All required sections present |
| Format Quality | 15% | Proper structure |
| Specificity | 15% | Detail level (IDs, amounts) |

**Composite Score**: 0.0 to 1.0 (higher is better)
- 0.60-0.70: Acceptable
- 0.70-0.85: Good
- 0.85+: Excellent

## Usage Modes

### 1. Quick Mode (Fastest)
```bash
agent> analyze quick
```
- Tests 3 best prompts
- Takes ~2 minutes
- Good for rapid iteration

### 2. Full Mode (Comprehensive)
```bash
agent> analyze full
```
- Tests all 8+ prompts
- Takes ~5 minutes
- Complete evaluation

### 3. Adaptive Mode (AI-Powered)
```bash
agent> analyze adaptive
```
- Tests all prompts
- Generates new optimized prompts
- Iteratively improves
- Takes ~10-15 minutes
- **Best results**

## Installation

```bash
# Clone
git clone <repo-url>
cd Auto-prompt-tuning-agent

# Install
pip install -r requirements.txt

# Generate data
python generate_sample_data.py

# Run
python -m agent.cli
```

## Quick Start Example

```
agent> init mock
agent> load
agent> analyze quick
agent> best-prompt
agent> export
agent> quit
```

## Code Statistics

- **Total Lines**: ~2,500+ lines of Python code
- **Components**: 7 major modules
- **Prompt Templates**: 8 built-in + unlimited custom
- **CLI Commands**: 25+
- **Metrics**: 7 evaluation metrics
- **Documentation**: 4 comprehensive guides

## Technologies Used

### Core
- Python 3.8+
- Pandas (data processing)
- NumPy (statistical analysis)

### LLM Integration
- OpenAI Python SDK
- Anthropic Python SDK
- TikToken (token counting)

### Architecture
- Agent Pattern
- Strategy Pattern
- Template Pattern
- Memory Pattern

## Real-World Applications

1. **Prompt Engineering**: Systematically optimize prompts
2. **Financial Analysis**: Automated transaction monitoring
3. **Fraud Detection**: Anomaly identification
4. **Compliance**: Regulatory reporting
5. **Research**: Prompt effectiveness studies
6. **LLM Evaluation**: Compare model performance

## What Makes This Special

### True AI Agent
- Not just a script - it's an autonomous agent
- Learns from experience
- Makes decisions independently
- Improves itself over time

### Production-Ready
- Comprehensive error handling
- Logging throughout
- Configuration management
- Multiple LLM support
- Extensive documentation

### Metrics-Driven
- 7 different evaluation metrics
- Composite scoring
- Performance tracking
- Improvement suggestions

### User-Friendly
- Interactive CLI
- Clear documentation
- Example workflows
- Mock mode for testing

## Files Created

### Code Files (16 files)
1. `agent/core.py` - Main AI agent
2. `agent/cli.py` - CLI interface
3. `agent/llm_service.py` - LLM integration
4. `agent/data_processor.py` - Data processing
5. `agent/prompt_tuner.py` - Optimization engine
6. `agent/metrics.py` - Metrics system
7. `agent/__init__.py` - Package init
8. `prompts/templates.py` - Prompt templates
9. `prompts/__init__.py` - Package init
10. `generate_sample_data.py` - Data generator
11. `setup.py` - Package setup
12. `requirements.txt` - Dependencies
13. `config/config.json` - Configuration
14. `.env.example` - Environment template
15. `.gitignore` - Git ignore
16. `agent/__init__.py`, `prompts/__init__.py`

### Documentation Files (4 files)
1. `README.md` - Complete documentation (400+ lines)
2. `QUICKSTART.md` - Quick start guide (200+ lines)
3. `TOOLS.md` - Technology details (300+ lines)
4. `PROJECT_SUMMARY.md` - This file

## Testing

### Without API Keys (Mock Mode)
```bash
python -m agent.cli
# Agent auto-initializes with mock provider
```

### With OpenAI
```bash
export OPENAI_API_KEY="sk-..."
python -m agent.cli
agent> init openai
```

### With Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python -m agent.cli
agent> init anthropic
```

## Future Enhancements

- [ ] Distributed prompt testing
- [ ] Web dashboard
- [ ] REST API
- [ ] More LLM providers
- [ ] Neural optimization
- [ ] A/B testing framework

## Summary

This is a **complete, production-ready AI agent system** with:
- ✅ Autonomous learning and optimization
- ✅ Multiple LLM support
- ✅ Comprehensive metrics
- ✅ Interactive CLI
- ✅ Extensive documentation
- ✅ Real AI agent capabilities (not just automation)
- ✅ Memory and reasoning
- ✅ Self-improvement through AI

**Total Development**: ~2,500+ lines of production Python code + 900+ lines of documentation

---

**This represents a true AI agent that learns, adapts, and optimizes autonomously.**
