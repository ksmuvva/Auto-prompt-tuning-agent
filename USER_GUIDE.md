# User Guide: Auto-Prompt-Tuning-Agent for Financial Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [FW Requirements Analysis](#fw-requirements-analysis)
4. [Prompt Strategies](#prompt-strategies)
5. [Model Selection](#model-selection)
6. [Validation & Metrics](#validation--metrics)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

The Auto-Prompt-Tuning-Agent is an AI-powered system for automated bank transaction analysis with focus on precision, accuracy, and bias-free operation.

### Key Features
- **98% Precision & Accuracy Targets** - Validates against ground truth
- **<2% Bias Target** - Ensures fair and consistent analysis
- **FW Requirements** - Comprehensive financial analysis (FW15-FW50)
- **Multiple Strategies** - Template-based, Dynamic generation, or Hybrid
- **Multi-Model Support** - OpenAI, Anthropic Claude, Google Gemini, Cohere, Mistral, local models

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Auto-prompt-tuning-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python generate_sample_data.py
```

This creates:
- `data/transactions_01.csv` through `transactions_30.csv` (3,000 transactions)
- `data/ground_truth_master.json` (validation data - never exposed to LLM)

### 3. Start the CLI

```bash
python -m agent.cli
```

Or initialize with a specific provider:

```bash
python -m agent.cli --provider openai
python -m agent.cli --provider gemini
```

### 4. Basic Workflow

```
agent> load                    # Load transaction data
agent> analyze-all-fw          # Run all FW analyses
agent> validate-results        # Validate against ground truth
agent> show-metrics            # View performance metrics
```

---

## FW Requirements Analysis

### FW15: High-Value Transactions (>£250)

Identifies and groups all transactions exceeding £250.

```
agent> analyze-fw15
```

**Output:**
- List of all high-value transactions
- Total count and amount
- Average transaction value
- Grouping by merchant/category

**Validation:**
- Precision ≥ 98%
- Accuracy ≥ 98%
- Zero false negatives

### FW20: Luxury Brands & Money Transfers

Detects luxury brand purchases and money transfer services.

```
agent> analyze-fw20-luxury      # Luxury brands only
agent> analyze-fw20-transfer    # Money transfers only
```

**Luxury Brands Detected:**
- Gucci, Louis Vuitton, Prada, Chanel, Rolex, Hermès, Cartier, Burberry, etc.

**Money Transfer Services:**
- Western Union, MoneyGram, Wise, PayPal Transfer, Revolut, etc.

### FW25: Missing Audit Trail

Identifies transactions lacking proper audit documentation.

```
agent> analyze-fw25
```

**Flags:**
- Unknown merchants
- Missing merchant information
- Large cash withdrawals without notes
- Anonymous transactions

### FW30: Missing Months

Detects gaps in statement coverage (expects 6 consecutive months).

```
agent> analyze-fw30
```

**Output:**
- Date range analysis
- List of actual months with transactions
- List of missing months
- Continuity assessment

### FW40: Light-Touch Fraud Detection

Identifies errors and potential fraud indicators without full investigation.

```
agent> analyze-fw40
```

**Detection Categories:**
1. **Misspellings** - Bank names, merchant names
2. **Calculation Errors** - Decimal point errors, incorrect totals
3. **Data Quality Issues** - Duplicates, invalid dates, missing fields

### FW45: Gambling Analysis

Analyzes gambling transactions over a 6-month period.

```
agent> analyze-fw45
```

**Gambling Operators:**
- Bet365, William Hill, Paddy Power, Ladbrokes, Betfair, Sky Bet, 888 Casino, etc.

**Analysis Includes:**
- Total spend
- Frequency analysis
- Pattern detection (increasing/decreasing trend)
- Risk indicators

### FW50: Large Debt Payments

Tracks debt repayments ≥£500.

```
agent> analyze-fw50
```

**Debt Categories:**
- Credit card payments
- Loan repayments
- Mortgage payments
- Finance agreements

### Analyze All Requirements

Run comprehensive analysis of all FW requirements:

```
agent> analyze-all-fw
```

---

## Prompt Strategies

The system supports three prompt generation strategies:

### 1. Template-Based (Default)

Uses predefined, optimized prompt templates for each FW requirement.

**Advantages:**
- Fast (no LLM calls for prompt generation)
- Consistent results
- Well-tested templates

**Usage:**
```
agent> set-strategy template
agent> analyze-fw15
```

### 2. Dynamic Generation

Uses meta-prompting to generate prompts tailored to your specific data.

**Advantages:**
- Adapts to your data patterns
- Learns from failures
- Achieves higher precision

**Usage:**
```
agent> set-strategy dynamic
agent> analyze-fw15
```

### 3. Hybrid Approach

Tries template first, falls back to dynamic if performance is poor.

**Advantages:**
- Best of both worlds
- Automatic optimization
- Balanced performance/cost

**Usage:**
```
agent> set-strategy hybrid
agent> analyze-fw15
```

### Comparing Strategies

```
agent> compare-strategies
```

This runs the same analysis with all three strategies and provides a recommendation.

---

## Model Selection

### Supported Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| OpenAI | GPT-4, GPT-4-Turbo, GPT-3.5-Turbo | OPENAI_API_KEY |
| Anthropic | Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku | ANTHROPIC_API_KEY |
| Google | Gemini-Pro, Gemini-Pro-Vision | GOOGLE_API_KEY |
| Cohere | Command, Command-Light | COHERE_API_KEY |
| Mistral | Mistral-Medium, Mistral-Small | MISTRAL_API_KEY |
| Ollama | Llama2, Mistral, CodeLlama | Local (no API key) |
| LM Studio | Any local model | Local (no API key) |
| Mock | Testing without API calls | None |

### Setting Up API Keys

```bash
# Linux/Mac
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"

# Windows PowerShell
$env:OPENAI_API_KEY="your-key-here"
$env:ANTHROPIC_API_KEY="your-key-here"
$env:GOOGLE_API_KEY="your-key-here"
```

### Switching Providers

```
agent> list-models              # See all available models
agent> set-provider gemini      # Switch to Google Gemini
agent> set-model gemini-pro     # Set specific model
```

### Comparing Models

```
agent> compare-models
```

Runs the same analysis on multiple models and compares:
- Performance (precision, accuracy)
- Speed (latency)
- Cost (API costs)

---

## Validation & Metrics

### Ground Truth System

The system uses a master file (`data/ground_truth_master.json`) that contains validated answers for all FW requirements. This file is **never exposed to the LLM** - it's only used for validation.

### Loading Ground Truth

```
agent> load-ground-truth
```

### Validating Results

After running any analysis:

```
agent> validate-results
```

This compares LLM predictions against ground truth and calculates:
- **Precision** = TP / (TP + FP)
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Recall** = TP / (TP + FN)
- **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)

### Viewing Metrics

```
agent> show-metrics
```

Example output:
```
=== PERFORMANCE METRICS ===
Current Performance:
  Precision: 0.9876 (98.76%)
  Accuracy: 0.9891 (98.91%)
  F1 Score: 0.9883
  Bias Score: 0.0142 (1.42%)
```

### Checking Targets

```
agent> check-targets
```

Verifies:
- ✓ Precision ≥ 98%
- ✓ Accuracy ≥ 98%
- ✓ Bias < 2%

### Bias Detection

```
agent> bias-report
```

Tests for bias in:
1. **Merchant Name Variations** - "Gucci" vs "GUCCI" vs "gucci"
2. **Currency Formats** - "£250.00" vs "250 GBP" vs "GBP 250"
3. **Date Formats** - "2025-01-15" vs "15/01/2025" vs "15 Jan 2025"

Target: Overall bias < 2%

---

## Advanced Features

### Custom Prompts

Add your own prompt templates:

```
agent> add-prompt
```

Follow the interactive prompts to define:
- Prompt name
- Description
- Template text (use {data} and {threshold} placeholders)

### Comparative Analysis

Compare different approaches:

```
agent> compare-prompts          # Compare multiple prompts
agent> compare-models           # Compare different LLMs
agent> compare-strategies       # Compare template/dynamic/hybrid
```

### Recommendations

Get AI-powered recommendations:

```
agent> recommend-best performance    # Best for performance
agent> recommend-best cost           # Best for cost
agent> recommend-best balanced       # Balanced recommendation
```

### Agent Reasoning

Ask the agent questions:

```
agent> ask "How can I improve precision for FW15?"
agent> ask "Why is my accuracy low?"
agent> ask "What's the best strategy for my data?"
```

### Export Results

```
agent> export
```

Exports:
- `results/tuning_results_<timestamp>.json`
- `results/best_prompt_<timestamp>.txt`
- `results/metrics_<timestamp>.json`

---

## Troubleshooting

### Issue: "Agent not initialized"

**Solution:**
```
agent> init openai
```

### Issue: "Ground truth file not found"

**Solution:**
```bash
python generate_sample_data.py
```

### Issue: API key errors

**Solution:**
```bash
# Set your API keys
export OPENAI_API_KEY="sk-..."
```

Or use mock provider for testing:
```
agent> init mock
```

### Issue: Low precision/accuracy

**Solutions:**
1. Switch to dynamic prompt generation:
   ```
   agent> set-strategy dynamic
   ```

2. Try a more powerful model:
   ```
   agent> set-provider openai
   agent> set-model gpt-4
   ```

3. Run adaptive tuning:
   ```
   agent> adaptive-tune
   ```

### Issue: High bias detected

**Solution:**
Run bias report to see specific issues:
```
agent> bias-report
```

Then adjust prompts to be more format-agnostic.

### Issue: Slow performance

**Solutions:**
1. Use faster model: `set-model gpt-3.5-turbo`
2. Use template strategy: `set-strategy template`
3. Use local model: `set-provider ollama`

---

## Best Practices

### 1. Always Load Ground Truth

```
agent> load-ground-truth
```

This enables validation of all analyses.

### 2. Start with Template Strategy

For initial testing, use template-based prompts (fastest and most consistent).

### 3. Validate Frequently

After each analysis, run `validate-results` to ensure you're meeting targets.

### 4. Use Hybrid for Production

```
agent> set-strategy hybrid
```

Gets the best balance of speed and accuracy.

### 5. Monitor Bias

Regularly run `bias-report` to ensure fairness.

### 6. Export Important Results

```
agent> export
```

Save your best configurations for future use.

---

## Example Workflow

Complete workflow for financial analysis:

```bash
# 1. Start agent
python -m agent.cli --provider gemini

# 2. Load data and ground truth
agent> load
agent> load-ground-truth

# 3. Set strategy
agent> set-strategy hybrid

# 4. Run comprehensive analysis
agent> analyze-all-fw

# 5. Validate results
agent> validate-results
agent> show-metrics
agent> check-targets

# 6. Check bias
agent> bias-report

# 7. Compare approaches (optional)
agent> compare-strategies

# 8. Export results
agent> export

# 9. Get recommendations
agent> recommendations
```

---

## Support

For issues or questions:
1. Check the main README.md
2. Review ARCHITECTURE.md for system design
3. See FEATURES.md for detailed feature descriptions
4. Run `help` command in the CLI

---

**Version:** 1.0  
**Last Updated:** 2025
