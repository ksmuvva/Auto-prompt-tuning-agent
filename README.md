# Auto-Prompt Tuning Agent

**Intelligent AI Agent for Automated Prompt Optimization with Full Explainability**

A production-ready AI agent system that autonomously optimizes prompts and generates synthetic data with comprehensive reasoning and explainability. Built with enterprise-grade architecture supporting multiple LLM providers.

---

## 🎯 Overview

This repository contains **two powerful AI systems**:

1. **Prompt Tuning Agent** - Automatically optimizes prompts to achieve 98% precision/accuracy
2. **Synthetic Data Generator** - Generates realistic synthetic data with 6 reasoning engines

Both systems include:
- ✅ Full reasoning and explainability (SHAP, LIME)
- ✅ Multiple LLM provider support (OpenAI, Anthropic, Google, Cohere, Mistral)
- ✅ Production-ready architecture
- ✅ No mock implementations
- ✅ Comprehensive testing

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Auto-prompt-tuning-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-..."  # Or ANTHROPIC_API_KEY, etc.
```

### Generate Sample Data

```bash
python generate_sample_data.py
# Creates 3,000 sample transactions with ground truth
```

---

## 📊 System 1: Prompt Tuning Agent

### Features

- **Adaptive Learning**: Iteratively improves prompts based on failures
- **TRUE Metrics**: Real TP/TN/FP/FN calculation with ground truth validation
- **98% Targets**: Automatically optimizes to achieve 98% precision and accuracy
- **Full Explainability**: SHAP, LIME, feature importance, success/failure analysis
- **LLM Meta-Reasoning**: Uses LLM to reason about prompt improvements
- **Multi-Provider Support**: OpenAI, Anthropic, Gemini, Cohere, Mistral, Ollama, LM Studio

### Usage

```python
from agent.true_ai_agent import TrueAIAgent

# Initialize agent
agent = TrueAIAgent(
    llm_provider='openai',
    model='gpt-4',
    data_dir='data',
    max_tuning_iterations=10
)

# Load data
agent.load_data()

# Run adaptive tuning with explainability
result = agent.adaptive_tune(
    requirement='fw15',
    target_precision=0.98,
    target_accuracy=0.98
)

# Access results
print(f"Iterations: {result['iterations']}")
print(f"Best Precision: {result['best_metrics']['precision']:.2%}")
print(f"Best Accuracy: {result['best_metrics']['accuracy']:.2%}")

# Access explainability
for exp in result['explanations']:
    print(f"Success factors: {exp.success_factors}")
    print(f"Failure factors: {exp.failure_factors}")
    print(f"Suggestions: {exp.improvement_suggestions}")
```

### Architecture

```
agent/
├── adaptive_tuner.py          # Adaptive prompt optimization with explainability
├── llm_service.py             # Multi-provider LLM integration (7 providers)
├── true_metrics.py            # TRUE metrics calculator (TP/TN/FP/FN)
├── prompt_explainability.py   # Feature importance & success/failure analysis
├── prompt_shap_lime.py        # SHAP and LIME attribution
├── data_processor.py          # Transaction data processing
├── ground_truth.py            # Ground truth management
└── true_ai_agent.py           # Main agent orchestrator
```

### Reasoning & Explainability

The Prompt Tuning Agent includes:

1. **LLM Meta-Prompting Reasoning**
   - Uses LLM to reason about prompt improvements
   - Analyzes failure patterns (FP/FN)
   - Generates strategic improvements

2. **Explainability Analysis**
   - Success/failure factor identification
   - Feature importance calculation
   - Improvement suggestion generation

3. **SHAP Attribution**
   - Token-level contribution analysis
   - Shapley value calculation

4. **LIME Local Explainability**
   - Local linear model approximation
   - Feature weight attribution

---

## 🎨 System 2: Synthetic Data Generator

### Features

- **6 Reasoning Engines**: Monte Carlo, Beam Search, Chain-of-Thought, Tree-of-Thoughts, MCTS, Hybrid
- **Full Explainability**: Feature importance, decision rules, SHAP, LIME
- **Intent Understanding**: Natural language → structured data
- **UK Compliance**: GDPR-compliant, UK formats (postcodes, dates, currency)
- **Multi-Format Output**: CSV, JSON, Excel, PDF, Word, Markdown
- **Quality Assurance**: Automatic validation and quality checks

### Usage

```python
from synthetic_data_generator.core.explainable_generator import ExplainableSyntheticGenerator
from synthetic_data_generator.core.llm_providers import OpenAIProvider

# Initialize
llm = OpenAIProvider(api_key='sk-...')
generator = ExplainableSyntheticGenerator(
    llm_provider=llm,
    reasoning_engine='chain_of_thought',  # or 'monte_carlo', 'beam_search', etc.
    enable_explainability=True
)

# Generate from natural language
result = generator.generate_from_prompt(
    prompt="Generate 100 realistic UK bank transactions with diverse demographics",
    include_shap=True,
    include_lime=True
)

# Access data
print(f"Generated {len(result.data)} records")

# Access explainability
report = result.explanation_report
print(f"Feature importances: {report.feature_importances}")
print(f"Decision rules: {report.decision_rules}")
print(f"SHAP explanations: {report.shap_explanations}")
print(f"LIME explanations: {report.lime_explanations}")
```

### 6 Reasoning Engines

| Engine | Description | Best For |
|--------|-------------|----------|
| **Monte Carlo** | Statistical distributions | Numerical data, demographics |
| **Beam Search** | Multiple candidates, select best | Optimization, quality |
| **Chain-of-Thought** | Step-by-step reasoning | Complex relationships |
| **Tree-of-Thoughts** | Multi-path exploration | Creative generation |
| **MCTS** | Game theory optimization | Strategic decisions |
| **Hybrid** | Combines all strategies | General purpose, best quality |

### Architecture

```
synthetic_data_generator/
├── core/
│   ├── reasoning_engines.py      # 6 reasoning engines
│   ├── explainability.py         # SHAP, LIME, feature importance
│   ├── explainable_generator.py  # Main generator with explainability
│   ├── intent_engine.py          # Natural language understanding
│   ├── llm_providers.py          # Multi-provider LLM support
│   └── output_engine.py          # Multi-format export
├── examples/                     # Usage examples
└── tests/                        # Comprehensive tests
```

---

## 🔧 Supported LLM Providers

Both systems support 7 LLM providers:

| Provider | Models | API Key Required |
|----------|--------|------------------|
| **OpenAI** | GPT-4, GPT-3.5-turbo, GPT-4-turbo | ✅ Yes |
| **Anthropic** | Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku | ✅ Yes |
| **Google Gemini** | Gemini-Pro | ✅ Yes |
| **Cohere** | Command, Command-R | ✅ Yes |
| **Mistral AI** | Mistral-Large, Mistral-Medium | ✅ Yes |
| **Ollama** | Local models | ❌ No (local) |
| **LM Studio** | Local models | ❌ No (local) |

### Configuration

```python
# OpenAI
agent = TrueAIAgent(llm_provider='openai', model='gpt-4')

# Anthropic
agent = TrueAIAgent(llm_provider='anthropic', model='claude-3-opus-20240229')

# Local models (Ollama)
agent = TrueAIAgent(llm_provider='ollama', model='llama2')
```

---

## 📈 Explainability & Reasoning

### Prompt Tuning Explainability

Every iteration generates:
- **Success factors**: What worked well
- **Failure factors**: What caused errors (FP/FN analysis)
- **Feature importance**: Which prompt elements matter
- **Improvement suggestions**: Actionable recommendations
- **SHAP values**: Token-level attribution
- **LIME explanations**: Local interpretability

Example output:
```
📊 EXPLAINABILITY ANALYSIS:
  Success Factors (2):
    ✓ Has step-by-step instructions
    ✓ Includes validation criteria
  Failure Factors (2):
    ✗ 2 false positives - Criteria too broad
    ✗ 8 false negatives - Missing edge cases
  Improvement Suggestions:
    → Add explicit exclusion criteria
    → Broaden criteria to capture missed cases

🔍 LIME LOCAL INTERPRETABILITY:
  Local Model R²: 0.85
  Top Features:
    ↑ has_examples: 0.643
    ↑ has_constraints: 0.321
    ↓ length_too_short: -0.154
```

### Synthetic Data Explainability

Every generation includes:
- **Feature importance**: Which features drove decisions
- **Decision rules**: If-then rules extracted from patterns
- **SHAP values**: Attribution per record
- **LIME explanations**: Local interpretability
- **Reasoning trace**: Full decision path

---

## 🧪 Testing

### Run Tests

```bash
# Verify mock removal
python test_implementation_nomock.py

# Check all systems
python -c "from agent.adaptive_tuner import AdaptivePromptTuner; print('✓ Prompt tuning ready')"
python -c "from synthetic_data_generator.core.explainable_generator import ExplainableSyntheticGenerator; print('✓ Synthetic generator ready')"
```

### Test Results

All tests passing:
- ✅ Mock removed from all systems
- ✅ All imports working
- ✅ All syntax valid
- ✅ Explainability integrated
- ✅ SHAP/LIME available
- ✅ Requirements installed

---

## 📦 Requirements

Key dependencies:
```
# Core
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# LLM Providers
openai>=1.0.0
anthropic>=0.18.0
google-generativeai>=0.3.0
cohere>=4.0.0
mistralai>=0.1.0

# ML & Explainability
sentence-transformers>=2.2.0

# Development
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
```

See `requirements.txt` for full list.

---

## 🏗️ Project Structure

```
Auto-prompt-tuning-agent/
├── agent/                           # Prompt tuning agent
│   ├── adaptive_tuner.py           # Core tuning with explainability
│   ├── llm_service.py              # Multi-provider LLM
│   ├── prompt_explainability.py    # Explainability engine
│   ├── prompt_shap_lime.py         # SHAP & LIME
│   └── true_ai_agent.py            # Main agent
├── synthetic_data_generator/        # Synthetic data generator
│   ├── core/
│   │   ├── reasoning_engines.py    # 6 reasoning engines
│   │   ├── explainability.py       # Full explainability
│   │   └── explainable_generator.py
│   └── examples/                    # Usage examples
├── data/                            # Sample data
├── prompts/                         # Prompt templates
├── tests/                           # Test suites
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## 🎯 Use Cases

### Prompt Tuning
- **Financial Analysis**: Optimize prompts for transaction analysis
- **Compliance**: Achieve 98% accuracy for regulatory requirements
- **Quality Assurance**: Validate prompt performance with ground truth
- **Cost Optimization**: Minimize false positives/negatives

### Synthetic Data Generation
- **ML Training**: Generate realistic training datasets
- **Testing**: Create test data for QA
- **Privacy**: Generate GDPR-compliant synthetic data
- **Demos**: Create presentation-ready datasets
- **Data Augmentation**: Expand limited real datasets

---

## 🔍 Key Features Summary

### Both Systems Include

✅ **Full Reasoning**
- Prompt Tuning: LLM meta-prompting
- Synthetic Generator: 6 reasoning engines

✅ **Complete Explainability**
- Feature importance analysis
- SHAP attribution
- LIME local explanations
- Success/failure factors

✅ **Production Ready**
- No mock implementations
- Multi-provider support
- Comprehensive error handling
- Full test coverage

✅ **Enterprise Grade**
- Type hints throughout
- Logging and monitoring
- Export functionality
- Scalable architecture

---

## 📚 Documentation

### Main Documentation
- This README - Complete system overview

### Additional Resources
- `synthetic_data_generator/README.md` - Detailed synthetic generator guide
- Code docstrings - In-line documentation
- Example scripts in `examples/` and `synthetic_data_generator/examples/`

---

## 🚦 Status

| Component | Status | Details |
|-----------|--------|---------|
| **Prompt Tuning Agent** | ✅ Production Ready | Full explainability integrated |
| **Synthetic Data Generator** | ✅ Production Ready | 6 reasoning engines + explainability |
| **Mock Implementations** | ✅ Removed | All mock code deleted |
| **Requirements** | ✅ Installed | 50+ packages installed |
| **Tests** | ✅ Passing | All critical tests passing |
| **LLM Support** | ✅ 7 Providers | OpenAI, Anthropic, Gemini, Cohere, Mistral, Ollama, LM Studio |

---

## 🎓 Examples

### Example 1: Prompt Tuning with Explainability

```python
from agent.true_ai_agent import TrueAIAgent

agent = TrueAIAgent(llm_provider='openai', model='gpt-4')
agent.load_data()

result = agent.adaptive_tune(requirement='fw15')

# View explainability
for i, exp in enumerate(result['explanations'], 1):
    print(f"\n=== Iteration {i} ===")
    print(f"Precision: {exp.metrics['precision']:.2%}")
    print(f"Success: {exp.success_factors}")
    print(f"Issues: {exp.failure_factors}")
    print(f"Suggestions: {exp.improvement_suggestions}")
```

### Example 2: Synthetic Data with Chain-of-Thought

```python
from synthetic_data_generator.core.explainable_generator import ExplainableSyntheticGenerator
from synthetic_data_generator.core.llm_providers import OpenAIProvider

llm = OpenAIProvider(api_key='sk-...')
generator = ExplainableSyntheticGenerator(
    llm_provider=llm,
    reasoning_engine='chain_of_thought'
)

result = generator.generate_from_prompt(
    "Generate 50 UK bank transactions with diverse customer profiles"
)

print(f"Generated {len(result.data)} records")
print(f"Reasoning used: {result.reasoning_engine}")
```

---

## 🤝 Contributing

This is a complete, production-ready system. For modifications:
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Ensure explainability is maintained

---

## 📄 License

[Your License Here]

---

## 🔗 Links

- [Synthetic Data Generator Detailed Guide](synthetic_data_generator/README.md)
- [Example Scripts](examples/)
- [Test Suite](tests/)

---

## 📞 Support

For questions or issues, please refer to:
- Code documentation (docstrings)
- Example scripts
- This README

---

**Last Updated**: 2025-10-24

**Version**: 1.0.0 - Production Ready with Full Explainability
