# Advanced Synthetic Data Generator

** LLM-Agnostic, UK-Compliant Synthetic Data Generation**

A sophisticated synthetic data generator that understands user intent at a micro level and generates real-world quality data in multiple formats using advanced reasoning techniques.

## 🎯 Key Features

### 🧠 Intent Understanding
- **Deep Comprehension**: Understands WHAT you want and WHY
- **Context Awareness**: Knows how data will be used (training, testing, demos)
- **Implicit Requirements**: Infers unstated needs

### 🔬 Multi-Reasoning Engines
- **Monte Carlo Sampling**: Realistic statistical distributions
- **Beam Search**: Multiple candidates, select best
- **Chain-of-Thought**: Step-by-step reasoning for complex data
- **Tree-of-Thoughts**: Explores multiple generation paths
- **MCTS (Monte Carlo Tree Search)**: Exploration-exploitation balance for optimal generation
- **Hybrid Reasoning**: Combines all strategies with adaptive weighting

### 🇬🇧 UK Standards Compliance
- **GDPR Compliant**: Synthetic data, no real PII
- **UK Formats**: Postcodes, dates (DD/MM/YYYY), phone numbers
- **Diverse Demographics**: Realistic UK population representation
- **Currency**: Proper £ formatting

### 🤖 LLM Agnostic
Works with any LLM provider:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Mock (for testing, no API key needed)

### 📊 Multi-Format Output
- CSV (tabular data)
- JSON (APIs, structured data)
- Excel (.xlsx, with formulas)
- PDF (professional reports)
- Word (.docx, formatted documents)
- Markdown (documentation)

### 🔍 Advanced Features
- **Ambiguity Detection**: Asks clarifying questions
- **Pattern Learning**: Learns from your examples
- **Quality Assurance**: Validates generated data
- **Batch Export**: Export to multiple formats at once
- **Explainability**: Comprehensive AI/ML explainability with SHAP, LIME, feature importance, and decision rules

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to the synthetic data generator
cd synthetic_data_generator

# Install dependencies
pip install -r requirements_synthetic.txt
```

### Basic Usage (Interactive CLI)

```bash
# Start the natural language CLI
python -m cli.nlp_interface

# Or directly
python cli/nlp_interface.py
```

#### Example Session:

```
You: Generate 1000 UK customer records for e-commerce testing

🧠 Analyzing your request...

📋 Intent Detected:
  • Data Type: customer records
  • Count: 1000
  • Geography: UK
  • Purpose: testing
  • Domain: e-commerce
  • Format: CSV (default)

📐 Generating schema...
  Fields: customer_id, name, email, age, postcode, phone

⚙️  Using monte_carlo reasoning engine...
🎲 Generating 1000 records...
🇬🇧 Applying UK standards...
💾 Exporting to CSV...

✅ Success! Data exported to: generated_data_customer.csv
```

---

## 📖 Usage Examples

### Example 1: Basic Generation

```python
from core.llm_providers import LLMFactory
from core.intent_engine import IntentEngine
from core.reasoning_engines import ReasoningEngineFactory
from core.output_engine import OutputEngine

# Initialize LLM (using mock for demo)
llm = LLMFactory.create('mock')

# Parse user intent
intent_engine = IntentEngine(llm)
intent = intent_engine.parse_intent("Generate 100 UK customer records")

# Get schema
schema = intent_engine.get_schema_suggestion(intent)

# Generate data using Monte Carlo
engine = ReasoningEngineFactory.create('monte_carlo', llm)
results = engine.generate(intent, schema, count=100)

# Extract data
data = [result.data for result in results]

# Export to CSV
output_engine = OutputEngine()
output_engine.export(data, 'customers.csv', 'csv')
```

### Example 2: Use Real LLM (OpenAI)

```python
from core.llm_providers import LLMFactory

# Initialize with OpenAI (requires API key in environment)
llm = LLMFactory.create('openai', model='gpt-4-turbo-preview')

# Rest is the same...
```

### Example 3: Multi-Format Export

```python
from core.output_engine import BatchOutputEngine

batch = BatchOutputEngine()

# Export to multiple formats at once
results = batch.export_multiple(
    data=data,
    output_dir='./output',
    base_name='customers',
    formats=['csv', 'json', 'excel', 'markdown'],
    metadata={'title': 'Customer Data', 'geography': 'UK'}
)

# Output:
# ✓ Exported to ./output/customers.csv
# ✓ Exported to ./output/customers.json
# ✓ Exported to ./output/customers.xlsx
# ✓ Exported to ./output/customers.md
```

### Example 4: UK Standards

```python
from core.uk_standards import UKStandardsGenerator

uk_gen = UKStandardsGenerator()

# Generate UK-compliant data
postcode = uk_gen.generate_postcode('London')  # SW1A 1AA
phone = uk_gen.generate_phone('mobile')        # 07700 900123
first, last = uk_gen.generate_name()           # Emma Thompson
email = uk_gen.generate_email(first, last)     # emma.thompson@gmail.com
address = uk_gen.generate_address('London')    # Full UK address

# Format UK standards
date = uk_gen.format_date_uk(datetime.now())   # 23/10/2025
currency = uk_gen.format_currency(1234.56)     # £1,234.56
```

### Example 5: Pattern Learning

```python
from core.pattern_learner import PatternLearner

learner = PatternLearner(llm)

# Learn from examples
examples = [
    'john.smith@company.co.uk',
    'jane.doe@company.co.uk'
]

pattern = learner.learn_from_examples(examples, 'email')

# Apply pattern to generate new values
new_emails = learner.apply_pattern('email', count=100)
```

### Example 6: Different Reasoning Engines

```python
# Monte Carlo - Best for statistical distributions
monte_carlo = ReasoningEngineFactory.create('monte_carlo', llm)
results_mc = monte_carlo.generate(intent, schema, count=100)

# Beam Search - Best for diversity and quality
beam = ReasoningEngineFactory.create('beam_search', llm, beam_width=5)
results_beam = beam.generate(intent, schema, count=100)

# Chain-of-Thought - Best for complex, interdependent data
cot = ReasoningEngineFactory.create('chain_of_thought', llm)
results_cot = cot.generate(intent, schema, count=100)

# Tree-of-Thoughts - Best for exploring multiple scenarios
tot = ReasoningEngineFactory.create('tree_of_thoughts', llm, num_branches=3)
results_tot = tot.generate(intent, schema, count=100)

# MCTS - Best for exploration-exploitation balance
mcts = ReasoningEngineFactory.create('mcts', llm, num_simulations=100)
results_mcts = mcts.generate(intent, schema, count=100)

# Hybrid - Combines all strategies with adaptive weighting
hybrid = ReasoningEngineFactory.create('hybrid', llm, adaptive=True)
results_hybrid = hybrid.generate(intent, schema, count=100)
```

### Example 7: Explainability Features

```python
from core.explainable_generator import ExplainableSyntheticGenerator, ExplainabilityDashboard

# Create explainable generator
generator = ExplainableSyntheticGenerator(
    llm_provider=llm,
    reasoning_engine='hybrid',
    enable_explainability=True
)

# Generate data with explanations
result = generator.generate_from_prompt(
    "Generate 100 customer records",
    include_shap=True,
    include_lime=True,
    export_explanation=True,
    output_dir='./output'
)

# Display explainability report
ExplainabilityDashboard.print_report(result)

# Get feature importance summary
print(generator.get_feature_importance_summary(result, top_k=5))

# Get decision rules summary
print(generator.get_decision_rules_summary(result, top_k=5))

# Export data and explanations
generator.export_data(result, './output/customers.csv', format='csv')
# Explanation reports automatically exported to:
# - ./output/explainability_report.json
# - ./output/explainability_report.md
```

---

## 🏗️ Architecture

```
synthetic_data_generator/
├── core/
│   ├── llm_providers.py          # LLM-agnostic provider architecture
│   ├── intent_engine.py          # Intent & context understanding
│   ├── ambiguity_detector.py     # Ambiguity detection
│   ├── reasoning_engines.py      # Multi-reasoning (MC, Beam, CoT, ToT, MCTS, Hybrid)
│   ├── explainability.py         # 🆕 Feature importance, SHAP, LIME, decision rules
│   ├── explainable_generator.py  # 🆕 Explainability-aware data generator
│   ├── uk_standards.py           # UK standards compliance
│   ├── pattern_learner.py        # Pattern recognition
│   └── output_engine.py          # Multi-format output
├── cli/
│   └── nlp_interface.py          # Natural language CLI
├── tests/
│   ├── test_synthetic_generator.py
│   └── test_explainability.py    # 🆕 Comprehensive explainability tests
├── examples/
│   ├── basic_usage.py
│   ├── explainability_demo.py    # 🆕 Explainability demonstrations
│   └── hybrid_reasoning_demo.py  # 🆕 Hybrid reasoning demonstrations
└── requirements_synthetic.txt
```

---

## 🧪 Running Tests

```bash
# Run comprehensive tests
cd synthetic_data_generator

# Test core functionality
python tests/test_synthetic_generator.py

# Test explainability features (25 tests)
python tests/test_explainability.py

# Run example demonstrations
python examples/basic_usage.py
python examples/explainability_demo.py
python examples/hybrid_reasoning_demo.py
```

### Test Coverage:
- ✓ LLM Providers (OpenAI, Anthropic, Gemini, Mock)
- ✓ Intent Engine (parsing, schema generation)
- ✓ Ambiguity Detection (clarification questions)
- ✓ Reasoning Engines (all 6 types: MC, Beam, CoT, ToT, MCTS, Hybrid)
- ✓ Explainability (Feature Importance, SHAP, LIME, Decision Rules)
- ✓ UK Standards (postcodes, phones, dates, currency)
- ✓ Pattern Learning (email, ID patterns)
- ✓ Output Engines (CSV, JSON, PDF, Word, Excel, Markdown)

---

## 📋 Natural Language Commands

The CLI understands natural language. Try:

```
Generate 1000 UK customer records
Create 500 patient records in JSON format
Make 100 employees with salaries in Excel
Generate banking transactions for testing
Create e-commerce orders for ML training
Generate healthcare patient data for UK NHS
```

You can specify:
- **Number**: "1000", "500 records"
- **Geography**: "UK", "US", "EU"
- **Format**: "CSV", "JSON", "Excel", "PDF"
- **Purpose**: "testing", "training", "demo"
- **Domain**: "e-commerce", "healthcare", "finance"

---

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="..."
```

### Provider Selection

```python
# In code
llm = LLMFactory.create('openai')      # OpenAI GPT-4
llm = LLMFactory.create('anthropic')   # Anthropic Claude
llm = LLMFactory.create('gemini')      # Google Gemini
llm = LLMFactory.create('mock')        # Mock (no API key)

# In CLI
You: use openai
You: use anthropic
You: use gemini
You: use mock
```

---

## 🎨 Customization

### Custom Schemas

```python
# Define your own schema
custom_schema = {
    'employee_id': 'string',
    'full_name': 'string',
    'email': 'email',
    'department': 'string',
    'salary': 'number',
    'hire_date': 'date',
    'uk_postcode': 'postcode'
}

# Generate with custom schema
engine.generate(intent, custom_schema, count=1000)
```

### Custom Patterns

```python
# Teach the system your patterns
learner.learn_from_examples(
    ['EMP-IT-2023-0001', 'EMP-HR-2024-0023'],
    'employee_id'
)

# Generate following your pattern
new_ids = learner.apply_pattern('employee_id', count=1000)
```

---

## 🌟 Advanced Features

### Ambiguity Resolution

When your request is unclear, the system asks clarifying questions:

```
You: Generate patient data

System: I have some questions to clarify your request:

1. How many patient records? (required)
   a) 100
   b) 1000 [DEFAULT]
   c) 10000
   d) Custom amount

2. What fields should be included? (required)
   a) Demographics only [DEFAULT]
   b) Medical history
   c) Lab results
   d) All of the above

3. Geography: UK (NHS standards) or other? (optional)
```

### GDPR Compliance

All generated data includes GDPR metadata:

```json
{
  "_metadata": {
    "data_type": "synthetic",
    "generated_date": "2025-10-23T10:30:00",
    "gdpr_compliant": true,
    "contains_pii": false,
    "purpose": "testing/development",
    "disclaimer": "This is synthetic data. Any resemblance to real persons is coincidental."
  }
}
```

---

## 📊 Output Format Examples

### CSV
```csv
customer_id,name,email,age,postcode
CUST001,Emma Thompson,emma.t@gmail.com,32,SW1A 1AA
CUST002,Mohammed Ali,m.ali@yahoo.co.uk,45,M1 1AE
```

### JSON
```json
{
  "metadata": {
    "generated_date": "2025-10-23",
    "record_count": 1000,
    "geography": "UK"
  },
  "data": [
    {
      "customer_id": "CUST001",
      "name": "Emma Thompson",
      "email": "emma.t@gmail.com"
    }
  ]
}
```

### Excel
- Sheet 1: Data (formatted table)
- Sheet 2: Metadata (properties)
- Styled headers, proper column widths

### PDF
- Professional formatting
- Title and metadata
- Paginated tables
- Headers and footers

---

## 🤝 Contributing

This is a hackathon-ready project. To extend:

1. **Add new LLM providers**: Extend `LLMProvider` in `llm_providers.py`
2. **Add new reasoning engines**: Extend `ReasoningEngine` in `reasoning_engines.py`
3. **Add new output formats**: Extend `OutputEngine` in `output_engine.py`
4. **Add new standards**: Extend `UKStandardsGenerator` for other countries

---

## 📄 License

This project is designed for hackathons and educational purposes.

---

## 🎯 Use Cases

- **ML Training**: Generate large datasets for training models
- **QA Testing**: Test applications with realistic data
- **Demos**: Create impressive demo data
- **Development**: Local development without production data
- **Research**: Academic research with synthetic datasets
- **Prototyping**: Quickly prototype data-driven applications

---

## 🔍 Explainability Features

### Comprehensive AI/ML Explainability

The generator provides world-class explainability for understanding how synthetic data is generated:

#### 1. Feature Importance Analysis
```python
# Shows which features are most important in generation
{
  "age": {
    "importance": 0.85,
    "type": "interactive",
    "explanation": "Age interacts with salary and job title"
  }
}
```

#### 2. Decision Rules Extraction
```python
# Extracts human-readable rules
IF age > 30 THEN salary ~ Normal(75000, 15000)
IF postcode.startswith('SW') THEN city = 'London'
```

#### 3. SHAP (SHapley Additive exPlanations)
```python
# Attribution for each feature in each record
Record #1:
  age: +15.2 (significantly above baseline)
  salary: -5.3 (below baseline)
  city: +2.1 (typical value)
```

#### 4. LIME (Local Interpretable Model-agnostic Explanations)
```python
# Local model for each record
Record #1 Local Model (fidelity: 0.92):
  • age=35 strongly influences this record (weight: 0.88)
  • salary=65000 moderately influences (weight: 0.55)
```

### Explainability Reports

Automatically generates:
- **JSON Reports**: Machine-readable explainability data
- **Markdown Reports**: Human-readable documentation
- **Interactive Dashboard**: Console-based visualization

```bash
# Run explainability demo
python examples/explainability_demo.py

# Generates:
# - Feature importance charts
# - Decision rule tables
# - SHAP value breakdowns
# - LIME interpretations
```

---

## 🏆 features

1. **Intent Understanding**: Goes beyond simple templates, understands WHY you need data
2. **Multi-Reasoning**: Six different AI reasoning approaches for maximum quality
3. **Explainability**: Industry-leading explainability with SHAP, LIME, feature importance, and decision rules
4. **Standards Compliance**: True UK compliance (GDPR, formats, demographics)
5. **LLM Agnostic**: Works with any LLM, not locked to one provider
6. **Multi-Format**: Seven different output formats
7. **Production Ready**: Comprehensive tests, error handling, documentation
8. **Natural Language**: Talk to it like a human, not a machine
9. **Adaptive Learning**: Hybrid engine learns which strategies work best

---

**Ready to generate world-class synthetic data? 🚀**

```bash
python cli/nlp_interface.py
```
