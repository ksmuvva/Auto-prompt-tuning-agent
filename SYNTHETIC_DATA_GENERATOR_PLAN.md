# Advanced Synthetic Data Generator - Architecture Plan

**Project:** World-Class Synthetic Data Generator for Hackathon
**Version:** 1.0
**Status:** Planning Phase

---

## Executive Summary

A sophisticated, LLM-agnostic synthetic data generator that understands user intent at a micro level and generates real-world quality data in multiple formats (CSV, PDF, Word, JSON, XLS, MD) using advanced reasoning techniques (Monte Carlo, Beam Reasoning, Chain-of-Thought).

**Key Differentiators:**
- 🧠 **Intent Understanding**: Deep comprehension of what user wants and why
- 🎯 **Context Awareness**: Understands how data will be used (training ML, testing, demos)
- 🔬 **Multi-Reasoning**: Monte Carlo, Beam, CoT, ToT for diverse, realistic data
- 🇬🇧 **UK Standards**: GDPR-compliant, UK formats (postcodes, dates, phone numbers)
- 🤖 **LLM Agnostic**: Works with any LLM (OpenAI, Anthropic, Gemini, local models)
- 📊 **Multi-Format**: CSV, PDF, Word, JSON, XLS, Markdown
- 🔍 **Ambiguity Detection**: Queries user when prompt is unclear
- 🎨 **Pattern Learning**: Learns and applies user-defined patterns

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface (NLP-Enabled)               │
│  "Generate 1000 customer records for UK e-commerce testing" │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Intent & Context Understanding Engine           │
│  • What: Customer records                                   │
│  • How Many: 1000                                           │
│  • Purpose: E-commerce testing                              │
│  • Geography: UK                                            │
│  • Constraints: Realistic, diverse, UK-compliant            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Ambiguity Detection System                   │
│  • Detect unclear requirements                              │
│  • Generate clarifying questions                            │
│  • Interactive dialogue with user                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Reasoning Engine Selector                 │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐ │
│  │ Monte Carlo  │ Beam Search  │ Chain-of-    │ Tree-of- │ │
│  │ Sampling     │ Reasoning    │ Thought      │ Thoughts │ │
│  └──────────────┴──────────────┴──────────────┴──────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Generation Engine                      │
│  • Schema inference from prompt                             │
│  • Constraint satisfaction                                  │
│  • Diversity optimization                                   │
│  • Real-world pattern matching                              │
│  • UK standards compliance                                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Quality Assurance Layer                   │
│  • Validation checks                                        │
│  • Consistency verification                                 │
│  • UK standards verification                                │
│  • Statistical distribution analysis                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Format Output Engine                      │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┐   │
│  │  CSV   │  JSON  │  PDF   │  Word  │  XLS   │   MD   │   │
│  └────────┴────────┴────────┴────────┴────────┴────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Intent & Context Understanding Engine

**Purpose:** Understand WHAT user wants and WHY at a micro level

**Capabilities:**
- Parse natural language prompts
- Extract key entities (data type, count, purpose, constraints)
- Infer implicit requirements
- Understand use case (training, testing, demo)
- Detect domain (finance, healthcare, e-commerce, etc.)

**Example Analysis:**
```
User Prompt: "Generate 1000 customer records for UK e-commerce testing"

Extracted Intent:
  - Entity: Customer records
  - Count: 1000
  - Geography: UK
  - Purpose: Testing
  - Domain: E-commerce
  - Implicit Requirements:
    * Realistic names, addresses, emails
    * UK postcodes, phone numbers
    * Diverse demographics
    * Valid email formats
    * Age distribution realistic for online shopping
    * GDPR compliance (no real personal data)
```

**Technology:**
- LLM-powered NLP with structured output
- Entity extraction
- Semantic similarity matching
- Domain knowledge graphs

---

### 2. Ambiguity Detection & Clarification System

**Purpose:** Detect unclear prompts and interactively clarify

**Detection Triggers:**
- Missing critical information
- Conflicting requirements
- Ambiguous terms
- Multiple possible interpretations

**Example Interaction:**
```
User: "Generate patient data"

System Detects Ambiguities:
  ❓ How many records?
  ❓ What fields (demographics only, medical history, lab results)?
  ❓ Geography (UK, US, other)?
  ❓ Purpose (training ML model, testing app, research)?
  ❓ Sensitivity level (synthetic only, realistic, anonymized real)?

System Asks:
  "I can generate patient data. To create the most useful dataset:

   1. How many patient records? (e.g., 100, 1000, 10000)
   2. What fields do you need?
      a) Demographics only (name, age, gender, address)
      b) Medical history (diagnoses, medications, allergies)
      c) Lab results (blood work, imaging, vitals)
      d) All of the above
   3. Geography: UK (NHS standards) or other?
   4. Purpose: ML training, app testing, or research?

   Please specify, or I'll use sensible defaults."
```

---

### 3. Multi-Reasoning Engine System

**Purpose:** Use multiple reasoning strategies for diverse, realistic data

#### 3.1 Monte Carlo Sampling
**Use Case:** Realistic statistical distributions

```python
# Example: Generate realistic age distribution for UK e-commerce customers
# Uses actual UK demographics + online shopping patterns

Ages generated via Monte Carlo:
  - Draw from multi-modal distribution
  - Peak 1: 25-34 (millennials, high online shopping)
  - Peak 2: 45-54 (Gen X, established buyers)
  - Long tail: 18-24, 35-44, 55-65+
  - Realistic variance and outliers
```

**Implementation:**
- Probabilistic sampling
- Multi-variate distributions
- Correlation preservation (age vs income, location vs purchasing power)

#### 3.2 Beam Reasoning
**Use Case:** Generate multiple candidate records, select best

```python
# Example: Generate one customer record

Beam Search (width=5):
  Candidate 1: "Emma Thompson, 32, London, emma.t@gmail.com"
  Candidate 2: "Mohammed Ali, 45, Birmingham, m.ali@yahoo.co.uk"
  Candidate 3: "Sophie Chen, 28, Manchester, sophie.chen@outlook.com"
  Candidate 4: "James O'Brien, 51, Edinburgh, james.obrien@btinternet.com"
  Candidate 5: "Priya Patel, 34, Leicester, priya.p@hotmail.co.uk"

Selection Criteria:
  ✓ Name diversity (ethnicity, gender)
  ✓ Age diversity
  ✓ Geographic spread
  ✓ Email provider variety
  ✓ Realistic patterns

Selected: All 5 (ensures diversity across 1000 records)
```

#### 3.3 Chain-of-Thought Reasoning
**Use Case:** Complex, interdependent data generation

```python
# Example: Generate realistic e-commerce transaction

CoT Steps:
  1. Customer demographics → Determine likely product preferences
     "Emma Thompson, 32, London → Tech-savvy, urban professional"

  2. Product preferences → Select products
     "Likely products: Smart home devices, organic groceries, fitness gear"

  3. Products → Realistic quantities & prices
     "1x Smart Thermostat £189.99, 2x Yoga Mat £24.99 each"

  4. Quantities + Season → Discount applicability
     "January sale: 15% off smart home → £161.49"

  5. All above → Payment method, delivery address
     "Card payment, next-day delivery to London flat"

  6. Consistency check
     "Age 32 → Credit card OK, Urban → Fast delivery available"
```

#### 3.4 Tree-of-Thoughts
**Use Case:** Explore multiple data generation paths

```python
# Example: Generate medical diagnosis record

ToT Exploration:
  Root: Patient with chest pain

  Branch 1: Cardiac cause
    → Myocardial infarction → Lab: Elevated troponin
    → Angina → Lab: Normal troponin, ECG changes
    → Pericarditis → Lab: Elevated inflammatory markers

  Branch 2: Respiratory cause
    → Pneumonia → Lab: WBC elevated, chest X-ray
    → Pulmonary embolism → Lab: D-dimer elevated

  Branch 3: Musculoskeletal
    → Costochondritis → Lab: All normal, exam findings

  Branch 4: GI cause
    → GERD → Lab: Normal, response to PPI

Evaluation: Select most statistically likely + ensure dataset diversity
Result: 60% cardiac, 20% respiratory, 10% MSK, 10% GI (realistic distribution)
```

---

### 4. UK Standards Compliance Layer

**Purpose:** Ensure all data follows UK standards and is GDPR-compliant

**UK Standards Implemented:**

#### 4.1 Dates & Times
```python
Format: DD/MM/YYYY (UK standard)
Examples:
  ✓ 23/10/2025
  ✗ 10/23/2025 (US format)

Time: 24-hour format
  ✓ 14:30
  ✗ 2:30 PM
```

#### 4.2 Addresses & Postcodes
```python
UK Postcode Format: Outward Code + Inward Code
Pattern: AN NAA, ANN NAA, AAN NAA, AANN NAA, ANA NAA, AANA NAA

Valid Examples:
  ✓ SW1A 1AA (Westminster)
  ✓ M1 1AE (Manchester)
  ✓ EH1 1YZ (Edinburgh)
  ✓ CF10 1BH (Cardiff)

Address Format:
  [Number] [Street Name]
  [City]
  [Postcode]

Example:
  10 Downing Street
  London
  SW1A 2AA
```

#### 4.3 Phone Numbers
```python
UK Phone Format: +44 or 0

Mobile: +44 7XXX XXXXXX or 07XXX XXXXXX
Landline London: +44 20 XXXX XXXX or 020 XXXX XXXX
Other cities: +44 1XXX XXXXXX or 01XXX XXXXXX

Examples:
  ✓ +44 7700 900123 (mobile)
  ✓ 07700 900123 (mobile)
  ✓ +44 20 7123 4567 (London landline)
  ✓ 0161 123 4567 (Manchester landline)
```

#### 4.4 Currency
```python
Format: £X,XXX.XX

Examples:
  ✓ £1,234.56
  ✓ £10.00
  ✗ $1,234.56 (US)
  ✗ €1.234,56 (EU)
```

#### 4.5 Names (UK Diversity)
```python
Ethnicity Distribution (realistic UK demographics):
  - White British: ~80%
  - Asian/Asian British: ~9%
  - Black/African/Caribbean: ~4%
  - Mixed: ~3%
  - Other: ~4%

Common UK Names by Origin:
  English: Oliver, George, Emma, Sophie
  Scottish: Callum, Isla, Finlay, Eilidh
  Welsh: Dylan, Cerys, Owain, Bethan
  Irish: Cian, Aoife, Niamh, Conor
  Pakistani: Mohammed, Aisha, Hassan, Fatima
  Indian: Arjun, Priya, Rahul, Ananya
  Polish: Jakub, Zofia, Jan, Maria
  Chinese: Wei, Mei, Jun, Li
```

#### 4.6 GDPR Compliance
```python
Synthetic Data Requirements:
  ✓ No real personal data
  ✓ Clearly marked as synthetic
  ✓ No traceable to real individuals
  ✓ Privacy-preserving

Metadata:
  {
    "data_type": "synthetic",
    "generated_date": "2025-10-23",
    "purpose": "testing",
    "gdpr_compliant": true,
    "contains_pii": false
  }
```

---

### 5. Pattern Recognition & Application System

**Purpose:** Learn from user examples and apply patterns

**Capabilities:**

#### 5.1 Pattern Learning
```python
User provides examples:

Example 1:
  Email: john.smith@company.co.uk
  Pattern: [firstname].[lastname]@company.co.uk

Example 2:
  Email: j.smith@company.co.uk
  Pattern: [first_initial].[lastname]@company.co.uk

Example 3:
  Email: jsmith@company.co.uk
  Pattern: [first_initial][lastname]@company.co.uk

System Learns:
  - Company domain: company.co.uk
  - Email formats: firstname.lastname OR f.lastname OR flastname
  - Distribution: 50% full name, 30% initial.last, 20% initiallast
```

#### 5.2 Pattern Application
```python
Generate 1000 emails following learned pattern:

Generated:
  1. emma.thompson@company.co.uk (full name format)
  2. m.ali@company.co.uk (initial.last format)
  3. schen@company.co.uk (initiallast format)
  4. james.obrien@company.co.uk (full name with apostrophe handled)
  ...
```

#### 5.3 Custom Patterns
```python
User Input:
  "Generate employee IDs following pattern: EMP-[DEPT]-[YYYY]-[NNNN]
   Where DEPT is IT, HR, FIN, or OPS
   YYYY is hire year
   NNNN is sequential number"

System Generates:
  EMP-IT-2023-0001
  EMP-HR-2024-0023
  EMP-FIN-2023-0156
  EMP-OPS-2024-0089
  ...
```

---

### 6. Multi-Format Output Engine

**Purpose:** Output data in requested format with proper formatting

#### 6.1 CSV
```csv
customer_id,name,email,age,postcode,join_date,total_spent
CUST001,Emma Thompson,emma.t@gmail.com,32,SW1A 1AA,15/03/2024,£1234.56
CUST002,Mohammed Ali,m.ali@yahoo.co.uk,45,M1 1AE,22/01/2024,£2456.78
```

**Features:**
- Proper escaping
- UTF-8 encoding
- UK date format
- Header row

#### 6.2 JSON
```json
[
  {
    "customer_id": "CUST001",
    "name": "Emma Thompson",
    "email": "emma.t@gmail.com",
    "age": 32,
    "address": {
      "street": "10 Downing Street",
      "city": "London",
      "postcode": "SW1A 1AA"
    },
    "join_date": "2024-03-15",
    "total_spent": 1234.56
  }
]
```

**Features:**
- Pretty-printed
- Nested structures
- ISO dates for APIs
- Type preservation

#### 6.3 PDF
```
┌─────────────────────────────────────────────────────┐
│           UK E-Commerce Customer Report             │
│                Generated: 23/10/2025                │
└─────────────────────────────────────────────────────┘

Customer ID: CUST001
Name: Emma Thompson
Email: emma.t@gmail.com
Age: 32
Address: 10 Downing Street, London, SW1A 1AA
Join Date: 15/03/2024
Total Spent: £1,234.56
─────────────────────────────────────────────────────
```

**Features:**
- Professional formatting
- UK standards
- Headers/footers
- Page numbers

#### 6.4 Word (.docx)
```
Similar to PDF but in Microsoft Word format
- Tables for structured data
- Headers and footers
- UK date formats
- Proper styling
```

#### 6.5 Excel (.xlsx)
```
Sheet 1: Customer Data
┌────────────┬─────────────────┬─────────────────────────┬─────┬───────────┐
│ Customer ID│ Name            │ Email                   │ Age │ Postcode  │
├────────────┼─────────────────┼─────────────────────────┼─────┼───────────┤
│ CUST001    │ Emma Thompson   │ emma.t@gmail.com        │ 32  │ SW1A 1AA  │
│ CUST002    │ Mohammed Ali    │ m.ali@yahoo.co.uk       │ 45  │ M1 1AE    │
└────────────┴─────────────────┴─────────────────────────┴─────┴───────────┘

Sheet 2: Summary Statistics
```

**Features:**
- Multiple sheets
- Formulas
- Formatting
- Charts (optional)

#### 6.6 Markdown
```markdown
# UK E-Commerce Customer Data

Generated: 23/10/2025

## Customer Records

| Customer ID | Name | Email | Age | Postcode |
|-------------|------|-------|-----|----------|
| CUST001 | Emma Thompson | emma.t@gmail.com | 32 | SW1A 1AA |
| CUST002 | Mohammed Ali | m.ali@yahoo.co.uk | 45 | M1 1AE |

## Summary Statistics
- Total Customers: 1000
- Average Age: 38.5
- Date Range: Jan 2024 - Oct 2025
```

---

### 7. LLM Agnostic Architecture

**Purpose:** Work with any LLM provider

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Cohere
- Mistral
- Local models (Ollama, LM Studio)

**Provider Interface:**
```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, schema: dict) -> dict:
        pass

# Usage
llm = LLMFactory.create(provider="openai", model="gpt-4")
result = llm.generate("Generate customer name")
```

---

## CLI Interface Design

### Interactive Mode
```bash
$ python synthetic_data_cli.py

╔════════════════════════════════════════════════════════════╗
║     Advanced Synthetic Data Generator                      ║
║     World-Class, LLM-Agnostic, UK-Compliant                ║
╚════════════════════════════════════════════════════════════╝

Enter your data requirements (or 'help' for examples):
> Generate 1000 UK customer records for e-commerce testing

🧠 Analyzing your request...

Intent Detected:
  ✓ Data Type: Customer records
  ✓ Count: 1000
  ✓ Geography: UK
  ✓ Purpose: E-commerce testing
  ✓ Format: Not specified

❓ Questions to clarify:

1. Output format?
   a) CSV (recommended for tabular data)
   b) JSON (recommended for APIs)
   c) Excel (.xlsx)
   d) PDF Report
   e) Word Document
   f) Markdown

2. Fields to include?
   a) Basic (name, email, age)
   b) Full (+ address, phone, demographics)
   c) Extended (+ purchase history, preferences)
   d) Custom (you specify)

3. LLM Provider? (default: openai)
   a) OpenAI (GPT-4)
   b) Anthropic (Claude)
   c) Google (Gemini)
   d) Local (Ollama)

Press Enter for defaults or specify: _
```

### Command Mode
```bash
# Quick generation
python synthetic_data_cli.py \
  --prompt "Generate 1000 UK customer records" \
  --format csv \
  --output customers.csv \
  --llm openai \
  --reasoning monte-carlo

# With pattern
python synthetic_data_cli.py \
  --prompt "Generate employee data" \
  --pattern examples/employee_pattern.json \
  --format xlsx \
  --output employees.xlsx

# Multiple formats
python synthetic_data_cli.py \
  --prompt "Generate patient records" \
  --format csv,json,pdf \
  --output-dir ./generated_data/

# Advanced
python synthetic_data_cli.py \
  --prompt "Generate UK NHS patient data for ML training" \
  --count 10000 \
  --format csv \
  --reasoning beam-search \
  --uk-standards strict \
  --gdpr-compliant \
  --validate \
  --output patients.csv
```

---

## Implementation Phases

### Phase 1: Core Engine (Week 1)
- ✅ Intent understanding engine
- ✅ Basic Monte Carlo sampling
- ✅ CSV output
- ✅ UK standards (dates, postcodes, phones)
- ✅ CLI interface (basic)

### Phase 2: Advanced Reasoning (Week 2)
- ✅ Beam reasoning
- ✅ Chain-of-Thought
- ✅ Tree-of-Thoughts
- ✅ Ambiguity detection
- ✅ Pattern learning

### Phase 3: Multi-Format Output (Week 3)
- ✅ JSON output
- ✅ PDF generation
- ✅ Word (.docx) generation
- ✅ Excel (.xlsx) generation
- ✅ Markdown output

### Phase 4: Advanced Features (Week 4)
- ✅ LLM agnostic architecture
- ✅ Context-aware generation
- ✅ Quality assurance layer
- ✅ Interactive clarification
- ✅ Documentation & examples

---

## Success Metrics (Hackathon)

1. **Functionality** (30 points)
   - ✓ Multiple formats supported
   - ✓ Multiple reasoning engines
   - ✓ UK standards compliance
   - ✓ LLM agnostic

2. **Innovation** (25 points)
   - ✓ Intent understanding at micro level
   - ✓ Multi-reasoning approach
   - ✓ Ambiguity handling
   - ✓ Pattern learning

3. **Real-World Applicability** (25 points)
   - ✓ Production-ready quality
   - ✓ GDPR compliant
   - ✓ Realistic data distributions
   - ✓ Multiple use cases (training, testing, demo)

4. **User Experience** (20 points)
   - ✓ Natural language interface
   - ✓ Interactive clarification
   - ✓ Clear documentation
   - ✓ Example-driven

---

## Technology Stack

**Core:**
- Python 3.11+
- LLM integration (OpenAI, Anthropic, etc.)
- Pandas (data manipulation)
- NumPy (statistical operations)

**Output Formats:**
- CSV: Built-in
- JSON: Built-in
- PDF: ReportLab
- Word: python-docx
- Excel: openpyxl
- Markdown: Built-in

**UK Standards:**
- UK postcode validation: UK Postcode Utils
- UK phone numbers: phonenumbers library
- GDPR compliance: Custom validation

**NLP & Reasoning:**
- LangChain (for multi-reasoning)
- Pydantic (structured outputs)
- Jinja2 (templating)

---

## Next Steps

1. **Review & Approve Architecture** ✓ (You are here)
2. **Start Implementation** (Phase 1)
3. **Iterative Development** (Phases 2-4)
4. **Testing & Refinement**
5. **Hackathon Preparation**
6. **Demo & Documentation**

---

**Status:** Ready for implementation approval
**Estimated Timeline:** 4 weeks to hackathon-ready
**Complexity:** High (Advanced, sophisticated system)
**Innovation Level:** World-class

Ready to build? 🚀
