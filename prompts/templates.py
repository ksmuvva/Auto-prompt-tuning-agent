"""
Prompt Template System
Provides various prompt formats for testing and optimization
"""

from typing import Dict, Any, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplate:
    """Base class for prompt templates"""

    def __init__(
        self,
        name: str,
        template: str,
        description: str = "",
        variables: List[str] = None
    ):
        self.name = name
        self.template = template
        self.description = description
        self.variables = variables or []

    def format(self, **kwargs) -> str:
        """Format the template with provided variables"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable in template {self.name}: {e}")
            return self.template

    def __repr__(self):
        return f"PromptTemplate(name='{self.name}')"


class PromptTemplateLibrary:
    """Library of prompt templates for bank transaction analysis"""

    def __init__(self):
        self.templates = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize built-in prompt templates"""

        # Template 1: Direct and concise
        self.add_template(PromptTemplate(
            name="direct_concise",
            description="Direct, concise instructions with bullet points",
            template="""Analyze the following bank transaction data:

{data}

Tasks:
1. Identify all transactions above {threshold} GBP
2. Detect anomalies in transaction patterns
3. Provide a summary with specific transaction IDs and amounts

Format your response as:
HIGH-VALUE TRANSACTIONS:
- List each transaction

ANOMALIES DETECTED:
- List each anomaly with reasoning

SUMMARY:
- Statistics and key findings
""",
            variables=["data", "threshold"]
        ))

        # Template 2: Detailed step-by-step
        self.add_template(PromptTemplate(
            name="detailed_step_by_step",
            description="Detailed step-by-step analysis instructions",
            template="""You are a financial analyst reviewing bank transaction data.

TRANSACTION DATA:
{data}

ANALYSIS INSTRUCTIONS:

Step 1: High-Value Transaction Detection
- Review each transaction in the dataset
- Identify all transactions exceeding {threshold} GBP
- For each high-value transaction, note: transaction ID, amount, date, description

Step 2: Anomaly Detection
- Look for unusual patterns including:
  * Transactions significantly above normal spending
  * Multiple transactions in rapid succession
  * Unusual merchant categories
  * Foreign currency transactions with suspicious patterns
  * Round number amounts that might indicate structured transactions

Step 3: Risk Assessment
- Categorize each anomaly by risk level (Low/Medium/High)
- Provide reasoning for each classification

Step 4: Summary Report
- Total number of high-value transactions
- Total number of anomalies detected
- Recommended actions

Please provide a detailed analysis following these steps.
""",
            variables=["data", "threshold"]
        ))

        # Template 3: JSON output format
        self.add_template(PromptTemplate(
            name="json_structured",
            description="Requests structured JSON output",
            template="""Analyze this bank transaction data and return results in JSON format:

{data}

Requirements:
- Find transactions > {threshold} GBP
- Detect anomalies

Return JSON in this exact format:
{{
  "high_value_transactions": [
    {{
      "transaction_id": "...",
      "amount": 0.0,
      "date": "...",
      "description": "...",
      "currency": "GBP"
    }}
  ],
  "anomalies": [
    {{
      "transaction_id": "...",
      "anomaly_type": "...",
      "risk_level": "...",
      "reasoning": "..."
    }}
  ],
  "summary": {{
    "total_high_value": 0,
    "total_anomalies": 0,
    "total_amount_high_value": 0.0
  }}
}}
""",
            variables=["data", "threshold"]
        ))

        # Template 4: Role-based with context
        self.add_template(PromptTemplate(
            name="role_based_expert",
            description="Role-based prompt with expert persona",
            template="""You are an expert fraud detection analyst with 10+ years of experience in banking security.

CONTEXT:
You are reviewing transaction data for potential fraud, compliance violations, and unusual activity patterns.

THRESHOLD: Transactions above {threshold} GBP require special attention.

DATA:
{data}

ANALYSIS REQUIRED:
As an expert, apply your knowledge to:

1. HIGH-VALUE ANALYSIS
   - Identify transactions exceeding the threshold
   - Assess legitimacy based on patterns

2. FRAUD INDICATORS
   - Unusual transaction amounts
   - Suspicious timing patterns
   - Merchant category anomalies
   - Velocity checks (frequency)

3. EXPERT RECOMMENDATIONS
   - Which transactions warrant further investigation
   - Suggested next steps for each flagged item

Provide your expert analysis with confidence scores where applicable.
""",
            variables=["data", "threshold"]
        ))

        # Template 5: Few-shot learning
        self.add_template(PromptTemplate(
            name="few_shot_examples",
            description="Includes examples of desired output",
            template="""Analyze bank transactions to find high-value transactions (>{threshold} GBP) and anomalies.

EXAMPLE 1:
Transaction: £500 at Electronics Store
Analysis: High-value transaction above threshold. Normal purchase pattern.
Classification: High-value, No anomaly

EXAMPLE 2:
Transaction: £3,000 unusual merchant
Analysis: Significantly above average spending, unfamiliar merchant
Classification: High-value, Anomaly - Risk Level: High

Now analyze this data:
{data}

Provide analysis in the same format as examples above.
""",
            variables=["data", "threshold"]
        ))

        # Template 6: Chain of thought
        self.add_template(PromptTemplate(
            name="chain_of_thought",
            description="Encourages reasoning process",
            template="""Analyze this transaction data step by step, showing your reasoning:

{data}

Threshold: {threshold} GBP

Think through this carefully:

First, let me identify the high-value transactions...
[Work through the data]

Next, let me look for anomaly patterns...
[Consider various anomaly types]

Now, let me categorize my findings...
[Organize results]

Finally, here's my structured analysis:
[Present final results]

Please show your complete reasoning process.
""",
            variables=["data", "threshold"]
        ))

        # Template 7: Minimal instructions
        self.add_template(PromptTemplate(
            name="minimal",
            description="Minimal prompt to test LLM capabilities",
            template="""Find transactions > {threshold} GBP and anomalies:

{data}
""",
            variables=["data", "threshold"]
        ))

        # Template 8: Table-based output
        self.add_template(PromptTemplate(
            name="table_format",
            description="Requests output in table format",
            template="""Analyze transactions and format results as tables:

DATA:
{data}

Create two tables:

TABLE 1: HIGH-VALUE TRANSACTIONS (>{threshold} GBP)
| ID | Date | Amount | Description | Risk |
|----+------+--------+-------------+------|

TABLE 2: ANOMALIES DETECTED
| ID | Type | Severity | Reason |
|----+------+----------+--------|

Provide tables with all findings.
""",
            variables=["data", "threshold"]
        ))

        # === FW-SPECIFIC TEMPLATES ===
        
        # FW15: High-Value Transactions
        self.add_template(PromptTemplate(
            name="fw15_high_value",
            description="FW15: Identify transactions exceeding £{threshold}",
            template="""You are a financial analyst specializing in high-value transaction detection.

TASK: Identify ALL transactions exceeding £{threshold}

DATA:
{data}

REQUIREMENTS:
1. Precision: ≥98% (avoid false positives)
2. Accuracy: ≥98% (correct identification)
3. Include ALL transactions where amount > {threshold}

EXCLUSION CRITERIA:
- Do NOT include transactions AT or BELOW £{threshold}
- Do NOT include refunds (negative amounts)

OUTPUT FORMAT:
For each transaction above threshold:
- Transaction ID: [ID]
- Amount: £[AMOUNT]
- Date: [DATE]
- Merchant: [MERCHANT]
- Category: [CATEGORY]

SUMMARY:
- Total count: [NUMBER]
- Total amount: £[SUM]
- Average: £[AVG]

Provide complete, accurate results.
""",
            variables=["data", "threshold"]
        ))

        # FW20: Luxury Brands & Money Transfers
        self.add_template(PromptTemplate(
            name="fw20_luxury_transfers",
            description="FW20: Detect luxury brands and money transfers",
            template="""You are an expert in transaction categorization with focus on luxury goods and financial transfers.

TASK: Identify luxury brand purchases and money transfer transactions

LUXURY BRANDS: Gucci, Louis Vuitton, Prada, Chanel, Rolex, Hermes, Cartier, Burberry, 
                Versace, Dior, Tiffany, Bulgari, Armani, Fendi

MONEY TRANSFER COMPANIES: Western Union, MoneyGram, Wise, PayPal Transfer, Revolut, 
                          TransferWise, WorldRemit, Xoom, Remitly, Azimo

DATA:
{data}

PRECISION CRITERIA (98%):
- Only include CONFIRMED luxury brands (check spelling variations)
- Only include CONFIRMED transfer services
- Do NOT include regular retailers

RECALL CRITERIA:
- Check for name variations (e.g., "LV", "Louis V")
- Include partial matches if clearly identified
- Check description field for brand names

OUTPUT:
LUXURY BRANDS:
[List each with transaction ID, amount, merchant]

MONEY TRANSFERS:
[List each with transaction ID, amount, service]

SMALL TRANSACTIONS AGGREGATED (>£{threshold}/month):
[Group by merchant if monthly total exceeds threshold]
""",
            variables=["data", "threshold"]
        ))

        # FW25: Missing Audit Trail
        self.add_template(PromptTemplate(
            name="fw25_missing_audit",
            description="FW25: Identify transactions lacking audit trail",
            template="""You are a compliance auditor identifying transactions with missing documentation.

TASK: Find ALL transactions lacking proper audit trail

INDICATORS OF MISSING AUDIT TRAIL:
- Unknown merchant / "Unknown Merchant"
- Missing merchant information
- Cash withdrawals >£500 without notes
- "Anonymous" transactions
- "Unspecified" merchants
- Foreign exchange without details
- Wire transfers without beneficiary

DATA:
{data}

PRECISION (98%):
- Do NOT flag legitimate merchants with generic names
- Distinguish between "Unknown" and legitimate business names

RECALL:
- Check ALL transaction types
- Review notes field for missing information
- Flag large cash withdrawals without documentation

OUTPUT:
MISSING AUDIT TRAIL TRANSACTIONS:
- Transaction ID: [ID]
- Amount: £[AMOUNT]
- Merchant: [MERCHANT]
- Reason: [WHY AUDIT TRAIL MISSING]
- Risk Level: [LOW/MEDIUM/HIGH]

SUMMARY:
- Total flagged: [COUNT]
- High risk: [COUNT]
- Total amount: £[SUM]
""",
            variables=["data"]
        ))

        # FW30: Missing Months
        self.add_template(PromptTemplate(
            name="fw30_missing_months",
            description="FW30: Detect missing months in statement sequence",
            template="""You are a temporal analysis expert detecting gaps in financial data.

TASK: Identify missing months in a 6-month bank statement sequence

DATA:
{data}

ANALYSIS STEPS:
1. Extract all unique months from transaction dates
2. Determine date range (min to max)
3. Generate expected month sequence
4. Identify gaps

PRECISION (100% required):
- Do NOT flag months with partial data as missing
- Only flag complete calendar month gaps

OUTPUT:
DATE RANGE:
- Start: [YYYY-MM-DD]
- End: [YYYY-MM-DD]
- Span: [N] months

ACTUAL MONTHS WITH TRANSACTIONS:
[List: YYYY-MM]

MISSING MONTHS:
[List: YYYY-MM]

ANALYSIS:
- Expected months: [COUNT]
- Actual months: [COUNT]
- Missing months: [COUNT]
- Is continuous: [YES/NO]
""",
            variables=["data"]
        ))

        # FW40: Light-touch Fraud Detection
        self.add_template(PromptTemplate(
            name="fw40_fraud_detection",
            description="FW40: Detect errors and inconsistencies",
            template="""You are a data quality analyst detecting errors and potential fraud indicators.

TASK: Identify misspellings, calculation errors, and data inconsistencies

DATA:
{data}

DETECTION CATEGORIES:

1. MISSPELLINGS:
   - Bank names (Barclays → "Barcley", HSBC → "HSCB")
   - Merchant names
   - Common typos

2. CALCULATION ERRORS:
   - Decimal point errors (£1000 vs £100.0)
   - Incorrect totals
   - Amount discrepancies

3. DATA QUALITY ISSUES:
   - Duplicate transaction IDs
   - Negative amounts for non-refunds
   - Invalid dates
   - Missing required fields

PRECISION (98%):
- Do NOT flag acceptable name variations
- Verify errors before flagging

OUTPUT:
MISSPELLINGS:
- Transaction: [ID]
- Found: [MISSPELLED]
- Should be: [CORRECT]

CALCULATION ERRORS:
- Transaction: [ID]
- Displayed: £[AMOUNT]
- Expected: £[AMOUNT]

DATA QUALITY ISSUES:
- Issue type: [TYPE]
- Details: [DESCRIPTION]

FRAUD RISK SCORE: [0-10]
REQUIRES REVIEW: [YES/NO]
""",
            variables=["data"]
        ))

        # FW45: Gambling Analysis
        self.add_template(PromptTemplate(
            name="fw45_gambling",
            description="FW45: Analyze gambling transactions over 6 months",
            template="""You are a behavioral finance analyst specializing in gambling activity detection.

TASK: Identify and analyze ALL gambling transactions over 6 months

GAMBLING OPERATORS: Bet365, William Hill, Paddy Power, Ladbrokes, Betfair, 
                    Sky Bet, 888 Casino, Coral, Betway, Unibet, PokerStars,
                    Online Casino, Poker, Betting

DATA:
{data}

ANALYSIS REQUIRED:
1. Total gambling spend
2. Frequency (transactions per month)
3. Pattern analysis (increasing/decreasing trend)
4. Largest single bet
5. Risk indicators

PRECISION (98%):
- Only confirmed gambling operators
- Check for operator name variations

OUTPUT:
GAMBLING TRANSACTIONS:
[List each: ID, Date, Amount, Operator]

SUMMARY:
- Total transactions: [COUNT]
- Total spend: £[AMOUNT]
- Average per transaction: £[AVG]
- Max single bet: £[MAX]
- Months with activity: [COUNT]

MONTHLY BREAKDOWN:
[Month: Count, Total spend]

PATTERN ANALYSIS:
- Trend: [INCREASING/STABLE/DECREASING]
- Risk level: [LOW/MEDIUM/HIGH]

RISK INDICATORS:
[List any concerning patterns]
""",
            variables=["data"]
        ))

        # FW50: Debt Payments
        self.add_template(PromptTemplate(
            name="fw50_debt_payments",
            description="FW50: Identify large debt payments ≥£{threshold}",
            template="""You are a credit analyst identifying debt repayment patterns.

TASK: Identify all debt payments ≥£{threshold}

DEBT PAYMENT KEYWORDS: Loan, Credit Card, Mortgage, Finance, Repayment,
                       Barclaycard, AMEX, Visa Payment, Mastercard Payment

DATA:
{data}

PRECISION (98%):
- Distinguish debt payments from regular purchases
- Verify payment type

DEBT CATEGORIES:
- Credit card payments
- Loan repayments
- Mortgage payments
- Finance agreements

OUTPUT:
LARGE DEBT PAYMENTS (≥£{threshold}):
- Transaction ID: [ID]
- Date: [DATE]
- Amount: £[AMOUNT]
- Creditor: [NAME]
- Type: [CREDIT CARD/LOAN/MORTGAGE/OTHER]

SUMMARY:
- Total payments: [COUNT]
- Total amount: £[SUM]
- Average payment: £[AVG]

CREDITOR BREAKDOWN:
[Creditor: Total paid, Payment count]

MONTHLY TOTALS:
[Month: Total debt payments]

DEBT BURDEN LEVEL: [LOW/MEDIUM/HIGH]
""",
            variables=["data", "threshold"]
        ))

        # === ADVANCED REASONING TEMPLATES ===

        # Beam Search Reasoning
        self.add_template(PromptTemplate(
            name="beam_reasoning",
            description="Beam search: Explore multiple reasoning paths simultaneously",
            template="""You are analyzing bank transactions using beam search reasoning.

TASK: {task_description}

DATA:
{data}

BEAM SEARCH APPROACH:
Explore 3 parallel reasoning paths simultaneously, then select the best.

PATH 1: Conservative Approach
- Use strict matching criteria
- Minimize false positives
- List findings: [...]

PATH 2: Comprehensive Approach  
- Use broader criteria
- Maximize recall
- List findings: [...]

PATH 3: Balanced Approach
- Balance precision and recall
- Use moderate criteria
- List findings: [...]

EVALUATION:
Compare paths on:
- Precision estimate
- Recall estimate
- Confidence level

SELECTED PATH: [1/2/3]
REASONING: [Why this path is best]

FINAL RESULTS:
[Results from selected path]

CONFIDENCE SCORE: [0-100%]
""",
            variables=["task_description", "data"]
        ))

        # Monte Carlo Sampling
        self.add_template(PromptTemplate(
            name="monte_carlo_reasoning",
            description="Monte Carlo: Sample-based probabilistic reasoning",
            template="""You are analyzing transactions using Monte Carlo sampling approach.

TASK: {task_description}

DATA:
{data}

MONTE CARLO APPROACH:
Run 5 independent analysis iterations with slight variations, then aggregate.

ITERATION 1: Strict threshold (confidence ≥90%)
Results: [...]

ITERATION 2: Moderate threshold (confidence ≥75%)
Results: [...]

ITERATION 3: Permissive threshold (confidence ≥60%)
Results: [...]

ITERATION 4: Pattern-based detection
Results: [...]

ITERATION 5: Keyword-based detection
Results: [...]

AGGREGATION:
Items detected in 4+ iterations: [HIGH CONFIDENCE]
Items detected in 3 iterations: [MEDIUM CONFIDENCE]
Items detected in 1-2 iterations: [LOW CONFIDENCE]

FINAL RESULTS:
[Include high + medium confidence items]

PRECISION ESTIMATE: [Based on consistency across iterations]
CONFIDENCE INTERVALS: [Range of possible values]
""",
            variables=["task_description", "data"]
        ))

        # Chain of Thought with Self-Verification
        self.add_template(PromptTemplate(
            name="chain_of_thought_verified",
            description="Chain of thought with self-verification for 98% accuracy",
            template="""You are a meticulous analyst using chain-of-thought reasoning with self-verification.

TASK: {task_description}

DATA:
{data}

STEP 1: Initial Analysis
Thought: Let me identify potential matches...
Findings: [List candidates]

STEP 2: Apply Precision Criteria
Thought: Now I'll verify each candidate against strict criteria...
For each candidate:
  - Does it meet ALL requirements? [YES/NO]
  - Confidence level: [0-100%]
Verified: [Items passing verification]

STEP 3: Check for Missing Items (Recall)
Thought: Did I miss anything? Let me scan again...
Additional items found: [...]

STEP 4: Self-Verification
Question: Are there any false positives in my results?
Check: [Review each item]
Removed: [Any items that don't meet 98% confidence]

Question: Are there any false negatives (missed items)?
Check: [Scan data again]
Added: [Any missed items]

STEP 5: Final Validation
Precision check: [All items meet strict criteria? YES/NO]
Completeness check: [All matching items included? YES/NO]

FINAL RESULTS:
[Verified, high-confidence results]

CONFIDENCE: [%]
ESTIMATED PRECISION: ≥98%
ESTIMATED ACCURACY: ≥98%
""",
            variables=["task_description", "data"]
        ))

        # Tree of Thoughts
        self.add_template(PromptTemplate(
            name="tree_of_thoughts",
            description="Tree of thoughts: Systematic exploration of solution space",
            template="""You are using tree-of-thoughts reasoning for systematic analysis.

TASK: {task_description}

DATA:
{data}

THOUGHT TREE:

ROOT: Problem - {task_description}

BRANCH 1: Amount-based detection
├─ Leaf 1.1: Exact threshold match
│  Results: [...]
├─ Leaf 1.2: Range-based detection  
│  Results: [...]
└─ Evaluation: [Which leaf performed better?]

BRANCH 2: Keyword-based detection
├─ Leaf 2.1: Strict keyword matching
│  Results: [...]
├─ Leaf 2.2: Fuzzy keyword matching
│  Results: [...]
└─ Evaluation: [Which leaf performed better?]

BRANCH 3: Pattern-based detection
├─ Leaf 3.1: Regular expressions
│  Results: [...]
├─ Leaf 3.2: Heuristic patterns
│  Results: [...]
└─ Evaluation: [Which leaf performed better?]

BRANCH SYNTHESIS:
Best from Branch 1: [...]
Best from Branch 2: [...]
Best from Branch 3: [...]

COMBINED RESULTS:
[Union of best approaches]

FINAL OPTIMIZATION:
[Remove duplicates, verify accuracy]

PRECISION: ≥98%
""",
            variables=["task_description", "data"]
        ))

        logger.info(f"Initialized {len(self.templates)} prompt templates (including FW-specific and advanced reasoning)")

    def add_template(self, template: PromptTemplate):
        """Add a template to the library"""
        self.templates[template.name] = template
        logger.debug(f"Added template: {template.name}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name"""
        return self.templates.get(name)

    def get_all_templates(self) -> List[PromptTemplate]:
        """Get all templates"""
        return list(self.templates.values())

    def list_templates(self) -> List[str]:
        """List all template names"""
        return list(self.templates.keys())

    def format_template(self, name: str, **kwargs) -> str:
        """Format a specific template"""
        template = self.get_template(name)
        if template:
            return template.format(**kwargs)
        else:
            logger.error(f"Template '{name}' not found")
            return ""

    def create_custom_template(
        self,
        name: str,
        template_text: str,
        description: str = ""
    ) -> PromptTemplate:
        """Create and add a custom template"""
        custom = PromptTemplate(
            name=name,
            template=template_text,
            description=description
        )
        self.add_template(custom)
        logger.info(f"Created custom template: {name}")
        return custom

    def export_templates(self, filepath: str):
        """Export templates to JSON file"""
        export_data = {
            name: {
                'template': t.template,
                'description': t.description,
                'variables': t.variables
            }
            for name, t in self.templates.items()
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(self.templates)} templates to {filepath}")

    def import_templates(self, filepath: str):
        """Import templates from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            for name, info in data.items():
                template = PromptTemplate(
                    name=name,
                    template=info['template'],
                    description=info.get('description', ''),
                    variables=info.get('variables', [])
                )
                self.add_template(template)

            logger.info(f"Imported {len(data)} templates from {filepath}")

        except Exception as e:
            logger.error(f"Error importing templates: {e}")
