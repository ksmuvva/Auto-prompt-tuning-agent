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

        logger.info(f"Initialized {len(self.templates)} prompt templates")

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
