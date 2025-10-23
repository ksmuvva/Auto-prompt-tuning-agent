"""
Intent & Context Understanding Engine

Understands WHAT user wants and WHY at a micro level
Extracts entities, counts, geography, purpose, constraints, and implicit requirements
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import re


@dataclass
class Intent:
    """User intent extracted from prompt"""
    data_type: str                    # e.g., "customer records", "patient data"
    entity: str                       # e.g., "Customer", "Patient"
    count: int                        # Number of records to generate
    geography: Optional[str] = None   # e.g., "UK", "US"
    purpose: Optional[str] = None     # e.g., "testing", "training", "demo"
    domain: Optional[str] = None      # e.g., "e-commerce", "healthcare", "finance"
    output_format: Optional[str] = None  # e.g., "csv", "json", "pdf"
    fields: Optional[List[str]] = None   # Specific fields requested
    constraints: Optional[Dict[str, Any]] = None  # Additional constraints
    implicit_requirements: Optional[List[str]] = None  # Inferred requirements
    ambiguities: Optional[List[str]] = None  # Detected ambiguities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def has_ambiguities(self) -> bool:
        """Check if there are ambiguities"""
        return bool(self.ambiguities)


class IntentEngine:
    """Engine for understanding user intent from natural language prompts"""

    def __init__(self, llm_provider):
        """
        Initialize intent engine

        Args:
            llm_provider: LLM provider instance for understanding complex prompts
        """
        self.llm = llm_provider

    def parse_intent(self, prompt: str) -> Intent:
        """
        Parse user prompt and extract intent

        Args:
            prompt: Natural language prompt from user

        Returns:
            Intent object with extracted information

        Examples:
            >>> engine = IntentEngine(llm)
            >>> intent = engine.parse_intent("Generate 1000 UK customer records for e-commerce testing")
            >>> print(intent.count)  # 1000
            >>> print(intent.geography)  # "UK"
        """
        # First try rule-based extraction (fast)
        basic_intent = self._extract_basic_intent(prompt)

        # Then use LLM for deeper understanding
        enhanced_intent = self._enhance_with_llm(prompt, basic_intent)

        return enhanced_intent

    def _extract_basic_intent(self, prompt: str) -> Intent:
        """Extract basic intent using rule-based patterns"""

        # Extract count
        count = self._extract_count(prompt)

        # Extract data type and entity
        data_type, entity = self._extract_data_type(prompt)

        # Extract geography
        geography = self._extract_geography(prompt)

        # Extract purpose
        purpose = self._extract_purpose(prompt)

        # Extract domain
        domain = self._extract_domain(prompt)

        # Extract output format
        output_format = self._extract_output_format(prompt)

        return Intent(
            data_type=data_type,
            entity=entity,
            count=count,
            geography=geography,
            purpose=purpose,
            domain=domain,
            output_format=output_format
        )

    def _extract_count(self, prompt: str) -> int:
        """Extract number of records to generate"""
        # Look for patterns like "1000", "1,000", "1k", "10k", etc.
        patterns = [
            r'(\d+)k\s+(?:records|rows|entries|items|customers|patients)',
            r'(\d+),(\d+)\s+(?:records|rows|entries|items|customers|patients)',
            r'(\d+)\s+(?:records|rows|entries|items|customers|patients)',
            r'generate\s+(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                if 'k' in pattern:
                    return int(match.group(1)) * 1000
                elif ',' in pattern:
                    return int(match.group(1) + match.group(2))
                else:
                    return int(match.group(1))

        return 100  # Default

    def _extract_data_type(self, prompt: str) -> tuple[str, str]:
        """Extract data type and entity"""
        # Common data types
        data_types = {
            'customer': ('customer records', 'Customer'),
            'patient': ('patient records', 'Patient'),
            'transaction': ('transaction records', 'Transaction'),
            'employee': ('employee records', 'Employee'),
            'product': ('product records', 'Product'),
            'user': ('user records', 'User'),
            'order': ('order records', 'Order'),
            'invoice': ('invoice records', 'Invoice'),
            'sale': ('sales records', 'Sale'),
        }

        prompt_lower = prompt.lower()
        for key, (data_type, entity) in data_types.items():
            if key in prompt_lower:
                return data_type, entity

        return 'records', 'Record'  # Default

    def _extract_geography(self, prompt: str) -> Optional[str]:
        """Extract geography/region"""
        geographies = ['uk', 'us', 'usa', 'eu', 'europe', 'asia', 'australia']

        prompt_lower = prompt.lower()
        for geo in geographies:
            if geo in prompt_lower:
                return geo.upper() if geo in ['uk', 'us', 'eu'] else geo.capitalize()

        return None

    def _extract_purpose(self, prompt: str) -> Optional[str]:
        """Extract purpose/use case"""
        purposes = {
            'testing': ['testing', 'test', 'qa'],
            'training': ['training', 'ml training', 'machine learning'],
            'demo': ['demo', 'demonstration', 'presentation'],
            'development': ['development', 'dev', 'development'],
        }

        prompt_lower = prompt.lower()
        for purpose, keywords in purposes.items():
            if any(kw in prompt_lower for kw in keywords):
                return purpose

        return None

    def _extract_domain(self, prompt: str) -> Optional[str]:
        """Extract domain/industry"""
        domains = {
            'e-commerce': ['e-commerce', 'ecommerce', 'online shopping', 'retail'],
            'healthcare': ['healthcare', 'medical', 'hospital', 'clinical'],
            'finance': ['finance', 'financial', 'banking', 'bank'],
            'education': ['education', 'school', 'university', 'academic'],
        }

        prompt_lower = prompt.lower()
        for domain, keywords in domains.items():
            if any(kw in prompt_lower for kw in keywords):
                return domain

        return None

    def _extract_output_format(self, prompt: str) -> Optional[str]:
        """Extract output format"""
        formats = ['csv', 'json', 'pdf', 'excel', 'xlsx', 'word', 'docx', 'markdown', 'md']

        prompt_lower = prompt.lower()
        for fmt in formats:
            if fmt in prompt_lower:
                # Normalize format names
                if fmt in ['xlsx', 'excel']:
                    return 'excel'
                elif fmt in ['docx', 'word']:
                    return 'word'
                elif fmt == 'md':
                    return 'markdown'
                return fmt

        return None

    def _enhance_with_llm(self, prompt: str, basic_intent: Intent) -> Intent:
        """Enhance basic intent with LLM for deeper understanding"""

        enhancement_prompt = f"""Analyze this data generation request and extract detailed intent:

User Request: "{prompt}"

Basic extraction:
- Data Type: {basic_intent.data_type}
- Count: {basic_intent.count}
- Geography: {basic_intent.geography}
- Purpose: {basic_intent.purpose}
- Domain: {basic_intent.domain}

Please provide:
1. Implicit requirements (what's needed but not stated)
2. Suggested fields for this data type
3. Any ambiguities or missing information
4. Constraints (realistic values, validation rules)

Respond with JSON:
{{
  "implicit_requirements": ["requirement1", "requirement2"],
  "suggested_fields": ["field1", "field2"],
  "ambiguities": ["ambiguity1"] or [],
  "constraints": {{"key": "value"}}
}}"""

        try:
            response = self.llm.generate_structured(
                enhancement_prompt,
                schema={
                    "implicit_requirements": "array",
                    "suggested_fields": "array",
                    "ambiguities": "array",
                    "constraints": "object"
                },
                temperature=0.3  # Lower temperature for structured extraction
            )

            # Update intent with LLM insights
            basic_intent.implicit_requirements = response.get('implicit_requirements', [])
            basic_intent.fields = response.get('suggested_fields', [])
            basic_intent.ambiguities = response.get('ambiguities', [])
            basic_intent.constraints = response.get('constraints', {})

        except Exception as e:
            print(f"Warning: LLM enhancement failed: {e}")
            # Fall back to basic intent
            basic_intent.implicit_requirements = []
            basic_intent.ambiguities = []

        return basic_intent

    def get_schema_suggestion(self, intent: Intent) -> Dict[str, str]:
        """
        Suggest data schema based on intent

        Args:
            intent: Parsed intent

        Returns:
            Dictionary of field names to types
        """
        schema_prompt = f"""Based on this data generation request, suggest a complete data schema.

Data Type: {intent.data_type}
Entity: {intent.entity}
Domain: {intent.domain}
Geography: {intent.geography}
Purpose: {intent.purpose}
Requested Fields: {intent.fields}

Provide a schema with field names and types. Include all necessary fields for a realistic dataset.

Respond with JSON mapping field names to types (string, number, date, boolean, email, phone, etc.):
{{
  "field_name": "type",
  "another_field": "type"
}}"""

        try:
            schema = self.llm.generate_structured(
                schema_prompt,
                schema={"field": "type"},
                temperature=0.3
            )
            return schema
        except Exception as e:
            print(f"Warning: Schema suggestion failed: {e}")
            return {
                "id": "string",
                "name": "string",
                "created_date": "date"
            }
