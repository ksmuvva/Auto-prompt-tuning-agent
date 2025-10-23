"""
Ambiguity Detection & Clarification System

Detects unclear prompts and interactively clarifies requirements
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .intent_engine import Intent


@dataclass
class Clarification:
    """Clarification question for user"""
    question: str
    options: List[str]
    default: Optional[str] = None
    required: bool = True


class AmbiguityDetector:
    """Detector for ambiguous or incomplete requests"""

    def __init__(self, llm_provider):
        """
        Initialize ambiguity detector

        Args:
            llm_provider: LLM provider for generating clarification questions
        """
        self.llm = llm_provider

    def detect_ambiguities(self, intent: Intent) -> List[Clarification]:
        """
        Detect ambiguities in user intent

        Args:
            intent: Parsed user intent

        Returns:
            List of clarification questions
        """
        clarifications = []

        # Check critical missing information
        if not intent.count or intent.count < 1:
            clarifications.append(Clarification(
                question="How many records should I generate?",
                options=["100", "1000", "10000", "Custom amount"],
                default="100",
                required=True
            ))

        if not intent.output_format:
            clarifications.append(Clarification(
                question="What output format would you like?",
                options=["CSV (tabular data)", "JSON (API/structured)", "Excel (.xlsx)",
                        "PDF (report)", "Word (.docx)", "Markdown"],
                default="CSV",
                required=True
            ))

        if not intent.fields or len(intent.fields) == 0:
            clarifications.append(Clarification(
                question=f"What fields should be included in the {intent.data_type}?",
                options=["Basic (minimal fields)", "Standard (common fields)",
                        "Extended (comprehensive)", "Custom (you specify)"],
                default="Standard",
                required=True
            ))

        if not intent.geography:
            clarifications.append(Clarification(
                question="What geography/region should the data represent?",
                options=["UK (British standards)", "US (American standards)",
                        "EU (European standards)", "Global", "Not applicable"],
                default="UK",
                required=False
            ))

        if not intent.purpose:
            clarifications.append(Clarification(
                question="What's the purpose of this data?",
                options=["Testing (QA/validation)", "Training (ML/AI)",
                        "Demo (presentation)", "Development (coding)", "Other"],
                default="Testing",
                required=False
            ))

        # Check for domain-specific ambiguities
        domain_clarifications = self._get_domain_specific_clarifications(intent)
        clarifications.extend(domain_clarifications)

        # Use LLM to detect additional ambiguities
        llm_clarifications = self._detect_with_llm(intent)
        clarifications.extend(llm_clarifications)

        return clarifications

    def _get_domain_specific_clarifications(self, intent: Intent) -> List[Clarification]:
        """Get domain-specific clarification questions"""
        clarifications = []

        # Healthcare domain
        if intent.domain == 'healthcare':
            clarifications.append(Clarification(
                question="What type of patient data?",
                options=["Demographics only", "Medical history",
                        "Lab results", "All of the above"],
                default="Demographics only",
                required=True
            ))
            clarifications.append(Clarification(
                question="What healthcare standard?",
                options=["UK (NHS)", "US (HIPAA)", "EU (GDPR)", "Other"],
                default="UK (NHS)",
                required=False
            ))

        # E-commerce domain
        elif intent.domain == 'e-commerce':
            if intent.entity.lower() == 'customer':
                clarifications.append(Clarification(
                    question="Include purchase history?",
                    options=["Yes, include transactions", "No, demographics only"],
                    default="No",
                    required=False
                ))

        # Finance domain
        elif intent.domain == 'finance':
            clarifications.append(Clarification(
                question="What currency?",
                options=["GBP (£)", "USD ($)", "EUR (€)", "Multiple"],
                default="GBP (£)",
                required=False
            ))

        return clarifications

    def _detect_with_llm(self, intent: Intent) -> List[Clarification]:
        """Use LLM to detect additional ambiguities"""

        detection_prompt = f"""Analyze this data generation request for ambiguities:

Data Type: {intent.data_type}
Count: {intent.count}
Geography: {intent.geography}
Purpose: {intent.purpose}
Domain: {intent.domain}
Fields: {intent.fields}

Detect any ambiguities, missing critical information, or areas that need clarification.
Consider:
- Data quality requirements
- Validation rules
- Constraints
- Relationships between fields
- Realistic distributions

Return up to 3 most important clarification questions as JSON:
{{
  "clarifications": [
    {{
      "question": "Question text?",
      "options": ["Option 1", "Option 2", "Option 3"],
      "default": "Option 1",
      "required": true
    }}
  ]
}}

Return empty array if no critical ambiguities detected."""

        try:
            response = self.llm.generate_structured(
                detection_prompt,
                schema={"clarifications": "array"},
                temperature=0.3
            )

            clarifications = []
            for item in response.get('clarifications', []):
                clarifications.append(Clarification(
                    question=item.get('question', ''),
                    options=item.get('options', []),
                    default=item.get('default'),
                    required=item.get('required', False)
                ))

            return clarifications

        except Exception as e:
            print(f"Warning: LLM ambiguity detection failed: {e}")
            return []

    def format_clarifications(self, clarifications: List[Clarification]) -> str:
        """
        Format clarifications as user-friendly text

        Args:
            clarifications: List of clarification questions

        Returns:
            Formatted string for display
        """
        if not clarifications:
            return "No ambiguities detected. Ready to generate!"

        output = ["I have some questions to clarify your request:\n"]

        for i, clarif in enumerate(clarifications, 1):
            required = " (required)" if clarif.required else " (optional)"
            output.append(f"\n{i}. {clarif.question}{required}")

            for j, option in enumerate(clarif.options, 1):
                default_marker = " [DEFAULT]" if option == clarif.default else ""
                output.append(f"   {chr(96+j)}) {option}{default_marker}")

        output.append("\nPress Enter to use defaults or specify your choices.")
        return "\n".join(output)

    def resolve_clarifications(
        self,
        intent: Intent,
        clarifications: List[Clarification],
        answers: Dict[str, str]
    ) -> Intent:
        """
        Apply user answers to resolve ambiguities

        Args:
            intent: Original intent
            clarifications: List of clarifications
            answers: User's answers mapping question to answer

        Returns:
            Updated intent with resolved ambiguities
        """
        # Update intent based on answers
        for clarif in clarifications:
            answer = answers.get(clarif.question, clarif.default)

            # Map answers to intent fields
            if "how many" in clarif.question.lower():
                try:
                    intent.count = int(answer)
                except ValueError:
                    pass

            elif "output format" in clarif.question.lower():
                format_map = {
                    "CSV": "csv",
                    "JSON": "json",
                    "Excel": "excel",
                    "PDF": "pdf",
                    "Word": "word",
                    "Markdown": "markdown"
                }
                for key, value in format_map.items():
                    if key in answer:
                        intent.output_format = value
                        break

            elif "geography" in clarif.question.lower():
                if "UK" in answer:
                    intent.geography = "UK"
                elif "US" in answer:
                    intent.geography = "US"
                elif "EU" in answer:
                    intent.geography = "EU"

            elif "purpose" in clarif.question.lower():
                if "Testing" in answer:
                    intent.purpose = "testing"
                elif "Training" in answer:
                    intent.purpose = "training"
                elif "Demo" in answer:
                    intent.purpose = "demo"
                elif "Development" in answer:
                    intent.purpose = "development"

        # Clear ambiguities flag
        intent.ambiguities = []

        return intent
