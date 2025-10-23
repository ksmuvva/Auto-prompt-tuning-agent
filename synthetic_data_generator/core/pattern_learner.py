"""
Pattern Recognition & Application System

Learns patterns from user examples and applies them to generated data
"""

from typing import List, Dict, Any, Optional
import re
from collections import Counter


class Pattern:
    """Represents a learned pattern"""

    def __init__(self, pattern_type: str, template: str, examples: List[str], metadata: Optional[Dict] = None):
        self.pattern_type = pattern_type
        self.template = template
        self.examples = examples
        self.metadata = metadata or {}

    def apply(self, **kwargs) -> str:
        """Apply pattern with given values"""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class PatternLearner:
    """Learns and applies patterns from examples"""

    def __init__(self, llm_provider):
        """
        Initialize pattern learner

        Args:
            llm_provider: LLM provider for complex pattern recognition
        """
        self.llm = llm_provider
        self.learned_patterns = {}

    def learn_from_examples(self, examples: List[str], field_name: str) -> Pattern:
        """
        Learn pattern from examples

        Args:
            examples: List of example values
            field_name: Name of the field

        Returns:
            Learned Pattern object

        Examples:
            >>> learner = PatternLearner(llm)
            >>> pattern = learner.learn_from_examples(
            ...     ["john.smith@company.co.uk", "j.doe@company.co.uk"],
            ...     "email"
            ... )
        """
        # Try rule-based pattern extraction first
        pattern = self._extract_pattern_rules(examples, field_name)

        if not pattern:
            # Use LLM for complex patterns
            pattern = self._extract_pattern_llm(examples, field_name)

        # Store learned pattern
        self.learned_patterns[field_name] = pattern

        return pattern

    def _extract_pattern_rules(self, examples: List[str], field_name: str) -> Optional[Pattern]:
        """Extract pattern using rule-based methods"""

        if not examples:
            return None

        # Detect pattern type
        if all('@' in ex for ex in examples):
            return self._learn_email_pattern(examples, field_name)
        elif all(re.match(r'^\d{3,}', ex) for ex in examples):
            return self._learn_id_pattern(examples, field_name)
        elif all(re.match(r'^\d{2}/\d{2}/\d{4}', ex) for ex in examples):
            return Pattern('date', 'DD/MM/YYYY', examples, {'format': 'UK date'})
        else:
            return None

    def _learn_email_pattern(self, examples: List[str], field_name: str) -> Pattern:
        """Learn email pattern from examples"""

        # Extract domains
        domains = [ex.split('@')[1] for ex in examples]
        domain_counts = Counter(domains)
        common_domain = domain_counts.most_common(1)[0][0]

        # Extract username patterns
        usernames = [ex.split('@')[0] for ex in examples]

        # Detect format patterns
        formats = []
        for username in usernames:
            if '.' in username:
                parts = username.split('.')
                if len(parts) == 2:
                    if len(parts[0]) > 1:
                        formats.append('firstname.lastname')
                    else:
                        formats.append('f.lastname')
            elif re.match(r'^[a-z]\w+$', username):
                formats.append('flastname')
            else:
                formats.append('custom')

        format_counts = Counter(formats)
        common_format = format_counts.most_common(1)[0][0]

        # Create pattern template
        if common_format == 'firstname.lastname':
            template = '{firstname}.{lastname}@' + common_domain
        elif common_format == 'f.lastname':
            template = '{first_initial}.{lastname}@' + common_domain
        elif common_format == 'flastname':
            template = '{first_initial}{lastname}@' + common_domain
        else:
            template = '{username}@' + common_domain

        return Pattern(
            'email',
            template,
            examples,
            {
                'domain': common_domain,
                'format': common_format,
                'format_distribution': dict(format_counts)
            }
        )

    def _learn_id_pattern(self, examples: List[str], field_name: str) -> Pattern:
        """Learn ID pattern from examples"""

        # Analyze ID structure
        # Look for common patterns like: EMP-IT-2023-0001, CUST001, etc.

        # Check for separators
        if all('-' in ex for ex in examples):
            separator = '-'
            parts = [ex.split(separator) for ex in examples]
            num_parts = len(parts[0])

            # Build template
            template_parts = []
            for i in range(num_parts):
                part_values = [p[i] for p in parts]
                if all(p.isdigit() for p in part_values):
                    # Numeric part
                    max_len = max(len(p) for p in part_values)
                    template_parts.append(f"{{number{i}:0{max_len}d}}")
                else:
                    # Check if constant or variable
                    if len(set(part_values)) == 1:
                        template_parts.append(part_values[0])  # Constant
                    else:
                        template_parts.append(f"{{part{i}}}")  # Variable

            template = separator.join(template_parts)
        else:
            # No separator, simple ID
            template = '{id}'

        return Pattern('id', template, examples, {'separator': separator if separator else None})

    def _extract_pattern_llm(self, examples: List[str], field_name: str) -> Pattern:
        """Extract complex patterns using LLM"""

        pattern_prompt = f"""Analyze these examples for field "{field_name}" and extract the pattern:

Examples:
{chr(10).join(f"- {ex}" for ex in examples[:10])}

Identify:
1. Pattern type (email, id, phone, custom, etc.)
2. Template with placeholders (e.g., "{{firstname}}.{{lastname}}@domain.com")
3. Any rules or constraints

Respond with JSON:
{{
  "pattern_type": "type",
  "template": "template with {{placeholders}}",
  "rules": ["rule1", "rule2"],
  "metadata": {{"key": "value"}}
}}"""

        try:
            response = self.llm.generate_structured(
                pattern_prompt,
                schema={
                    "pattern_type": "string",
                    "template": "string",
                    "rules": "array",
                    "metadata": "object"
                },
                temperature=0.3
            )

            return Pattern(
                response.get('pattern_type', 'custom'),
                response.get('template', '{value}'),
                examples,
                response.get('metadata', {})
            )

        except Exception as e:
            print(f"Warning: LLM pattern extraction failed: {e}")
            # Fallback: simple pattern
            return Pattern('custom', '{value}', examples, {})

    def apply_pattern(self, pattern_name: str, count: int, **kwargs) -> List[str]:
        """
        Apply learned pattern to generate new values

        Args:
            pattern_name: Name of the learned pattern
            count: Number of values to generate
            **kwargs: Additional parameters for pattern

        Returns:
            List of generated values
        """
        if pattern_name not in self.learned_patterns:
            raise ValueError(f"Pattern '{pattern_name}' not learned yet")

        pattern = self.learned_patterns[pattern_name]

        # Use LLM to generate values following the pattern
        generation_prompt = f"""Generate {count} new values following this pattern:

Pattern Type: {pattern.pattern_type}
Template: {pattern.template}
Examples:
{chr(10).join(f"- {ex}" for ex in pattern.examples[:5])}

Metadata: {pattern.metadata}

Generate {count} DIFFERENT values that match this exact pattern.
Ensure variety and realism.

Respond with JSON array of strings:
["value1", "value2", ...]"""

        try:
            response = self.llm.generate(generation_prompt, temperature=0.7)

            # Parse JSON array
            import json
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                values = json.loads(json_match.group(0))
                return values[:count]
            else:
                raise ValueError("Could not parse JSON array")

        except Exception as e:
            print(f"Warning: Pattern application failed: {e}")
            # Fallback: return examples with variations
            return [f"{ex}_variant_{i}" for i, ex in enumerate(pattern.examples[:count])]

    def suggest_patterns(self, data_samples: List[Dict[str, Any]]) -> Dict[str, Pattern]:
        """
        Suggest patterns from sample data

        Args:
            data_samples: Sample records

        Returns:
            Dictionary mapping field names to suggested patterns
        """
        if not data_samples:
            return {}

        patterns = {}

        # Group values by field
        fields = data_samples[0].keys()
        for field in fields:
            values = [str(sample.get(field, '')) for sample in data_samples if sample.get(field)]

            if len(values) >= 2:
                # Learn pattern from these examples
                try:
                    pattern = self.learn_from_examples(values, field)
                    patterns[field] = pattern
                except Exception as e:
                    print(f"Warning: Could not learn pattern for {field}: {e}")

        return patterns

    def validate_against_pattern(self, value: str, pattern: Pattern) -> bool:
        """
        Validate if a value matches a pattern

        Args:
            value: Value to validate
            pattern: Pattern to check against

        Returns:
            True if value matches pattern
        """
        validation_prompt = f"""Does this value match the pattern?

Value: {value}
Pattern Type: {pattern.pattern_type}
Template: {pattern.template}
Examples:
{chr(10).join(f"- {ex}" for ex in pattern.examples[:3])}

Respond with JSON:
{{
  "matches": true/false,
  "reason": "explanation"
}}"""

        try:
            response = self.llm.generate_structured(
                validation_prompt,
                schema={"matches": "boolean", "reason": "string"},
                temperature=0.1
            )

            return response.get('matches', False)

        except Exception as e:
            print(f"Warning: Pattern validation failed: {e}")
            return False
