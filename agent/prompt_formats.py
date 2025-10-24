"""
Multi-Format Prompt Support

This module provides support for different prompt formats:
1. XML - Structured hierarchical prompts
2. POML (Prompt Markup Language) - Custom prompt markup
3. JSON - JSON-based prompts
4. YAML - YAML-based prompts
5. Markdown - Markdown-formatted prompts
6. Plain text - Traditional text prompts

Program of Thoughts:
1. Define format parsers and formatters
2. Convert between formats
3. Validate format correctness
4. Optimize prompts for specific LLM providers (some prefer XML, others JSON)
"""

import json
import yaml
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import re


class PromptFormat(Enum):
    """Supported prompt formats"""
    XML = "xml"
    POML = "poml"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    PLAIN = "plain"


@dataclass
class PromptStructure:
    """
    Structured representation of a prompt

    Allows conversion between different formats.
    """
    role: str = "user"
    task: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    instructions: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_format: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'role': self.role,
            'task': self.task,
            'context': self.context,
            'instructions': self.instructions,
            'examples': self.examples,
            'constraints': self.constraints,
            'output_format': self.output_format,
            'metadata': self.metadata
        }


class PromptFormatter:
    """Base class for prompt formatters"""

    def __init__(self):
        self.format_type = PromptFormat.PLAIN

    def format(self, structure: PromptStructure) -> str:
        """Format a structured prompt"""
        raise NotImplementedError

    def parse(self, text: str) -> PromptStructure:
        """Parse a formatted prompt"""
        raise NotImplementedError


class XMLPromptFormatter(PromptFormatter):
    """
    XML format for prompts

    Example:
    <prompt>
        <role>analyst</role>
        <task>Analyze transaction data</task>
        <context>
            <data>...</data>
            <threshold>250</threshold>
        </context>
        <instructions>
            <instruction>Identify high-value transactions</instruction>
            <instruction>Detect anomalies</instruction>
        </instructions>
        <output_format>structured_list</output_format>
    </prompt>
    """

    def __init__(self):
        super().__init__()
        self.format_type = PromptFormat.XML

    def format(self, structure: PromptStructure) -> str:
        """Format as XML"""
        root = ET.Element('prompt')

        # Role
        if structure.role:
            role_elem = ET.SubElement(root, 'role')
            role_elem.text = structure.role

        # Task
        if structure.task:
            task_elem = ET.SubElement(root, 'task')
            task_elem.text = structure.task

        # Context
        if structure.context:
            context_elem = ET.SubElement(root, 'context')
            for key, value in structure.context.items():
                item_elem = ET.SubElement(context_elem, key)
                item_elem.text = str(value)

        # Instructions
        if structure.instructions:
            instructions_elem = ET.SubElement(root, 'instructions')
            for instruction in structure.instructions:
                inst_elem = ET.SubElement(instructions_elem, 'instruction')
                inst_elem.text = instruction

        # Examples
        if structure.examples:
            examples_elem = ET.SubElement(root, 'examples')
            for example in structure.examples:
                ex_elem = ET.SubElement(examples_elem, 'example')
                for key, value in example.items():
                    key_elem = ET.SubElement(ex_elem, key)
                    key_elem.text = str(value)

        # Constraints
        if structure.constraints:
            constraints_elem = ET.SubElement(root, 'constraints')
            for constraint in structure.constraints:
                const_elem = ET.SubElement(constraints_elem, 'constraint')
                const_elem.text = constraint

        # Output format
        if structure.output_format:
            output_elem = ET.SubElement(root, 'output_format')
            output_elem.text = structure.output_format

        # Convert to string with pretty printing
        return self._prettify(root)

    def _prettify(self, elem: ET.Element) -> str:
        """Pretty print XML"""
        from xml.dom import minidom
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def parse(self, text: str) -> PromptStructure:
        """Parse XML prompt"""
        root = ET.fromstring(text)

        structure = PromptStructure()

        # Parse role
        role_elem = root.find('role')
        if role_elem is not None and role_elem.text:
            structure.role = role_elem.text

        # Parse task
        task_elem = root.find('task')
        if task_elem is not None and task_elem.text:
            structure.task = task_elem.text

        # Parse context
        context_elem = root.find('context')
        if context_elem is not None:
            structure.context = {
                child.tag: child.text for child in context_elem
            }

        # Parse instructions
        instructions_elem = root.find('instructions')
        if instructions_elem is not None:
            structure.instructions = [
                inst.text for inst in instructions_elem.findall('instruction')
                if inst.text
            ]

        # Parse examples
        examples_elem = root.find('examples')
        if examples_elem is not None:
            structure.examples = [
                {child.tag: child.text for child in ex}
                for ex in examples_elem.findall('example')
            ]

        # Parse constraints
        constraints_elem = root.find('constraints')
        if constraints_elem is not None:
            structure.constraints = [
                const.text for const in constraints_elem.findall('constraint')
                if const.text
            ]

        # Parse output format
        output_elem = root.find('output_format')
        if output_elem is not None and output_elem.text:
            structure.output_format = output_elem.text

        return structure


class JSONPromptFormatter(PromptFormatter):
    """
    JSON format for prompts

    Example:
    {
        "role": "analyst",
        "task": "Analyze transaction data",
        "context": {...},
        "instructions": [...],
        "output_format": "structured_list"
    }
    """

    def __init__(self):
        super().__init__()
        self.format_type = PromptFormat.JSON

    def format(self, structure: PromptStructure) -> str:
        """Format as JSON"""
        return json.dumps(structure.to_dict(), indent=2)

    def parse(self, text: str) -> PromptStructure:
        """Parse JSON prompt"""
        data = json.loads(text)
        return PromptStructure(**data)


class YAMLPromptFormatter(PromptFormatter):
    """
    YAML format for prompts

    Example:
    role: analyst
    task: Analyze transaction data
    context:
      data: ...
      threshold: 250
    instructions:
      - Identify high-value transactions
      - Detect anomalies
    """

    def __init__(self):
        super().__init__()
        self.format_type = PromptFormat.YAML

    def format(self, structure: PromptStructure) -> str:
        """Format as YAML"""
        return yaml.dump(structure.to_dict(), default_flow_style=False, sort_keys=False)

    def parse(self, text: str) -> PromptStructure:
        """Parse YAML prompt"""
        data = yaml.safe_load(text)
        return PromptStructure(**data)


class MarkdownPromptFormatter(PromptFormatter):
    """
    Markdown format for prompts

    Example:
    # Task: Analyze transaction data

    ## Role
    You are a financial analyst.

    ## Instructions
    - Identify high-value transactions
    - Detect anomalies

    ## Context
    - Threshold: 250 GBP
    - Data: ...
    """

    def __init__(self):
        super().__init__()
        self.format_type = PromptFormat.MARKDOWN

    def format(self, structure: PromptStructure) -> str:
        """Format as Markdown"""
        lines = []

        # Task as title
        if structure.task:
            lines.append(f"# Task: {structure.task}\n")

        # Role
        if structure.role:
            lines.append(f"## Role\n")
            lines.append(f"You are a {structure.role}.\n")

        # Instructions
        if structure.instructions:
            lines.append("## Instructions\n")
            for instruction in structure.instructions:
                lines.append(f"- {instruction}")
            lines.append("")

        # Context
        if structure.context:
            lines.append("## Context\n")
            for key, value in structure.context.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Examples
        if structure.examples:
            lines.append("## Examples\n")
            for i, example in enumerate(structure.examples, 1):
                lines.append(f"### Example {i}\n")
                for key, value in example.items():
                    lines.append(f"**{key}**: {value}\n")

        # Constraints
        if structure.constraints:
            lines.append("## Constraints\n")
            for constraint in structure.constraints:
                lines.append(f"- {constraint}")
            lines.append("")

        # Output format
        if structure.output_format:
            lines.append(f"## Output Format\n")
            lines.append(f"{structure.output_format}\n")

        return "\n".join(lines)

    def parse(self, text: str) -> PromptStructure:
        """Parse Markdown prompt"""
        structure = PromptStructure()

        # Extract task from title
        task_match = re.search(r'#\s+Task:\s+(.+)', text)
        if task_match:
            structure.task = task_match.group(1).strip()

        # Extract role
        role_section = re.search(r'##\s+Role\s+You are a (.+?)\.', text, re.DOTALL)
        if role_section:
            structure.role = role_section.group(1).strip()

        # Extract instructions
        instructions_section = re.search(r'##\s+Instructions\s+((?:- .+\n?)+)', text)
        if instructions_section:
            instructions_text = instructions_section.group(1)
            structure.instructions = [
                line.strip('- ').strip()
                for line in instructions_text.split('\n')
                if line.strip().startswith('-')
            ]

        # Extract context
        context_section = re.search(r'##\s+Context\s+((?:- .+\n?)+)', text)
        if context_section:
            context_text = context_section.group(1)
            for line in context_text.split('\n'):
                if ':' in line:
                    key_value = line.strip('- ').split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip('*').strip()
                        value = key_value[1].strip()
                        structure.context[key] = value

        # Extract constraints
        constraints_section = re.search(r'##\s+Constraints\s+((?:- .+\n?)+)', text)
        if constraints_section:
            constraints_text = constraints_section.group(1)
            structure.constraints = [
                line.strip('- ').strip()
                for line in constraints_text.split('\n')
                if line.strip().startswith('-')
            ]

        return structure


class POMLPromptFormatter(PromptFormatter):
    """
    POML (Prompt Markup Language) - Custom format optimized for prompts

    Example:
    [ROLE: analyst]
    [TASK: Analyze transaction data]

    [INSTRUCTIONS]
    - Identify high-value transactions
    - Detect anomalies
    [/INSTRUCTIONS]

    [CONTEXT]
    threshold = 250
    data = ...
    [/CONTEXT]

    [OUTPUT: structured_list]
    """

    def __init__(self):
        super().__init__()
        self.format_type = PromptFormat.POML

    def format(self, structure: PromptStructure) -> str:
        """Format as POML"""
        lines = []

        # Role
        if structure.role:
            lines.append(f"[ROLE: {structure.role}]")

        # Task
        if structure.task:
            lines.append(f"[TASK: {structure.task}]")

        lines.append("")

        # Instructions
        if structure.instructions:
            lines.append("[INSTRUCTIONS]")
            for instruction in structure.instructions:
                lines.append(f"- {instruction}")
            lines.append("[/INSTRUCTIONS]")
            lines.append("")

        # Context
        if structure.context:
            lines.append("[CONTEXT]")
            for key, value in structure.context.items():
                lines.append(f"{key} = {value}")
            lines.append("[/CONTEXT]")
            lines.append("")

        # Examples
        if structure.examples:
            lines.append("[EXAMPLES]")
            for i, example in enumerate(structure.examples, 1):
                lines.append(f"[EXAMPLE {i}]")
                for key, value in example.items():
                    lines.append(f"{key}: {value}")
                lines.append(f"[/EXAMPLE {i}]")
            lines.append("[/EXAMPLES]")
            lines.append("")

        # Constraints
        if structure.constraints:
            lines.append("[CONSTRAINTS]")
            for constraint in structure.constraints:
                lines.append(f"- {constraint}")
            lines.append("[/CONSTRAINTS]")
            lines.append("")

        # Output format
        if structure.output_format:
            lines.append(f"[OUTPUT: {structure.output_format}]")

        return "\n".join(lines)

    def parse(self, text: str) -> PromptStructure:
        """Parse POML prompt"""
        structure = PromptStructure()

        # Extract role
        role_match = re.search(r'\[ROLE:\s*(.+?)\]', text)
        if role_match:
            structure.role = role_match.group(1).strip()

        # Extract task
        task_match = re.search(r'\[TASK:\s*(.+?)\]', text)
        if task_match:
            structure.task = task_match.group(1).strip()

        # Extract instructions
        instructions_match = re.search(r'\[INSTRUCTIONS\](.*?)\[/INSTRUCTIONS\]', text, re.DOTALL)
        if instructions_match:
            instructions_text = instructions_match.group(1)
            structure.instructions = [
                line.strip('- ').strip()
                for line in instructions_text.split('\n')
                if line.strip().startswith('-')
            ]

        # Extract context
        context_match = re.search(r'\[CONTEXT\](.*?)\[/CONTEXT\]', text, re.DOTALL)
        if context_match:
            context_text = context_match.group(1)
            for line in context_text.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    structure.context[key.strip()] = value.strip()

        # Extract constraints
        constraints_match = re.search(r'\[CONSTRAINTS\](.*?)\[/CONSTRAINTS\]', text, re.DOTALL)
        if constraints_match:
            constraints_text = constraints_match.group(1)
            structure.constraints = [
                line.strip('- ').strip()
                for line in constraints_text.split('\n')
                if line.strip().startswith('-')
            ]

        # Extract output format
        output_match = re.search(r'\[OUTPUT:\s*(.+?)\]', text)
        if output_match:
            structure.output_format = output_match.group(1).strip()

        return structure


class PlainTextFormatter(PromptFormatter):
    """Plain text formatter"""

    def __init__(self):
        super().__init__()
        self.format_type = PromptFormat.PLAIN

    def format(self, structure: PromptStructure) -> str:
        """Format as plain text"""
        lines = []

        if structure.task:
            lines.append(structure.task)
            lines.append("")

        if structure.instructions:
            for instruction in structure.instructions:
                lines.append(instruction)
            lines.append("")

        if structure.context:
            for key, value in structure.context.items():
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def parse(self, text: str) -> PromptStructure:
        """Parse plain text (basic extraction)"""
        return PromptStructure(task=text)


class PromptFormatConverter:
    """
    Converts prompts between different formats

    Program of Thoughts:
    1. Parse source format to PromptStructure
    2. Format PromptStructure to target format
    3. Validate conversion
    """

    def __init__(self):
        self.formatters = {
            PromptFormat.XML: XMLPromptFormatter(),
            PromptFormat.JSON: JSONPromptFormatter(),
            PromptFormat.YAML: YAMLPromptFormatter(),
            PromptFormat.MARKDOWN: MarkdownPromptFormatter(),
            PromptFormat.POML: POMLPromptFormatter(),
            PromptFormat.PLAIN: PlainTextFormatter()
        }

    def convert(self, text: str, source_format: PromptFormat,
                target_format: PromptFormat) -> str:
        """
        Convert prompt from one format to another

        Args:
            text: Source prompt text
            source_format: Source format
            target_format: Target format

        Returns:
            Converted prompt text
        """
        # Parse source
        source_formatter = self.formatters[source_format]
        structure = source_formatter.parse(text)

        # Format to target
        target_formatter = self.formatters[target_format]
        return target_formatter.format(structure)

    def auto_detect_format(self, text: str) -> PromptFormat:
        """
        Auto-detect prompt format

        Args:
            text: Prompt text

        Returns:
            Detected format
        """
        text_strip = text.strip()

        # Check for XML
        if text_strip.startswith('<') and '<prompt' in text_strip:
            return PromptFormat.XML

        # Check for JSON
        if text_strip.startswith('{'):
            try:
                json.loads(text)
                return PromptFormat.JSON
            except:
                pass

        # Check for YAML
        if re.search(r'^\w+:', text_strip, re.MULTILINE):
            try:
                yaml.safe_load(text)
                return PromptFormat.YAML
            except:
                pass

        # Check for POML
        if re.search(r'\[(ROLE|TASK|INSTRUCTIONS)[\]:]', text):
            return PromptFormat.POML

        # Check for Markdown
        if re.search(r'^#+\s', text_strip, re.MULTILINE):
            return PromptFormat.MARKDOWN

        # Default to plain text
        return PromptFormat.PLAIN

    def optimize_for_provider(self, text: str, current_format: PromptFormat,
                             provider: str) -> str:
        """
        Optimize prompt format for specific LLM provider

        Different providers prefer different formats:
        - Claude/Anthropic: XML format works well
        - GPT-4: Markdown or structured text
        - Gemini: JSON or Markdown

        Args:
            text: Prompt text
            current_format: Current format
            provider: LLM provider name

        Returns:
            Optimized prompt text
        """
        provider_preferences = {
            'anthropic': PromptFormat.XML,
            'openai': PromptFormat.MARKDOWN,
            'gemini': PromptFormat.JSON,
            'cohere': PromptFormat.MARKDOWN,
            'mistral': PromptFormat.MARKDOWN,
            'ollama': PromptFormat.PLAIN,
            'lmstudio': PromptFormat.PLAIN
        }

        preferred_format = provider_preferences.get(provider, PromptFormat.PLAIN)

        if current_format != preferred_format:
            return self.convert(text, current_format, preferred_format)

        return text
