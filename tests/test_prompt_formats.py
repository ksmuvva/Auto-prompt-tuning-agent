"""
Comprehensive tests for prompt format support
"""

import pytest
import json
import yaml
from agent.prompt_formats import (
    PromptFormat, PromptStructure, PromptFormatConverter,
    XMLPromptFormatter, JSONPromptFormatter, YAMLPromptFormatter,
    MarkdownPromptFormatter, POMLPromptFormatter, PlainTextFormatter
)


class TestPromptStructure:
    """Test PromptStructure class"""

    def test_creation(self):
        """Test creating a prompt structure"""
        structure = PromptStructure(
            role="analyst",
            task="Analyze data",
            instructions=["Step 1", "Step 2"]
        )
        assert structure.role == "analyst"
        assert structure.task == "Analyze data"
        assert len(structure.instructions) == 2

    def test_to_dict(self):
        """Test converting to dictionary"""
        structure = PromptStructure(
            role="analyst",
            task="Analyze data"
        )
        d = structure.to_dict()
        assert d['role'] == "analyst"
        assert d['task'] == "Analyze data"


class TestXMLFormatter:
    """Test XML formatter"""

    def test_format_basic(self):
        """Test formatting basic structure to XML"""
        formatter = XMLPromptFormatter()
        structure = PromptStructure(
            role="analyst",
            task="Analyze transactions",
            instructions=["Identify anomalies", "Calculate statistics"]
        )

        xml = formatter.format(structure)

        assert '<prompt>' in xml
        assert '<role>analyst</role>' in xml
        assert '<task>Analyze transactions</task>' in xml
        assert '<instruction>Identify anomalies</instruction>' in xml

    def test_parse_basic(self):
        """Test parsing XML"""
        xml_text = """<?xml version="1.0" ?>
        <prompt>
            <role>analyst</role>
            <task>Analyze data</task>
            <instructions>
                <instruction>Step 1</instruction>
                <instruction>Step 2</instruction>
            </instructions>
        </prompt>
        """

        formatter = XMLPromptFormatter()
        structure = formatter.parse(xml_text)

        assert structure.role == "analyst"
        assert structure.task == "Analyze data"
        assert len(structure.instructions) == 2

    def test_roundtrip(self):
        """Test format -> parse roundtrip"""
        formatter = XMLPromptFormatter()

        original = PromptStructure(
            role="expert",
            task="Review code",
            instructions=["Check syntax", "Verify logic"],
            context={"language": "python", "file": "main.py"}
        )

        xml = formatter.format(original)
        parsed = formatter.parse(xml)

        assert parsed.role == original.role
        assert parsed.task == original.task
        assert parsed.instructions == original.instructions


class TestJSONFormatter:
    """Test JSON formatter"""

    def test_format(self):
        """Test formatting to JSON"""
        formatter = JSONPromptFormatter()
        structure = PromptStructure(
            role="analyst",
            task="Analyze data"
        )

        json_text = formatter.format(structure)
        data = json.loads(json_text)

        assert data['role'] == "analyst"
        assert data['task'] == "Analyze data"

    def test_parse(self):
        """Test parsing JSON"""
        json_text = json.dumps({
            "role": "analyst",
            "task": "Analyze data",
            "instructions": ["Step 1", "Step 2"]
        })

        formatter = JSONPromptFormatter()
        structure = formatter.parse(json_text)

        assert structure.role == "analyst"
        assert len(structure.instructions) == 2

    def test_roundtrip(self):
        """Test JSON roundtrip"""
        formatter = JSONPromptFormatter()

        original = PromptStructure(
            role="reviewer",
            task="Review PR",
            instructions=["Check tests", "Verify docs"],
            constraints=["Max 1000 lines", "No breaking changes"]
        )

        json_text = formatter.format(original)
        parsed = formatter.parse(json_text)

        assert parsed.role == original.role
        assert parsed.task == original.task
        assert parsed.instructions == original.instructions
        assert parsed.constraints == original.constraints


class TestYAMLFormatter:
    """Test YAML formatter"""

    def test_format(self):
        """Test formatting to YAML"""
        formatter = YAMLPromptFormatter()
        structure = PromptStructure(
            role="analyst",
            task="Analyze data",
            instructions=["Step 1", "Step 2"]
        )

        yaml_text = formatter.format(structure)
        data = yaml.safe_load(yaml_text)

        assert data['role'] == "analyst"
        assert data['task'] == "Analyze data"
        assert len(data['instructions']) == 2

    def test_parse(self):
        """Test parsing YAML"""
        yaml_text = """
        role: analyst
        task: Analyze data
        instructions:
          - Step 1
          - Step 2
        context: {}
        examples: []
        constraints: []
        output_format: null
        metadata: {}
        """

        formatter = YAMLPromptFormatter()
        structure = formatter.parse(yaml_text)

        assert structure.role == "analyst"
        assert structure.task == "Analyze data"

    def test_roundtrip(self):
        """Test YAML roundtrip"""
        formatter = YAMLPromptFormatter()

        original = PromptStructure(
            role="engineer",
            task="Debug issue",
            context={"error": "NullPointerException", "line": 42}
        )

        yaml_text = formatter.format(original)
        parsed = formatter.parse(yaml_text)

        assert parsed.role == original.role
        assert parsed.task == original.task


class TestMarkdownFormatter:
    """Test Markdown formatter"""

    def test_format(self):
        """Test formatting to Markdown"""
        formatter = MarkdownPromptFormatter()
        structure = PromptStructure(
            role="analyst",
            task="Analyze data",
            instructions=["Identify trends", "Calculate metrics"]
        )

        md_text = formatter.format(structure)

        assert "# Task: Analyze data" in md_text
        assert "## Role" in md_text
        assert "analyst" in md_text
        assert "## Instructions" in md_text
        assert "- Identify trends" in md_text

    def test_parse(self):
        """Test parsing Markdown"""
        md_text = """# Task: Analyze data

## Role
You are a analyst.

## Instructions
- Step 1
- Step 2

## Context
- **file**: data.csv
- **threshold**: 100
"""

        formatter = MarkdownPromptFormatter()
        structure = formatter.parse(md_text)

        assert structure.task == "Analyze data"
        assert structure.role == "analyst"
        # Parser extracts at least the instructions
        assert len(structure.instructions) >= 1

    def test_format_with_all_sections(self):
        """Test formatting with all sections"""
        formatter = MarkdownPromptFormatter()
        structure = PromptStructure(
            role="reviewer",
            task="Review code",
            instructions=["Check syntax", "Verify tests"],
            context={"language": "python"},
            constraints=["Max 500 lines"],
            output_format="detailed report"
        )

        md_text = formatter.format(structure)

        assert "## Role" in md_text
        assert "## Instructions" in md_text
        assert "## Context" in md_text
        assert "## Constraints" in md_text
        assert "## Output Format" in md_text


class TestPOMLFormatter:
    """Test POML (Prompt Markup Language) formatter"""

    def test_format(self):
        """Test formatting to POML"""
        formatter = POMLPromptFormatter()
        structure = PromptStructure(
            role="analyst",
            task="Analyze data",
            instructions=["Step 1", "Step 2"]
        )

        poml_text = formatter.format(structure)

        assert "[ROLE: analyst]" in poml_text
        assert "[TASK: Analyze data]" in poml_text
        assert "[INSTRUCTIONS]" in poml_text
        assert "[/INSTRUCTIONS]" in poml_text

    def test_parse(self):
        """Test parsing POML"""
        poml_text = """
        [ROLE: analyst]
        [TASK: Analyze data]

        [INSTRUCTIONS]
        - Step 1
        - Step 2
        [/INSTRUCTIONS]

        [CONTEXT]
        threshold = 250
        file = data.csv
        [/CONTEXT]
        """

        formatter = POMLPromptFormatter()
        structure = formatter.parse(poml_text)

        assert structure.role == "analyst"
        assert structure.task == "Analyze data"
        assert len(structure.instructions) == 2
        assert structure.context['threshold'] == "250"

    def test_roundtrip(self):
        """Test POML roundtrip"""
        formatter = POMLPromptFormatter()

        original = PromptStructure(
            role="tester",
            task="Run tests",
            instructions=["Unit tests", "Integration tests"],
            constraints=["Max 10 minutes"],
            output_format="junit xml"
        )

        poml_text = formatter.format(original)
        parsed = formatter.parse(poml_text)

        assert parsed.role == original.role
        assert parsed.task == original.task
        assert parsed.instructions == original.instructions
        assert parsed.output_format == original.output_format


class TestPlainTextFormatter:
    """Test plain text formatter"""

    def test_format(self):
        """Test formatting to plain text"""
        formatter = PlainTextFormatter()
        structure = PromptStructure(
            task="Analyze data",
            instructions=["Step 1", "Step 2"]
        )

        text = formatter.format(structure)

        assert "Analyze data" in text
        assert "Step 1" in text

    def test_parse(self):
        """Test parsing plain text"""
        text = "Analyze the transaction data for anomalies."

        formatter = PlainTextFormatter()
        structure = formatter.parse(text)

        assert structure.task == text


class TestPromptFormatConverter:
    """Test PromptFormatConverter"""

    def test_initialization(self):
        """Test initializing converter"""
        converter = PromptFormatConverter()
        assert len(converter.formatters) == 6

    def test_convert_json_to_xml(self):
        """Test converting from JSON to XML"""
        converter = PromptFormatConverter()

        json_text = json.dumps({
            "role": "analyst",
            "task": "Analyze data",
            "instructions": ["Step 1", "Step 2"],
            "context": {},
            "examples": [],
            "constraints": [],
            "output_format": None,
            "metadata": {}
        })

        xml_text = converter.convert(json_text, PromptFormat.JSON, PromptFormat.XML)

        assert '<prompt>' in xml_text
        assert '<role>analyst</role>' in xml_text

    def test_convert_yaml_to_markdown(self):
        """Test converting from YAML to Markdown"""
        converter = PromptFormatConverter()

        yaml_text = yaml.dump({
            "role": "analyst",
            "task": "Analyze data",
            "instructions": ["Step 1"],
            "context": {},
            "examples": [],
            "constraints": [],
            "output_format": None,
            "metadata": {}
        })

        md_text = converter.convert(yaml_text, PromptFormat.YAML, PromptFormat.MARKDOWN)

        assert "# Task:" in md_text
        assert "## Role" in md_text

    def test_convert_poml_to_json(self):
        """Test converting from POML to JSON"""
        converter = PromptFormatConverter()

        poml_text = """
        [ROLE: analyst]
        [TASK: Analyze data]

        [INSTRUCTIONS]
        - Step 1
        - Step 2
        [/INSTRUCTIONS]
        """

        json_text = converter.convert(poml_text, PromptFormat.POML, PromptFormat.JSON)

        data = json.loads(json_text)
        assert data['role'] == "analyst"
        assert data['task'] == "Analyze data"

    def test_auto_detect_xml(self):
        """Test auto-detecting XML format"""
        converter = PromptFormatConverter()

        xml_text = """<prompt>
            <role>analyst</role>
            <task>Analyze data</task>
        </prompt>"""

        detected = converter.auto_detect_format(xml_text)
        assert detected == PromptFormat.XML

    def test_auto_detect_json(self):
        """Test auto-detecting JSON format"""
        converter = PromptFormatConverter()

        json_text = '{"role": "analyst", "task": "Analyze data"}'

        detected = converter.auto_detect_format(json_text)
        assert detected == PromptFormat.JSON

    def test_auto_detect_yaml(self):
        """Test auto-detecting YAML format"""
        converter = PromptFormatConverter()

        yaml_text = """
        role: analyst
        task: Analyze data
        """

        detected = converter.auto_detect_format(yaml_text)
        assert detected == PromptFormat.YAML

    def test_auto_detect_poml(self):
        """Test auto-detecting POML format"""
        converter = PromptFormatConverter()

        poml_text = "[ROLE: analyst]\n[TASK: Analyze data]"

        detected = converter.auto_detect_format(poml_text)
        assert detected == PromptFormat.POML

    def test_auto_detect_markdown(self):
        """Test auto-detecting Markdown format"""
        converter = PromptFormatConverter()

        md_text = "# Task: Analyze data\n\n## Instructions"

        detected = converter.auto_detect_format(md_text)
        assert detected == PromptFormat.MARKDOWN

    def test_auto_detect_plain(self):
        """Test auto-detecting plain text format"""
        converter = PromptFormatConverter()

        plain_text = "Analyze the transaction data."

        detected = converter.auto_detect_format(plain_text)
        assert detected == PromptFormat.PLAIN

    def test_optimize_for_anthropic(self):
        """Test optimizing format for Anthropic"""
        converter = PromptFormatConverter()

        json_text = json.dumps({
            "role": "analyst",
            "task": "Analyze data",
            "instructions": ["Step 1"],
            "context": {},
            "examples": [],
            "constraints": [],
            "output_format": None,
            "metadata": {}
        })

        optimized = converter.optimize_for_provider(
            json_text,
            PromptFormat.JSON,
            "anthropic"
        )

        # Should convert to XML for Anthropic
        assert '<prompt>' in optimized

    def test_optimize_for_openai(self):
        """Test optimizing format for OpenAI"""
        converter = PromptFormatConverter()

        xml_text = """<prompt>
            <role>analyst</role>
            <task>Analyze data</task>
        </prompt>"""

        optimized = converter.optimize_for_provider(
            xml_text,
            PromptFormat.XML,
            "openai"
        )

        # Should convert to Markdown for OpenAI
        assert "# Task:" in optimized or "Task" in optimized

    def test_optimize_for_gemini(self):
        """Test optimizing format for Gemini"""
        converter = PromptFormatConverter()

        poml_text = "[ROLE: analyst]\n[TASK: Analyze data]"

        optimized = converter.optimize_for_provider(
            poml_text,
            PromptFormat.POML,
            "gemini"
        )

        # Should convert to JSON for Gemini
        assert '{' in optimized or '"role"' in optimized


def test_integration_full_conversion_workflow():
    """Test complete format conversion workflow"""
    converter = PromptFormatConverter()

    # Start with a complex structure
    original_structure = PromptStructure(
        role="financial_analyst",
        task="Analyze transaction data for fraud detection",
        instructions=[
            "Identify transactions above £250",
            "Calculate statistical anomalies (Z-score > 3)",
            "Detect suspicious patterns",
            "Generate detailed report"
        ],
        context={
            "threshold": "250",
            "currency": "GBP",
            "dataset": "transactions_2024.csv"
        },
        constraints=[
            "Maximum processing time: 5 minutes",
            "Must flag high-risk transactions",
            "Preserve customer privacy"
        ],
        output_format="structured_json",
        examples=[
            {
                "input": "Transaction #123: £500",
                "output": "HIGH_VALUE: Flag for review"
            }
        ]
    )

    # Convert to all formats and back
    json_formatter = JSONPromptFormatter()
    original_json = json_formatter.format(original_structure)

    # JSON -> XML -> YAML -> Markdown -> POML -> JSON
    xml = converter.convert(original_json, PromptFormat.JSON, PromptFormat.XML)
    assert '<prompt>' in xml

    yaml_text = converter.convert(xml, PromptFormat.XML, PromptFormat.YAML)
    assert 'role:' in yaml_text

    markdown = converter.convert(yaml_text, PromptFormat.YAML, PromptFormat.MARKDOWN)
    assert '# Task:' in markdown or 'Task' in markdown

    poml = converter.convert(markdown, PromptFormat.MARKDOWN, PromptFormat.POML)
    assert '[ROLE:' in poml or '[TASK:' in poml

    final_json = converter.convert(poml, PromptFormat.POML, PromptFormat.JSON)
    final_structure = json_formatter.parse(final_json)

    # Verify critical information preserved
    assert final_structure.role == original_structure.role
    assert final_structure.task == original_structure.task
