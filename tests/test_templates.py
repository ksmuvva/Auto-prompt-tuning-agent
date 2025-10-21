"""
Tests for Prompt Templates
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.templates import PromptTemplate, PromptTemplateLibrary


class TestPromptTemplate:
    """Test individual prompt templates"""

    def test_template_creation(self):
        """Test creating a template"""
        template = PromptTemplate(
            name="test_template",
            template="Analyze this: {data}",
            description="Test template"
        )

        assert template.name == "test_template"
        assert "{data}" in template.template
        assert template.description == "Test template"

    def test_template_formatting(self):
        """Test template formatting with variables"""
        template = PromptTemplate(
            name="test",
            template="Data: {data}, Threshold: {threshold}"
        )

        result = template.format(data="test_data", threshold=250)
        assert "test_data" in result
        assert "250" in result

    def test_missing_variable(self):
        """Test template with missing variable"""
        template = PromptTemplate(
            name="test",
            template="Data: {data}, Threshold: {threshold}"
        )

        # Should handle missing variable gracefully
        result = template.format(data="test_data")
        assert isinstance(result, str)


class TestPromptTemplateLibrary:
    """Test prompt template library"""

    def test_initialization(self):
        """Test library initializes with built-in templates"""
        library = PromptTemplateLibrary()
        templates = library.list_templates()

        assert len(templates) >= 8  # Should have at least 8 built-in templates
        assert 'direct_concise' in templates
        assert 'json_structured' in templates
        assert 'role_based_expert' in templates

    def test_get_template(self):
        """Test retrieving a template"""
        library = PromptTemplateLibrary()
        template = library.get_template('direct_concise')

        assert template is not None
        assert template.name == 'direct_concise'
        assert len(template.template) > 0

    def test_get_nonexistent_template(self):
        """Test retrieving non-existent template"""
        library = PromptTemplateLibrary()
        template = library.get_template('nonexistent')

        assert template is None

    def test_format_template(self):
        """Test formatting a template from library"""
        library = PromptTemplateLibrary()
        result = library.format_template(
            'direct_concise',
            data="test data",
            threshold=250
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "test data" in result

    def test_create_custom_template(self):
        """Test creating custom template"""
        library = PromptTemplateLibrary()
        initial_count = len(library.list_templates())

        custom = library.create_custom_template(
            name="custom_test",
            template_text="Custom: {data}",
            description="Custom template"
        )

        assert custom.name == "custom_test"
        assert len(library.list_templates()) == initial_count + 1
        assert 'custom_test' in library.list_templates()

    def test_get_all_templates(self):
        """Test getting all templates"""
        library = PromptTemplateLibrary()
        all_templates = library.get_all_templates()

        assert isinstance(all_templates, list)
        assert len(all_templates) >= 8
        assert all(isinstance(t, PromptTemplate) for t in all_templates)

    def test_template_variables(self):
        """Test templates have correct variables"""
        library = PromptTemplateLibrary()
        template = library.get_template('direct_concise')

        assert 'data' in template.variables
        assert 'threshold' in template.variables


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
