"""
Natural Language CLI Interface for Synthetic Data Generator

Interactive CLI that understands natural language commands
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional, Dict, Any
from core.llm_providers import LLMFactory
from core.intent_engine import IntentEngine, Intent
from core.ambiguity_detector import AmbiguityDetector
from core.reasoning_engines import ReasoningEngineFactory, GenerationResult
from core.uk_standards import UKStandardsGenerator, UKStandardsEnforcer
from core.pattern_learner import PatternLearner
from core.output_engine import OutputEngine, BatchOutputEngine


class SyntheticDataCLI:
    """Natural language CLI for synthetic data generation"""

    def __init__(self):
        self.llm = None
        self.intent_engine = None
        self.ambiguity_detector = None
        self.pattern_learner = None
        self.uk_standards = UKStandardsGenerator()
        self.uk_enforcer = UKStandardsEnforcer()
        self.output_engine = OutputEngine()
        self.batch_output = BatchOutputEngine()

        self.current_provider = None
        self.current_model = None

    def start(self):
        """Start the interactive CLI"""
        self.print_banner()

        # Initialize with default provider (mock for testing)
        self.initialize_provider('mock')

        print("\nWelcome to the Advanced Synthetic Data Generator!")
        print("Type 'help' for examples or start with a data generation request.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() in ['help', '?']:
                    self.show_help()
                elif user_input.lower().startswith('use '):
                    self.handle_provider_change(user_input)
                else:
                    self.handle_generation_request(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Try rephrasing your request or type 'help' for examples.\n")

    def print_banner(self):
        """Print CLI banner"""
        banner = """
╔════════════════════════════════════════════════════════════╗
║     Advanced Synthetic Data Generator                      ║
║     World-Class, LLM-Agnostic, UK-Compliant                ║
╚════════════════════════════════════════════════════════════╝
"""
        print(banner)

    def show_help(self):
        """Show help information"""
        help_text = """
Available Commands:

Provider Management:
  use openai              - Switch to OpenAI (requires API key)
  use anthropic           - Switch to Anthropic Claude
  use gemini              - Switch to Google Gemini
  use mock                - Use mock provider (no API key needed)

Generation Requests (Natural Language):
  Generate 1000 UK customer records for e-commerce testing
  Create 500 patient records for healthcare ML training in JSON format
  Make 100 employee records with UK addresses in Excel format
  Generate transaction data for banking system testing

Options you can specify:
  - Number of records (e.g., "1000", "500")
  - Geography (e.g., "UK", "US", "EU")
  - Purpose (e.g., "testing", "training", "demo")
  - Format (e.g., "CSV", "JSON", "Excel", "PDF", "Word", "Markdown")
  - Domain (e.g., "e-commerce", "healthcare", "finance")

Examples:
  You: Generate 1000 UK customer records
  You: Create 500 patients for ML training in JSON
  You: Make employee data with salaries in Excel
  You: Generate banking transactions for testing

Other Commands:
  help / ?                - Show this help
  exit / quit             - Exit the program
"""
        print(help_text)

    def initialize_provider(self, provider: str, model: Optional[str] = None):
        """Initialize LLM provider and engines"""
        try:
            self.llm = LLMFactory.create(provider, model=model)
            self.current_provider = provider
            self.current_model = model or 'default'

            # Initialize engines
            self.intent_engine = IntentEngine(self.llm)
            self.ambiguity_detector = AmbiguityDetector(self.llm)
            self.pattern_learner = PatternLearner(self.llm)

            print(f"✓ Initialized {provider} provider")

        except Exception as e:
            print(f"❌ Failed to initialize {provider}: {e}")
            print("Falling back to mock provider...")
            self.initialize_provider('mock')

    def handle_provider_change(self, user_input: str):
        """Handle provider change command"""
        parts = user_input.lower().split()
        if len(parts) >= 2:
            provider = parts[1]
            self.initialize_provider(provider)
        else:
            print("Usage: use <provider>")
            print("Available providers:", LLMFactory.list_providers())

    def handle_generation_request(self, user_input: str):
        """Handle data generation request"""
        print("\n🧠 Analyzing your request...")

        # Parse intent
        intent = self.intent_engine.parse_intent(user_input)

        print(f"\n📋 Intent Detected:")
        print(f"  • Data Type: {intent.data_type}")
        print(f"  • Count: {intent.count}")
        print(f"  • Geography: {intent.geography or 'Not specified'}")
        print(f"  • Purpose: {intent.purpose or 'Not specified'}")
        print(f"  • Domain: {intent.domain or 'Not specified'}")
        print(f"  • Format: {intent.output_format or 'CSV (default)'}")

        # Detect ambiguities
        clarifications = self.ambiguity_detector.detect_ambiguities(intent)

        if clarifications:
            print(f"\n❓ I have {len(clarifications)} question(s) to clarify:\n")
            print(self.ambiguity_detector.format_clarifications(clarifications))

            # For now, use defaults
            print("\n⏩ Using defaults for this demo...\n")
            answers = {}
            intent = self.ambiguity_detector.resolve_clarifications(intent, clarifications, answers)

        # Get schema
        print("📐 Generating schema...")
        schema = self.intent_engine.get_schema_suggestion(intent)
        print(f"  Fields: {', '.join(schema.keys())}")

        # Select reasoning engine
        reasoning_engine = 'monte_carlo'  # Default
        print(f"\n⚙️  Using {reasoning_engine} reasoning engine...")

        # Generate data
        print(f"🎲 Generating {intent.count} records...")
        generated_data = self.generate_data(intent, schema, reasoning_engine)

        # Apply UK standards if UK geography
        if intent.geography and intent.geography.upper() == 'UK':
            print("🇬🇧 Applying UK standards...")
            generated_data = [
                self.uk_enforcer.enforce_standards(record, schema)
                for record in generated_data
            ]

        # Export data
        output_format = intent.output_format or 'csv'
        output_path = f"generated_data_{intent.entity.lower()}"

        print(f"\n💾 Exporting to {output_format.upper()}...")

        metadata = {
            'title': f'Generated {intent.data_type.title()}',
            'generated_date': self.uk_standards.generate_random_date(2025, 2025),
            'geography': intent.geography,
            'purpose': intent.purpose,
            'record_count': len(generated_data)
        }

        try:
            final_path = self.output_engine.export(
                generated_data,
                output_path,
                output_format,
                metadata
            )
            print(f"✅ Success! Data exported to: {final_path}")
            print(f"   Records: {len(generated_data)}")
            print(f"   Format: {output_format.upper()}")

        except Exception as e:
            print(f"❌ Export failed: {e}")

        print()  # Blank line

    def generate_data(self, intent: Intent, schema: Dict[str, str], reasoning_type: str) -> list:
        """Generate synthetic data"""

        # Create reasoning engine
        engine = ReasoningEngineFactory.create(reasoning_type, self.llm)

        # Generate data
        results = engine.generate(intent, schema, count=intent.count)

        # Extract data from results
        data = [result.data for result in results]

        return data


def main():
    """Main entry point"""
    cli = SyntheticDataCLI()
    cli.start()


if __name__ == '__main__':
    main()
