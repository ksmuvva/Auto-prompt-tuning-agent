"""
Interactive CLI Interface for Prompt Tuning AI Agent
Provides user-friendly command-line interaction
"""

import sys
import argparse
import logging
from typing import Optional
import json
from pathlib import Path

from agent.core import PromptTuningAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentCLI:
    """Command-line interface for the AI agent"""

    def __init__(self):
        self.agent: Optional[PromptTuningAgent] = None
        self.running = False

    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     PROMPT TUNING AI AGENT                                   ║
║     Automated Prompt Optimization for Data Analysis          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

AI Agent for bank transaction analysis with automated prompt tuning.

Type 'help' for available commands or 'quit' to exit.
"""
        print(banner)

    def print_help(self):
        """Print available commands"""
        help_text = """
AVAILABLE COMMANDS:
===================

SETUP & CONFIGURATION:
  init <provider>        Initialize agent with LLM provider
                         Providers: openai, anthropic, mock
                         Example: init openai

  config                 Show current configuration
  status                 Show agent status

DATA OPERATIONS:
  load                   Load and process CSV transaction data
  data-info              Show information about loaded data

PROMPT MANAGEMENT:
  list-prompts           List all available prompt templates
  add-prompt             Add a custom prompt template (interactive)
  show-prompt <name>     Show a specific prompt template

ANALYSIS & TUNING:
  analyze <mode>         Run analysis
                         Modes: quick, full, adaptive
                         Example: analyze adaptive

  quick-test             Quick test with 3 best prompts
  full-test              Test all available prompts
  adaptive-tune          Run adaptive AI optimization

RESULTS:
  results                Show latest results
  best-prompt            Show best performing prompt
  recommendations        Get AI recommendations
  export                 Export all results to files

AGENT INTERACTION:
  ask <question>         Ask the agent a question (AI reasoning)
  think <query>          Agent thinks about a query
  reset                  Reset agent state

UTILITY:
  help                   Show this help message
  clear                  Clear screen
  quit / exit            Exit the CLI

EXAMPLES:
  > init mock
  > load
  > analyze adaptive
  > best-prompt
  > ask "How can I improve my prompts?"
"""
        print(help_text)

    def initialize_agent(self, provider: str = "mock"):
        """Initialize the AI agent"""
        try:
            print(f"\nInitializing AI Agent with '{provider}' provider...")

            config = {}

            # Load config file if exists
            config_file = Path("config/config.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)

            self.agent = PromptTuningAgent(
                llm_provider=provider,
                data_dir="data",
                output_dir="results",
                config=config
            )

            print(f"✓ Agent initialized successfully!")
            print(f"  Provider: {provider}")
            print(f"  Data directory: data/")
            print(f"  Output directory: results/")

            return True

        except Exception as e:
            print(f"✗ Error initializing agent: {e}")
            logger.error(f"Agent initialization failed: {e}")
            return False

    def handle_command(self, command: str) -> bool:
        """
        Handle user command

        Returns: True to continue, False to exit
        """
        if not command.strip():
            return True

        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Commands that don't require initialized agent
        if cmd in ['help', 'h', '?']:
            self.print_help()
            return True

        if cmd in ['quit', 'exit', 'q']:
            print("\nGoodbye! Agent shutting down...")
            return False

        if cmd == 'clear':
            print("\033[2J\033[H")  # Clear screen
            return True

        if cmd == 'init':
            provider = args if args else "mock"
            self.initialize_agent(provider)
            return True

        # Commands that require initialized agent
        if not self.agent:
            print("⚠ Agent not initialized. Use 'init <provider>' first.")
            print("  Example: init mock")
            return True

        # Handle agent commands
        try:
            if cmd == 'config':
                print(json.dumps(self.agent.config, indent=2))

            elif cmd == 'status':
                status = self.agent.get_status()
                print("\n=== AGENT STATUS ===")
                print(f"State: {status['state']}")
                print(f"Templates Available: {status['templates_available']}")
                print(f"Best Prompt: {status['best_prompt']}")
                print(f"Best Score: {status['best_score']:.3f}")
                print(f"Memory: {status['memory_size']}")

            elif cmd == 'load':
                print("\nLoading transaction data...")
                results = self.agent.load_and_process_data()
                if results:
                    print(f"✓ Loaded {len(results.get('full_data', []))} transactions")
                    print(f"  High-value transactions (>250 GBP): {len(results.get('high_value_transactions', []))}")
                    print(f"  Statistical anomalies detected: {len(results.get('statistical_anomalies', []))}")

            elif cmd == 'data-info':
                if not self.agent.state['data_loaded']:
                    print("⚠ No data loaded. Use 'load' command first.")
                else:
                    results = self.agent.data_processor.process_all()
                    print("\n=== DATA INFORMATION ===")
                    print(json.dumps(results['validation_report'], indent=2, default=str))

            elif cmd == 'list-prompts':
                prompts = self.agent.template_library.list_templates()
                print(f"\n=== AVAILABLE PROMPTS ({len(prompts)}) ===")
                for i, prompt_name in enumerate(prompts, 1):
                    template = self.agent.template_library.get_template(prompt_name)
                    print(f"{i}. {prompt_name}")
                    if template.description:
                        print(f"   {template.description}")

            elif cmd == 'show-prompt':
                if not args:
                    print("Usage: show-prompt <name>")
                else:
                    template = self.agent.template_library.get_template(args)
                    if template:
                        print(f"\n=== PROMPT: {template.name} ===")
                        print(f"Description: {template.description}")
                        print(f"\nTemplate:\n{template.template}")
                    else:
                        print(f"Prompt '{args}' not found")

            elif cmd == 'add-prompt':
                print("\n=== ADD CUSTOM PROMPT ===")
                name = input("Prompt name: ").strip()
                description = input("Description: ").strip()
                print("Enter prompt template (use {data} and {threshold} as placeholders):")
                print("Type 'END' on a new line when finished:")

                template_lines = []
                while True:
                    line = input()
                    if line.strip() == 'END':
                        break
                    template_lines.append(line)

                template_text = '\n'.join(template_lines)

                if self.agent.add_custom_prompt(name, template_text, description):
                    print(f"✓ Custom prompt '{name}' added successfully!")
                else:
                    print("✗ Failed to add custom prompt")

            elif cmd == 'analyze':
                mode = args if args in ['quick', 'full', 'adaptive'] else 'quick'
                print(f"\nRunning analysis in '{mode}' mode...")
                print("This may take a few moments...\n")

                result = self.agent.run_analysis(mode=mode)

                if result['success']:
                    print(f"\n✓ Analysis complete!")
                    print(f"  Mode: {mode}")
                    print(f"  Best Prompt: {result['best_prompt']}")
                    print(f"  Best Score: {result['best_score']:.3f}")
                else:
                    print(f"✗ Analysis failed: {result.get('error')}")

            elif cmd == 'quick-test':
                print("\nRunning quick test...")
                result = self.agent.run_analysis(mode='quick')
                if result['success']:
                    print(f"✓ Best: {result['best_prompt']} (score: {result['best_score']:.3f})")

            elif cmd == 'full-test':
                print("\nRunning full test on all prompts...")
                result = self.agent.run_analysis(mode='full')
                if result['success']:
                    print(f"✓ Best: {result['best_prompt']} (score: {result['best_score']:.3f})")

            elif cmd == 'adaptive-tune':
                print("\nRunning adaptive tuning with AI optimization...")
                print("This will take several minutes...\n")
                result = self.agent.run_analysis(mode='adaptive')
                if result['success']:
                    print(f"✓ Adaptive tuning complete!")
                    print(f"  Best: {result['best_prompt']} (score: {result['best_score']:.3f})")

            elif cmd == 'results':
                if self.agent.best_results:
                    print("\n=== LATEST RESULTS ===")
                    print(json.dumps(self.agent.best_results, indent=2, default=str))
                else:
                    print("⚠ No results available. Run an analysis first.")

            elif cmd == 'best-prompt':
                if self.agent.prompt_tuner.best_prompt:
                    template = self.agent.prompt_tuner.get_best_prompt_template()
                    print(f"\n=== BEST PROMPT ===")
                    print(f"Name: {template.name}")
                    print(f"Score: {self.agent.prompt_tuner.best_score:.3f}")
                    print(f"\nTemplate:\n{template.template}")
                else:
                    print("⚠ No best prompt yet. Run an analysis first.")

            elif cmd == 'recommendations':
                recommendations = self.agent.get_recommendations()
                print("\n=== AI RECOMMENDATIONS ===")
                print(json.dumps(recommendations, indent=2, default=str))

            elif cmd == 'export':
                print("\nExporting results...")
                exports = self.agent.export_results()
                print("✓ Exported files:")
                for key, filepath in exports.items():
                    print(f"  {key}: {filepath}")

            elif cmd in ['ask', 'think']:
                if not args:
                    print(f"Usage: {cmd} <question>")
                else:
                    print(f"\n🤔 Agent thinking...\n")
                    response = self.agent.think(args)
                    print(f"🤖 Agent: {response}\n")

            elif cmd == 'reset':
                self.agent.reset()
                print("✓ Agent state reset")

            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")

        except Exception as e:
            print(f"✗ Error executing command: {e}")
            logger.error(f"Command execution error: {e}", exc_info=True)

        return True

    def run_interactive(self):
        """Run interactive CLI mode"""
        self.print_banner()
        self.running = True

        # Auto-initialize with mock provider
        print("Auto-initializing with mock provider for testing...")
        self.initialize_agent("mock")
        print("\nReady! Type 'help' for commands.\n")

        while self.running:
            try:
                command = input("agent> ").strip()
                if not self.handle_command(command):
                    break
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Unexpected error: {e}", exc_info=True)

    def run_command(self, command: str):
        """Run a single command (non-interactive)"""
        if not self.agent:
            self.initialize_agent("mock")
        self.handle_command(command)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Prompt Tuning AI Agent - Automated prompt optimization"
    )
    parser.add_argument(
        '-c', '--command',
        help='Execute a single command and exit'
    )
    parser.add_argument(
        '-p', '--provider',
        default='mock',
        choices=['openai', 'anthropic', 'mock'],
        help='LLM provider to use (default: mock)'
    )

    args = parser.parse_args()

    cli = AgentCLI()

    if args.command:
        # Single command mode
        cli.initialize_agent(args.provider)
        cli.run_command(args.command)
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()
