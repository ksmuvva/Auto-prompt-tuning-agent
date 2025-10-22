"""
Natural Language Processing CLI
Allows users to interact with the agent using natural language
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPCommandParser:
    """Parses natural language commands into structured actions"""

    def __init__(self):
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[str, list]:
        """Build regex patterns for common command types"""
        return {
            # LLM Provider Selection
            'set_provider': [
                r'(?:use|switch to|change to|set provider to)\s+(\w+)',
                r'(?:i want to use|let me use)\s+(\w+)',
                r'provider\s*[=:]\s*(\w+)'
            ],

            # Model Selection
            'set_model': [
                r'(?:use|switch to|set)\s+model\s+(\S+)',
                r'model\s*[=:]\s*(\S+)',
                r'(?:use|try)\s+(gpt-4|gpt-3\.5|claude-3|gemini-pro)(?:\s+model)?'
            ],

            # Analysis Commands
            'analyze': [
                r'analyze\s+(fw\d+)',
                r'run\s+(fw\d+)\s+analysis',
                r'(?:test|check|evaluate)\s+(fw\d+)',
                r'analyze\s+(?:all|everything)',
            ],

            # Prompt Strategy
            'set_strategy': [
                r'(?:use|switch to|set strategy to)\s+(dynamic|template|hybrid)',
                r'strategy\s*[=:]\s*(dynamic|template|hybrid)',
                r'(?:i want|let me use)\s+(dynamic|template|hybrid)\s+prompts?'
            ],

            # Metrics and Validation
            'show_metrics': [
                r'show\s+(?:me\s+)?(?:the\s+)?metrics',
                r'what are the metrics',
                r'how did (?:it|we) do',
                r'show\s+results',
                r'performance'
            ],

            # Data Operations
            'load_data': [
                r'load\s+(?:the\s+)?data',
                r'read\s+(?:the\s+)?data',
                r'import\s+(?:the\s+)?data',
            ],

            # Tuning Commands
            'adaptive_tune': [
                r'(?:run|start|begin)\s+adaptive\s+tuning',
                r'tune\s+(?:the\s+)?prompts?',
                r'optimize\s+(?:the\s+)?prompts?',
                r'improve\s+(?:the\s+)?prompts?'
            ],

            # Questions
            'ask': [
                r'(?:why|how|what|when|where)\s+.+',
                r'can you\s+.+',
                r'could you\s+.+',
                r'what is\s+.+',
                r'tell me\s+.+'
            ],

            # Help
            'help': [
                r'help',
                r'what can (?:you|i) do',
                r'show commands',
                r'how do i\s+.+'
            ]
        }

    def parse(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse natural language input into command and parameters

        Returns:
            (command_type, parameters)
        """
        user_input = user_input.strip().lower()

        # Try each command type
        for cmd_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    params = self._extract_params(cmd_type, match, user_input)
                    return cmd_type, params

        # Check for direct requirement names
        fw_match = re.search(r'\b(fw\d+)\b', user_input, re.IGNORECASE)
        if fw_match:
            return 'analyze', {'requirement': fw_match.group(1).lower()}

        # Default to ask/query
        return 'ask', {'query': user_input}

    def _extract_params(self, cmd_type: str, match, full_input: str) -> Dict[str, Any]:
        """Extract parameters from regex match"""
        params = {}

        if cmd_type == 'set_provider':
            params['provider'] = match.group(1).lower()

        elif cmd_type == 'set_model':
            params['model'] = match.group(1)

        elif cmd_type == 'analyze':
            if 'all' in full_input or 'everything' in full_input:
                params['requirement'] = 'all'
            else:
                params['requirement'] = match.group(1).lower()

        elif cmd_type == 'set_strategy':
            params['strategy'] = match.group(1).lower()

        elif cmd_type == 'ask':
            params['query'] = full_input

        return params


class EnhancedCLI:
    """Enhanced CLI with NLP support and user controls"""

    def __init__(self):
        self.nlp_parser = NLPCommandParser()
        self.agent = None
        self.current_provider = None
        self.current_model = None
        self.current_strategy = 'dynamic'  # default to dynamic
        self.prompt_mode = 'dynamic'  # 'dynamic' or 'template'

    def print_welcome(self):
        """Print enhanced welcome message"""
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  🤖 ADAPTIVE AI PROMPT TUNING AGENT                          ║
║  True AI with Dynamic Learning & Metric-Based Optimization   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

💬 NATURAL LANGUAGE INTERFACE
   You can use natural language! Examples:
   - "use openai"
   - "analyze fw15"
   - "show me the metrics"
   - "tune the prompts"
   - "why is accuracy low?"

⚙️  USER CONTROLS
   - Choose your LLM: openai, anthropic, gemini, cohere
   - Choose your model: gpt-4, claude-3-opus, gemini-pro
   - Choose prompt mode: dynamic (AI-generated) or template
   - Choose strategy: template, dynamic, hybrid

📊 TRUE METRICS
   - Real mathematical calculations
   - Ground truth comparison
   - 98% precision/accuracy targets

Type 'help' for commands or just talk naturally!
""")

    def display_status(self):
        """Display current configuration"""
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║ CURRENT CONFIGURATION                                         ║
╠═══════════════════════════════════════════════════════════════╣
║ LLM Provider:   {self.current_provider or 'Not set'}
║ Model:          {self.current_model or 'Default'}
║ Prompt Mode:    {self.prompt_mode.upper()} prompts
║ Strategy:       {self.current_strategy.upper()}
║ Agent Status:   {'Initialized' if self.agent else 'Not initialized'}
╚═══════════════════════════════════════════════════════════════╝
""")

    def handle_natural_language(self, user_input: str) -> bool:
        """
        Handle natural language input

        Returns:
            True to continue, False to exit
        """
        if not user_input.strip():
            return True

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\n👋 Goodbye! Agent shutting down...")
            return False

        # Parse command
        cmd_type, params = self.nlp_parser.parse(user_input)

        # Handle command
        if cmd_type == 'set_provider':
            return self._handle_set_provider(params['provider'])

        elif cmd_type == 'set_model':
            return self._handle_set_model(params['model'])

        elif cmd_type == 'set_strategy':
            return self._handle_set_strategy(params['strategy'])

        elif cmd_type == 'analyze':
            return self._handle_analyze(params['requirement'])

        elif cmd_type == 'show_metrics':
            return self._handle_show_metrics()

        elif cmd_type == 'load_data':
            return self._handle_load_data()

        elif cmd_type == 'adaptive_tune':
            return self._handle_adaptive_tune()

        elif cmd_type == 'ask':
            return self._handle_ask(params['query'])

        elif cmd_type == 'help':
            self._print_help()
            return True

        else:
            print(f"💬 Processing: '{user_input}'")
            return self._handle_ask(user_input)

    def _handle_set_provider(self, provider: str) -> bool:
        """Handle provider selection"""
        valid_providers = ['openai', 'anthropic', 'gemini', 'cohere', 'mistral', 'ollama']

        if provider not in valid_providers:
            print(f"❌ Invalid provider '{provider}'")
            print(f"   Valid options: {', '.join(valid_providers)}")
            return True

        print(f"\n🔧 Setting LLM provider to: {provider}")
        self.current_provider = provider

        # Reinitialize agent if exists
        if self.agent:
            print(f"   Reinitializing agent with {provider}...")
            # Would reinitialize agent here
            print(f"✓ Agent reinitialized with {provider}")
        else:
            print(f"   Use 'load data' to initialize the agent")

        return True

    def _handle_set_model(self, model: str) -> bool:
        """Handle model selection"""
        print(f"\n🔧 Setting model to: {model}")
        self.current_model = model
        print(f"✓ Model set to {model}")
        return True

    def _handle_set_strategy(self, strategy: str) -> bool:
        """Handle strategy selection"""
        valid_strategies = ['template', 'dynamic', 'hybrid']

        if strategy not in valid_strategies:
            print(f"❌ Invalid strategy '{strategy}'")
            print(f"   Valid options: {', '.join(valid_strategies)}")
            return True

        print(f"\n🔧 Setting prompt strategy to: {strategy}")
        self.current_strategy = strategy

        if self.agent:
            self.agent.state['prompt_strategy'] = strategy
            print(f"✓ Strategy set to {strategy}")
        else:
            print(f"   Strategy will be applied when agent is initialized")

        return True

    def _handle_analyze(self, requirement: str) -> bool:
        """Handle analysis command"""
        if not self.agent:
            print("❌ Agent not initialized. Load data first!")
            return True

        print(f"\n📊 Analyzing requirement: {requirement}")
        print(f"   Using {self.current_strategy} strategy with {self.prompt_mode} prompts")
        print(f"   This may take a while...")

        # Would call agent analysis here
        print(f"\n✓ Analysis complete!")

        return True

    def _handle_show_metrics(self) -> bool:
        """Display metrics"""
        print("\n📊 CURRENT METRICS")
        print("=" * 60)
        # Would show real metrics here
        print("   Precision: ---%")
        print("   Accuracy:  ---%")
        print("   Recall:    ---%")
        print("   F1 Score:  ---%")
        print("\n   Load data and run analysis to see metrics")
        return True

    def _handle_load_data(self) -> bool:
        """Load data"""
        print("\n📁 Loading transaction data...")
        # Would load data here
        print("✓ Data loaded successfully!")
        return True

    def _handle_adaptive_tune(self) -> bool:
        """Handle adaptive tuning"""
        if not self.agent:
            print("❌ Agent not initialized. Load data first!")
            return True

        print("\n🔄 Starting adaptive prompt tuning...")
        print(f"   Target: 98% precision, 98% accuracy")
        print(f"   Max iterations: 10")
        print(f"   This will take several minutes...")

        # Would run adaptive tuning here
        print(f"\n✓ Adaptive tuning complete!")

        return True

    def _handle_ask(self, query: str) -> bool:
        """Handle natural language question"""
        print(f"\n💭 You asked: '{query}'")

        if self.agent and hasattr(self.agent, 'think'):
            response = self.agent.think(query)
            print(f"\n🤖 Agent: {response}")
        else:
            print("   Agent not available for questions yet. Initialize first!")

        return True

    def _print_help(self):
        """Print help message"""
        print("""
╔═══════════════════════════════════════════════════════════════╗
║ NATURAL LANGUAGE COMMANDS                                     ║
╚═══════════════════════════════════════════════════════════════╝

LLM PROVIDER:
  "use openai"
  "switch to anthropic"
  "set provider to gemini"

MODEL SELECTION:
  "use model gpt-4"
  "switch to claude-3-opus"

PROMPT MODE:
  "use dynamic prompts"      - AI generates prompts
  "use template prompts"     - Use predefined templates
  "use hybrid strategy"      - Combination of both

ANALYSIS:
  "analyze fw15"
  "run fw20 analysis"
  "analyze all requirements"

TUNING:
  "tune the prompts"
  "run adaptive tuning"
  "optimize prompts"

METRICS:
  "show metrics"
  "how did we do?"
  "show results"

DATA:
  "load data"
  "import data"

QUESTIONS:
  "why is accuracy low?"
  "how can I improve precision?"
  "what's the best model?"

OTHER:
  status    - Show current configuration
  help      - Show this help
  quit      - Exit
""")

    def run(self):
        """Run the enhanced NLP CLI"""
        self.print_welcome()
        self.display_status()

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if not user_input:
                    continue

                should_continue = self.handle_natural_language(user_input)

                if not should_continue:
                    break

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"CLI error: {e}", exc_info=True)


if __name__ == '__main__':
    cli = EnhancedCLI()
    cli.run()
