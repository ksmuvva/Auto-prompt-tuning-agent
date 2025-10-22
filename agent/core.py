"""
AI Agent Core
Main intelligent agent with memory, reasoning, and autonomous capabilities
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import time

from agent.llm_service import LLMService
from agent.data_processor import TransactionDataProcessor
from agent.prompt_tuner import PromptTuner
from agent.metrics import PromptMetrics
from prompts.templates import PromptTemplateLibrary
from agent.ground_truth import GroundTruthManager
from agent.requirement_analyzer import RequirementAnalyzer
from agent.dynamic_prompts import DynamicPromptGenerator
from agent.comparative import ComparativeAnalyzer
from agent.bias_detector import BiasDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMemory:
    """Agent's memory system for storing context and learning"""

    def __init__(self, memory_file: str = "logs/agent_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(exist_ok=True)

        self.short_term = []  # Recent interactions
        self.long_term = {}   # Persistent knowledge
        self.learned_patterns = []  # Patterns learned over time

        self.load_memory()

    def load_memory(self):
        """Load memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.long_term = data.get('long_term', {})
                    self.learned_patterns = data.get('learned_patterns', [])
                    logger.info("Loaded agent memory from disk")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")

    def save_memory(self):
        """Save memory to disk"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'long_term': self.long_term,
                    'learned_patterns': self.learned_patterns,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2, default=str)
            logger.debug("Saved agent memory to disk")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def add_interaction(self, interaction: Dict[str, Any]):
        """Add to short-term memory"""
        self.short_term.append({
            'timestamp': datetime.now().isoformat(),
            'data': interaction
        })

        # Keep only last 50 interactions in short-term
        if len(self.short_term) > 50:
            self.short_term = self.short_term[-50:]

    def store_knowledge(self, key: str, value: Any):
        """Store in long-term memory"""
        self.long_term[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        self.save_memory()

    def recall(self, key: str) -> Optional[Any]:
        """Recall from long-term memory"""
        if key in self.long_term:
            return self.long_term[key]['value']
        return None

    def learn_pattern(self, pattern: Dict[str, Any]):
        """Learn a new pattern"""
        self.learned_patterns.append({
            'pattern': pattern,
            'learned_at': datetime.now().isoformat()
        })
        self.save_memory()


class PromptTuningAgent:
    """
    Main AI Agent for Automated Prompt Tuning

    This agent autonomously:
    - Analyzes bank transaction data
    - Tests multiple prompt strategies
    - Learns from results and optimizes prompts
    - Provides intelligent recommendations
    """

    def __init__(
        self,
        llm_provider: str = "mock",
        data_dir: str = "data",
        output_dir: str = "results",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the AI agent"""
        logger.info("Initializing Prompt Tuning AI Agent...")

        self.config = config or {}
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Initialize components
        self.memory = AgentMemory()
        self.llm_service = LLMService(provider=llm_provider, **self.config.get('llm', {}))
        self.data_processor = TransactionDataProcessor(data_dir=data_dir)
        self.template_library = PromptTemplateLibrary()
        self.prompt_tuner = PromptTuner(
            self.llm_service,
            self.template_library,
            output_dir=output_dir
        )

        # === NEW COMPONENTS FOR FW REQUIREMENTS ===
        self.ground_truth_manager = GroundTruthManager()
        self.requirement_analyzer = RequirementAnalyzer(data_dir=data_dir)
        self.dynamic_prompt_generator = DynamicPromptGenerator(self.llm_service)
        self.comparative_analyzer = ComparativeAnalyzer()
        self.bias_detector = BiasDetector()

        # Agent state
        self.state = {
            'initialized': True,
            'data_loaded': False,
            'tuning_active': False,
            'current_task': None,
            'prompt_strategy': self.config.get('prompt_strategy', 'template'),  # template | dynamic | hybrid
            'validation_enabled': True,
            'bias_checking_enabled': True
        }

        # Store best results
        self.best_results = None

        logger.info(f"Agent initialized with {llm_provider} provider")
        self.memory.store_knowledge('initialization', {
            'provider': llm_provider,
            'timestamp': datetime.now().isoformat()
        })

    def load_and_process_data(self) -> Dict[str, Any]:
        """Load and process transaction data"""
        logger.info("Loading and processing transaction data...")

        self.state['current_task'] = 'data_processing'

        results = self.data_processor.process_all()

        if results:
            self.state['data_loaded'] = True
            self.memory.store_knowledge('last_data_load', {
                'records': len(results.get('full_data', [])),
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"Loaded {len(results.get('full_data', []))} transactions")

        self.state['current_task'] = None
        return results

    def run_analysis(
        self,
        mode: str = "quick",
        threshold: float = 250.0
    ) -> Dict[str, Any]:
        """
        Run transaction analysis

        Args:
            mode: 'quick' (test few prompts), 'full' (test all), 'adaptive' (AI optimization)
            threshold: Transaction amount threshold in GBP
        """
        logger.info(f"Starting analysis in '{mode}' mode...")

        self.state['tuning_active'] = True
        self.state['current_task'] = f'analysis_{mode}'

        # Load data if not already loaded
        if not self.state['data_loaded']:
            data_results = self.load_and_process_data()
        else:
            data_results = self.data_processor.process_all()

        # Prepare data for LLM
        llm_data = data_results['llm_formatted_data']
        ground_truth = data_results['ground_truth']

        # Run appropriate tuning mode
        if mode == "quick":
            logger.info("Quick mode: Testing 3 best-performing prompt templates")
            templates_to_test = ['direct_concise', 'json_structured', 'role_based_expert']

            results = []
            for template_name in templates_to_test:
                metrics = self.prompt_tuner.test_single_prompt(
                    template_name,
                    llm_data,
                    ground_truth,
                    threshold
                )
                results.append(metrics)

            comparison = self.prompt_tuner.metrics_evaluator.compare_prompts(results)

            tuning_results = {
                'mode': 'quick',
                'prompts_tested': len(results),
                'metrics': results,
                'comparison': comparison
            }

        elif mode == "full":
            logger.info("Full mode: Testing all available prompt templates")
            iteration_result = self.prompt_tuner.run_tuning_iteration(
                llm_data,
                ground_truth,
                threshold,
                test_all=True
            )
            tuning_results = iteration_result

        elif mode == "adaptive":
            logger.info("Adaptive mode: AI-powered prompt optimization")
            tuning_results = self.prompt_tuner.run_adaptive_tuning(
                llm_data,
                ground_truth,
                threshold,
                max_iterations=self.config.get('max_iterations', 3),
                target_score=self.config.get('target_score', 0.85)
            )

        else:
            logger.error(f"Unknown mode: {mode}")
            return {'success': False, 'error': f'Unknown mode: {mode}'}

        # Store results
        self.best_results = tuning_results
        self.memory.store_knowledge('last_analysis', {
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'best_score': self.prompt_tuner.best_score
        })

        # Learn patterns
        if self.prompt_tuner.best_prompt:
            self.memory.learn_pattern({
                'best_prompt': self.prompt_tuner.best_prompt,
                'score': self.prompt_tuner.best_score,
                'mode': mode
            })

        self.state['tuning_active'] = False
        self.state['current_task'] = None

        return {
            'success': True,
            'mode': mode,
            'results': tuning_results,
            'best_prompt': self.prompt_tuner.best_prompt,
            'best_score': self.prompt_tuner.best_score
        }

    def add_custom_prompt(self, name: str, template_text: str, description: str = "") -> bool:
        """Add a custom prompt template"""
        try:
            self.template_library.create_custom_template(name, template_text, description)
            self.memory.store_knowledge(f'custom_prompt_{name}', {
                'description': description,
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"Added custom prompt: {name}")
            return True
        except Exception as e:
            logger.error(f"Error adding custom prompt: {e}")
            return False

    def get_recommendations(self) -> Dict[str, Any]:
        """Get AI agent recommendations"""
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'suggestions': []
        }

        if self.best_results:
            best_prompt = self.prompt_tuner.best_prompt
            best_score = self.prompt_tuner.best_score

            recommendations['best_prompt'] = best_prompt
            recommendations['best_score'] = best_score

            if best_score < 0.7:
                recommendations['suggestions'].append(
                    "Consider running adaptive mode for AI-optimized prompts"
                )
            elif best_score < 0.85:
                recommendations['suggestions'].append(
                    "Good performance! Run adaptive mode to further optimize"
                )
            else:
                recommendations['suggestions'].append(
                    "Excellent performance! Current prompt is highly effective"
                )

        # Check learned patterns
        if self.memory.learned_patterns:
            recent_patterns = self.memory.learned_patterns[-5:]
            avg_score = sum(p['pattern'].get('score', 0) for p in recent_patterns) / len(recent_patterns)
            recommendations['average_historical_score'] = avg_score

            if avg_score > best_score if best_score else 0:
                recommendations['suggestions'].append(
                    "Historical patterns show potential for improvement - try different prompt strategies"
                )

        return recommendations

    def export_results(self) -> Dict[str, str]:
        """Export all results"""
        exports = {}

        # Export tuning results
        if self.prompt_tuner.tuning_history:
            results_file = self.prompt_tuner.export_results()
            exports['tuning_results'] = results_file

        # Export best prompt
        best_prompt_file = self.prompt_tuner.save_best_prompt()
        if best_prompt_file:
            exports['best_prompt'] = best_prompt_file

        # Export metrics
        metrics_file = str(Path(self.output_dir) / f"metrics_{int(time.time())}.json")
        self.prompt_tuner.metrics_evaluator.export_metrics(metrics_file)
        exports['metrics'] = metrics_file

        logger.info(f"Exported {len(exports)} result files")
        return exports

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'state': self.state,
            'memory_size': {
                'short_term': len(self.memory.short_term),
                'long_term': len(self.memory.long_term),
                'learned_patterns': len(self.memory.learned_patterns)
            },
            'llm_stats': self.llm_service.get_usage_stats(),
            'templates_available': len(self.template_library.list_templates()),
            'best_prompt': self.prompt_tuner.best_prompt,
            'best_score': self.prompt_tuner.best_score
        }

    def think(self, query: str) -> str:
        """
        Agent's reasoning capability - use LLM to reason about queries

        This demonstrates true AI agent capability
        """
        logger.info(f"Agent thinking about: {query}")

        # Gather context from memory
        status = self.get_status()
        recent_memory = self.memory.short_term[-5:] if self.memory.short_term else []

        thinking_prompt = f"""You are an AI agent specialized in prompt optimization for data analysis.

CURRENT STATUS:
- Best prompt so far: {status.get('best_prompt', 'None')}
- Best score: {status.get('best_score', 0):.2f}
- Templates available: {status.get('templates_available', 0)}
- Patterns learned: {status['memory_size']['learned_patterns']}

USER QUERY: {query}

Provide a thoughtful, helpful response based on your expertise in prompt engineering and your current knowledge.
"""

        result = self.llm_service.generate(thinking_prompt)

        if result.get('success'):
            return result['response']
        else:
            return "I'm having trouble processing that query. Please try rephrasing."

    def reset(self):
        """Reset agent state (but keep memory)"""
        logger.info("Resetting agent state...")
        self.best_results = None
        self.prompt_tuner.best_prompt = None
        self.prompt_tuner.best_score = 0.0
        self.prompt_tuner.tuning_history = []
        self.state['data_loaded'] = False
        self.state['tuning_active'] = False
        logger.info("Agent reset complete")

    # === NEW METHODS FOR FW REQUIREMENTS ===

    def analyze_fw_requirement(
        self,
        requirement: str,
        use_dynamic_prompt: bool = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a specific FW requirement

        Args:
            requirement: FW requirement code (e.g., 'fw15', 'fw20', 'fw45')
            use_dynamic_prompt: Use dynamic prompt generation instead of templates
            validate: Validate against ground truth

        Returns:
            Analysis results with metrics
        """
        logger.info(f"Analyzing {requirement.upper()}...")

        if use_dynamic_prompt is None:
            use_dynamic_prompt = self.state['prompt_strategy'] == 'dynamic'

        # Get data from requirement analyzer
        req_lower = requirement.lower()

        if req_lower == 'fw15':
            ground_truth_data = self.requirement_analyzer.analyze_fw15_high_value(threshold=250)
        elif req_lower in ['fw20', 'fw20-luxury', 'fw20-transfer']:
            ground_truth_data = self.requirement_analyzer.analyze_fw20_similar_transactions(threshold=250)
        elif req_lower == 'fw25':
            ground_truth_data = self.requirement_analyzer.analyze_fw25_missing_audit()
        elif req_lower == 'fw30':
            ground_truth_data = self.requirement_analyzer.analyze_fw30_missing_months()
        elif req_lower == 'fw40':
            ground_truth_data = self.requirement_analyzer.analyze_fw40_fraud_detection()
        elif req_lower == 'fw45':
            ground_truth_data = self.requirement_analyzer.analyze_fw45_gambling()
        elif req_lower == 'fw50':
            ground_truth_data = self.requirement_analyzer.analyze_fw50_debt_payments()
        else:
            return {'success': False, 'error': f'Unknown requirement: {requirement}'}

        # Get or generate prompt
        if use_dynamic_prompt:
            logger.info("Using dynamic prompt generation...")
            prompt = self.dynamic_prompt_generator.generate_fw_specific_prompt(
                requirement=req_lower,
                threshold=250
            )
        else:
            logger.info("Using template-based prompt...")
            template_name = f"{req_lower}_high_value" if req_lower == 'fw15' else req_lower
            template = self.template_library.get_template(template_name)
            if not template:
                # Fallback to dynamic generation
                prompt = self.dynamic_prompt_generator.generate_fw_specific_prompt(
                    requirement=req_lower,
                    threshold=250
                )
            else:
                prompt = template.format(
                    data=json.dumps(ground_truth_data, indent=2, default=str),
                    threshold=250
                )

        # Get LLM analysis
        llm_result = self.llm_service.generate(prompt)

        if not llm_result.get('success'):
            return {
                'success': False,
                'error': llm_result.get('error'),
                'requirement': requirement
            }

        llm_response = llm_result['response']

        # Validate against ground truth if enabled
        validation_results = None
        if validate and self.state['validation_enabled']:
            logger.info("Validating against ground truth...")
            # Parse LLM response to extract predictions
            # This is simplified - actual implementation would parse the response
            predictions = self._parse_llm_response_for_validation(llm_response, req_lower)

            validation_results = self._validate_with_ground_truth(
                requirement=req_lower,
                predictions=predictions
            )

        return {
            'success': True,
            'requirement': requirement,
            'ground_truth_data': ground_truth_data,
            'llm_response': llm_response,
            'validation': validation_results,
            'prompt_used': 'dynamic' if use_dynamic_prompt else 'template',
            'tokens_used': llm_result.get('tokens_used', 0),
            'latency': llm_result.get('latency', 0)
        }

    def analyze_all_fw_requirements(
        self,
        use_dynamic_prompts: bool = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis of all FW requirements

        Returns:
            Dictionary with results for all requirements
        """
        logger.info("Starting comprehensive FW requirements analysis...")

        requirements = ['fw15', 'fw20', 'fw25', 'fw30', 'fw40', 'fw45', 'fw50']
        results = {}

        for req in requirements:
            logger.info(f"Processing {req.upper()}...")
            results[req] = self.analyze_fw_requirement(
                requirement=req,
                use_dynamic_prompt=use_dynamic_prompts,
                validate=validate
            )

        # Generate comprehensive report
        report = self._generate_comprehensive_report(results)

        return {
            'success': True,
            'individual_results': results,
            'comprehensive_report': report,
            'timestamp': datetime.now().isoformat()
        }

    def _parse_llm_response_for_validation(
        self,
        llm_response: str,
        requirement: str
    ) -> List[Any]:
        """
        Parse LLM response to extract predictions for validation

        This is a simplified version - actual implementation would have
        more sophisticated parsing based on requirement type
        """
        # For now, return empty list - would be populated by actual parsing
        # In production, this would extract transaction IDs, amounts, etc.
        return []

    def _validate_with_ground_truth(
        self,
        requirement: str,
        predictions: List[Any]
    ) -> Dict[str, Any]:
        """
        Validate predictions against ground truth

        Returns:
            Validation metrics (precision, accuracy, etc.)
        """
        if requirement == 'fw15':
            validation = self.ground_truth_manager.validate_fw15_high_value(predictions)
        elif requirement == 'fw20':
            validation = self.ground_truth_manager.validate_fw20_luxury_brands(predictions)
        elif requirement == 'fw25':
            validation = self.ground_truth_manager.validate_fw25_missing_audit(predictions)
        elif requirement == 'fw30':
            validation = self.ground_truth_manager.validate_fw30_missing_months(predictions)
        elif requirement == 'fw45':
            validation = self.ground_truth_manager.validate_fw45_gambling(predictions)
        elif requirement == 'fw50':
            validation = self.ground_truth_manager.validate_fw50_debt_payments(predictions)
        else:
            return {'error': f'No validation method for {requirement}'}

        return validation

    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'total_requirements': len(results),
                'successful': sum(1 for r in results.values() if r.get('success')),
                'failed': sum(1 for r in results.values() if not r.get('success'))
            },
            'metrics': {},
            'recommendations': []
        }

        # Aggregate metrics
        for req, data in results.items():
            if data.get('success') and data.get('validation'):
                report['metrics'][req] = {
                    'precision': data['validation'].get('precision', 0),
                    'accuracy': data['validation'].get('accuracy', 0),
                    'meets_target': data['validation'].get('meets_98_target', False)
                }

        return report

    def compare_prompt_strategies(
        self,
        requirement: str,
        strategies: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare different prompt strategies (template vs dynamic vs hybrid)

        Args:
            requirement: FW requirement to test
            strategies: List of strategies to compare (defaults to all)

        Returns:
            Comparative analysis results
        """
        if strategies is None:
            strategies = ['template', 'dynamic', 'hybrid']

        logger.info(f"Comparing strategies for {requirement}...")

        results = {}

        for strategy in strategies:
            logger.info(f"Testing {strategy} strategy...")

            use_dynamic = strategy == 'dynamic'
            if strategy == 'hybrid':
                # Hybrid: Try template first, fallback to dynamic
                result = self.analyze_fw_requirement(
                    requirement=requirement,
                    use_dynamic_prompt=False,
                    validate=True
                )
                # If template performance is poor, try dynamic
                if result.get('validation', {}).get('precision', 1.0) < 0.98:
                    result = self.analyze_fw_requirement(
                        requirement=requirement,
                        use_dynamic_prompt=True,
                        validate=True
                    )
            else:
                result = self.analyze_fw_requirement(
                    requirement=requirement,
                    use_dynamic_prompt=use_dynamic,
                    validate=True
                )

            results[strategy] = result

        # Use comparative analyzer to generate comparison
        comparison = self.comparative_analyzer.compare_strategies(results)

        return {
            'requirement': requirement,
            'strategies_tested': strategies,
            'results': results,
            'comparison': comparison,
            'recommendation': self.comparative_analyzer.recommend_best_option(
                'strategy',
                results,
                criteria='balanced'
            )
        }

    def run_bias_detection(
        self,
        requirement: str = None
    ) -> Dict[str, Any]:
        """
        Run bias detection tests

        Args:
            requirement: Specific requirement to test (or all if None)

        Returns:
            Bias detection report
        """
        logger.info("Running bias detection...")

        # Create a simple analysis function for testing
        def analysis_function(data):
            # Simplified - would use actual LLM analysis
            return {'detected_items': []}

        # Test merchant name variations
        merchant_variations = [
            ['Gucci', 'GUCCI', 'gucci'],
            ['Western Union', 'WESTERN UNION', 'WesternUnion']
        ]

        merchant_bias = self.bias_detector.test_merchant_name_variations(
            analysis_function,
            merchant_variations
        )

        # Test currency format bias
        amount_formats = [
            '£250.00',
            '250 GBP',
            '250.00',
            'GBP 250'
        ]

        currency_bias = self.bias_detector.test_currency_format_bias(
            analysis_function,
            amount_formats
        )

        # Test date format bias
        date_formats = [
            '2025-01-15',
            '15/01/2025',
            '01-15-2025',
            '15 Jan 2025'
        ]

        date_bias = self.bias_detector.test_date_format_bias(
            analysis_function,
            date_formats
        )

        # Calculate overall bias
        all_tests = {
            'merchant_name_bias': merchant_bias,
            'currency_format_bias': currency_bias,
            'date_format_bias': date_bias
        }

        overall_bias = self.bias_detector.calculate_overall_bias(all_tests)

        bias_report = self.bias_detector.generate_bias_report(all_tests)

        return {
            'overall_bias_score': overall_bias,
            'meets_2_percent_target': overall_bias < 0.02,
            'individual_tests': all_tests,
            'report': bias_report
        }

