"""
TRUE AI AGENT - Fully Adaptive with Real Metrics
No mocks, true learning, iterative optimization
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path

from agent.llm_service import LLMService
from agent.data_processor import TransactionDataProcessor
from agent.ground_truth import GroundTruthManager
from agent.true_metrics import TrueMetricsCalculator
from agent.adaptive_tuner import AdaptivePromptTuner
from prompts.templates import PromptTemplateLibrary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrueAIAgent:
    """
    TRUE Adaptive AI Agent with:
    - Real LLM integration (no mocks)
    - True mathematical metrics (ground truth comparison)
    - Dynamic prompt generation based on failures
    - Iterative optimization until 98% targets met
    - User choice of LLM providers and models
    - Multiple output format testing
    """

    def __init__(
        self,
        llm_provider: str = 'openai',
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        data_dir: str = 'data',
        output_dir: str = 'results',
        max_tuning_iterations: int = 10
    ):
        """
        Initialize TRUE AI Agent

        Args:
            llm_provider: LLM provider (openai, anthropic, gemini, cohere, etc.)
            model: Specific model to use
            api_key: API key for LLM provider
            data_dir: Directory containing transaction CSVs
            output_dir: Directory for results
            max_tuning_iterations: Max iterations for adaptive tuning
        """
        logger.info(f"Initializing TRUE AI Agent with {llm_provider}")

        self.llm_provider = llm_provider
        self.model = model
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Initialize components
        logger.info("Initializing components...")

        # LLM Service - REAL, NO MOCKS
        llm_config = {}
        if model:
            llm_config['model'] = model
        if api_key:
            llm_config['api_key'] = api_key

        self.llm_service = LLMService(provider=llm_provider, **llm_config)
        logger.info(f"✓ LLM Service: {llm_provider}" + (f" ({model})" if model else ""))

        # Data Processing
        self.data_processor = TransactionDataProcessor(data_dir=data_dir)
        logger.info(f"✓ Data Processor: {data_dir}")

        # Ground Truth Manager
        self.ground_truth_manager = GroundTruthManager()
        logger.info(f"✓ Ground Truth Manager")

        # TRUE Metrics Calculator
        self.metrics_calculator = TrueMetricsCalculator()
        logger.info(f"✓ TRUE Metrics Calculator")

        # Adaptive Tuner
        self.adaptive_tuner = AdaptivePromptTuner(
            self.llm_service,
            max_iterations=max_tuning_iterations
        )
        logger.info(f"✓ Adaptive Tuner (max {max_tuning_iterations} iterations)")

        # Template Library (optional, for comparison)
        self.template_library = PromptTemplateLibrary()
        logger.info(f"✓ Template Library ({len(self.template_library.list_templates())} templates)")

        # Agent state
        self.state = {
            'initialized': True,
            'data_loaded': False,
            'current_provider': llm_provider,
            'current_model': model,
            'prompt_mode': 'dynamic',  # 'dynamic' or 'template'
            'last_analysis': None
        }

        # Results storage
        self.results = {
            'analyses': [],
            'best_prompts': {},
            'metrics_history': []
        }

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("TRUE AI Agent initialized successfully!")

    def load_data(self) -> Dict[str, Any]:
        """Load transaction data and ground truth"""
        logger.info("Loading data...")

        # Load transaction data
        data_result = self.data_processor.process_all()

        # Load ground truth
        gt_loaded = self.ground_truth_manager.load_ground_truth()

        if data_result and gt_loaded:
            self.state['data_loaded'] = True
            logger.info(f"✓ Loaded {len(data_result.get('full_data', []))} transactions")
            logger.info(f"✓ Loaded ground truth")

            return {
                'success': True,
                'transactions': len(data_result.get('full_data', [])),
                'ground_truth': gt_loaded
            }
        else:
            logger.error("Failed to load data")
            return {'success': False}

    def analyze_with_dynamic_tuning(
        self,
        requirement: str,
        requirement_description: str,
        target_precision: float = 0.98,
        target_accuracy: float = 0.98,
        test_formats: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a requirement using ADAPTIVE TUNING

        This is the TRUE AI behavior:
        1. Tests initial prompt
        2. Calculates TRUE metrics
        3. If targets not met, uses LLM to improve prompt based on failures
        4. Tests improved prompt
        5. Repeats until 98% targets met or max iterations

        Args:
            requirement: FW requirement code (e.g., 'fw15')
            requirement_description: Description for prompt generation
            target_precision: Minimum precision (default 98%)
            target_accuracy: Minimum accuracy (default 98%)
            test_formats: Whether to test multiple output formats

        Returns:
            Complete analysis results with best prompt and metrics
        """
        if not self.state['data_loaded']:
            logger.error("Data not loaded. Call load_data() first!")
            return {'success': False, 'error': 'Data not loaded'}

        logger.info(f"\n{'='*70}")
        logger.info(f"ADAPTIVE ANALYSIS: {requirement}")
        logger.info(f"Description: {requirement_description}")
        logger.info(f"Target: P≥{target_precision:.0%}, A≥{target_accuracy:.0%}")
        logger.info(f"{'='*70}\n")

        # Get data
        data_result = self.data_processor.process_all()
        llm_data = data_result['llm_formatted_data']
        total_transactions = len(data_result.get('full_data', []))

        # Get ground truth for this requirement
        ground_truth_data = self.ground_truth_manager.get_ground_truth_for_requirement(requirement)

        # Convert to expected format
        ground_truth = {
            'high_value_transactions': [t['transaction_id'] for t in ground_truth_data]
        }

        if not ground_truth:
            logger.error(f"No ground truth available for {requirement}")
            return {'success': False, 'error': 'No ground truth'}

        # Run adaptive tuning
        if test_formats:
            # Test multiple formats and find best
            tuning_result = self.adaptive_tuner.test_multiple_formats(
                requirement=requirement_description,
                data=llm_data,
                ground_truth=ground_truth,
                total_transactions=total_transactions,
                formats=['json', 'markdown', 'text']
            )
        else:
            # Single adaptive tuning run
            tuning_result = self.adaptive_tuner.adaptive_tune(
                requirement=requirement_description,
                data=llm_data,
                ground_truth=ground_truth,
                total_transactions=total_transactions,
                target_precision=target_precision,
                target_accuracy=target_accuracy
            )

        # Store results
        analysis_result = {
            'requirement': requirement,
            'description': requirement_description,
            'tuning_result': tuning_result,
            'target_achieved': tuning_result.get('target_achieved', False),
            'best_prompt': tuning_result.get('best_prompt'),
            'best_metrics': tuning_result.get('best_metrics'),
            'iterations': tuning_result.get('iterations'),
            'timestamp': datetime.now().isoformat()
        }

        self.results['analyses'].append(analysis_result)
        self.results['best_prompts'][requirement] = tuning_result.get('best_prompt')
        self.results['metrics_history'].extend(tuning_result.get('history', []))

        # Save to file
        self._save_results()

        return analysis_result

    def analyze_with_template(
        self,
        requirement: str,
        template_name: str
    ) -> Dict[str, Any]:
        """
        Analyze using a specific template (no tuning)

        For comparison with dynamic tuning
        """
        if not self.state['data_loaded']:
            return {'success': False, 'error': 'Data not loaded'}

        logger.info(f"Analyzing {requirement} with template: {template_name}")

        # Get data
        data_result = self.data_processor.process_all()
        llm_data = data_result['llm_formatted_data']
        total_transactions = len(data_result.get('full_data', []))

        # Get ground truth
        ground_truth_data = self.ground_truth_manager.get_ground_truth_for_requirement(requirement)
        ground_truth = {
            'high_value_transactions': [t['transaction_id'] for t in ground_truth_data]
        }

        # Format template
        prompt = self.template_library.format_template(
            template_name,
            data=llm_data,
            threshold=250.0
        )

        # Test prompt
        result = self.llm_service.generate(prompt)

        if not result.get('success'):
            return {'success': False, 'error': 'LLM generation failed'}

        # Calculate TRUE metrics
        metrics = self.metrics_calculator.calculate_metrics(
            result['response'],
            ground_truth,
            total_transactions
        )

        return {
            'success': True,
            'requirement': requirement,
            'template': template_name,
            'metrics': metrics,
            'prompt': prompt,
            'response': result['response']
        }

    def compare_dynamic_vs_template(
        self,
        requirement: str,
        requirement_description: str,
        template_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare dynamic tuning vs template-based approaches

        Shows why dynamic is superior
        """
        if template_names is None:
            template_names = ['direct_concise', 'role_based_expert', 'chain_of_thought']

        logger.info(f"Comparing dynamic vs template for {requirement}")

        results = {
            'requirement': requirement,
            'dynamic_result': None,
            'template_results': [],
            'winner': None
        }

        # Run dynamic tuning
        logger.info("\n1. DYNAMIC TUNING")
        dynamic_result = self.analyze_with_dynamic_tuning(
            requirement,
            requirement_description,
            test_formats=False
        )
        results['dynamic_result'] = dynamic_result

        # Run template tests
        logger.info("\n2. TEMPLATE TESTS")
        for template in template_names:
            logger.info(f"   Testing template: {template}")
            template_result = self.analyze_with_template(requirement, template)
            results['template_results'].append(template_result)

        # Determine winner
        dynamic_score = 0
        if dynamic_result.get('best_metrics'):
            dm = dynamic_result['best_metrics']
            dynamic_score = (dm['precision'] + dm['accuracy']) / 2

        best_template_score = 0
        best_template = None
        for tr in results['template_results']:
            if tr.get('success') and tr.get('metrics'):
                score = (tr['metrics']['precision'] + tr['metrics']['accuracy']) / 2
                if score > best_template_score:
                    best_template_score = score
                    best_template = tr['template']

        results['winner'] = 'dynamic' if dynamic_score > best_template_score else 'template'
        results['dynamic_score'] = dynamic_score
        results['best_template_score'] = best_template_score
        results['best_template'] = best_template

        logger.info(f"\nWINNER: {results['winner'].upper()}")
        logger.info(f"Dynamic Score: {dynamic_score:.2%}")
        logger.info(f"Best Template Score: {best_template_score:.2%} ({best_template})")

        return results

    def analyze_all_requirements(
        self,
        use_dynamic: bool = True,
        test_formats: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze ALL FW requirements

        Args:
            use_dynamic: Use adaptive tuning (True) or templates (False)
            test_formats: Test multiple output formats
        """
        requirements = {
            'fw15': 'Identify high-value transactions over £250',
            'fw20_luxury': 'Detect luxury brand purchases',
            'fw20_transfer': 'Identify money transfers',
            'fw25': 'Find transactions with missing audit trail',
            'fw30': 'Detect missing months in transaction history',
            'fw40': 'Light-touch fraud detection',
            'fw45': 'Identify gambling transactions',
            'fw50': 'Detect large debt payments'
        }

        results = {
            'method': 'dynamic' if use_dynamic else 'template',
            'total_requirements': len(requirements),
            'results': {},
            'summary': {
                'targets_met': 0,
                'targets_missed': 0,
                'average_precision': 0.0,
                'average_accuracy': 0.0
            }
        }

        total_precision = 0.0
        total_accuracy = 0.0

        for req_code, description in requirements.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Analyzing {req_code}: {description}")
            logger.info(f"{'='*70}")

            if use_dynamic:
                result = self.analyze_with_dynamic_tuning(
                    req_code,
                    description,
                    test_formats=test_formats
                )
            else:
                result = self.analyze_with_template(req_code, 'role_based_expert')

            results['results'][req_code] = result

            # Update summary
            if use_dynamic and result.get('best_metrics'):
                metrics = result['best_metrics']
                if metrics['meets_98_percent_target']:
                    results['summary']['targets_met'] += 1
                else:
                    results['summary']['targets_missed'] += 1

                total_precision += metrics['precision']
                total_accuracy += metrics['accuracy']

        # Calculate averages
        n = len(requirements)
        results['summary']['average_precision'] = total_precision / n
        results['summary']['average_accuracy'] = total_accuracy / n

        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: {results['summary']['targets_met']}/{n} requirements met targets")
        logger.info(f"Average Precision: {results['summary']['average_precision']:.2%}")
        logger.info(f"Average Accuracy: {results['summary']['average_accuracy']:.2%}")
        logger.info(f"{'='*70}\n")

        return results

    def _save_results(self):
        """Save results to file"""
        output_file = Path(self.output_dir) / f"true_ai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'provider': self.llm_provider,
            'model': self.model,
            'data_loaded': self.state['data_loaded'],
            'prompt_mode': self.state['prompt_mode'],
            'analyses_completed': len(self.results['analyses']),
            'best_prompts_count': len(self.results['best_prompts'])
        }
