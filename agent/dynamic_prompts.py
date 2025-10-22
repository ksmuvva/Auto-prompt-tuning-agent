"""
Dynamic Prompt Generator
AI-powered prompt generation to meet specific metric targets (98% precision/accuracy)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicPromptGenerator:
    """
    Generates and optimizes prompts dynamically using AI
    
    This component uses meta-prompting to create prompts that achieve
    the target precision (98%) and accuracy (98%) metrics.
    """

    def __init__(self, llm_service, requirement_name: str = ""):
        self.llm_service = llm_service
        self.requirement_name = requirement_name
        self.generation_history = []

    def generate_from_failures(
        self,
        failures: List[Dict[str, Any]],
        current_prompt: str,
        current_metrics: Dict[str, float]
    ) -> str:
        """
        Generate improved prompt based on failure analysis
        
        Args:
            failures: List of false positives and false negatives
            current_prompt: The current prompt that's underperforming
            current_metrics: Current precision, accuracy, etc.
        
        Returns:
            Improved prompt string
        """
        # Build meta-prompt for prompt generation
        meta_prompt = f"""You are an expert prompt engineer specializing in financial data analysis.

TASK: Create an improved prompt to analyze bank transactions for {self.requirement_name}.

CURRENT PROMPT PERFORMANCE:
- Precision: {current_metrics.get('precision', 0):.2%} (Target: 98%)
- Accuracy: {current_metrics.get('accuracy', 0):.2%} (Target: 98%)
- Recall: {current_metrics.get('recall', 0):.2%}
- F1 Score: {current_metrics.get('f1_score', 0):.2%}

CURRENT PROMPT:
{current_prompt}

FAILURE ANALYSIS:
"""

        # Add false positives
        if failures.get('false_positives'):
            meta_prompt += f"\nFalse Positives (incorrectly identified): {len(failures['false_positives'])}\n"
            for i, fp in enumerate(failures['false_positives'][:5], 1):
                meta_prompt += f"  {i}. {fp}\n"
            if len(failures['false_positives']) > 5:
                meta_prompt += f"  ... and {len(failures['false_positives']) - 5} more\n"

        # Add false negatives
        if failures.get('false_negatives'):
            meta_prompt += f"\nFalse Negatives (missed): {len(failures['false_negatives'])}\n"
            for i, fn in enumerate(failures['false_negatives'][:5], 1):
                meta_prompt += f"  {i}. {fn}\n"
            if len(failures['false_negatives']) > 5:
                meta_prompt += f"  ... and {len(failures['false_negatives']) - 5} more\n"

        meta_prompt += """

REQUIREMENTS FOR IMPROVED PROMPT:
1. Must reduce false positives by being more specific and adding explicit exclusion criteria
2. Must reduce false negatives by being more comprehensive and covering edge cases
3. Must achieve precision ≥ 98% and accuracy ≥ 98%
4. Should use clear, unambiguous language
5. Should include examples if helpful
6. Should specify exact output format
7. Should include validation criteria

GENERATE AN IMPROVED PROMPT that addresses these failures and meets the target metrics.
Return ONLY the improved prompt text, no explanations.
"""

        # Generate improved prompt
        result = self.llm_service.generate(meta_prompt)
        
        if result.get('success'):
            improved_prompt = result['response'].strip()
            
            # Store in history
            self.generation_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': 'from_failures',
                'current_metrics': current_metrics,
                'improved_prompt': improved_prompt,
                'failure_count': len(failures.get('false_positives', [])) + len(failures.get('false_negatives', []))
            })
            
            logger.info("Generated improved prompt from failure analysis")
            return improved_prompt
        else:
            logger.error("Failed to generate improved prompt")
            return current_prompt

    def optimize_for_metric(
        self,
        target_metric: str,
        target_value: float,
        current_value: float,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate prompt optimized for a specific metric
        
        Args:
            target_metric: 'precision', 'accuracy', 'recall', or 'f1'
            target_value: Target value (e.g., 0.98)
            current_value: Current value
            context: Additional context about the task
        """
        gap = target_value - current_value
        
        optimization_strategies = {
            'precision': """
Focus on reducing false positives:
- Add more specific filtering criteria
- Include explicit exclusion rules
- Use stricter matching patterns
- Validate each detection before including it
""",
            'accuracy': """
Improve overall correctness:
- Balance precision and recall
- Cover both common and edge cases
- Add validation steps
- Use multiple detection methods
""",
            'recall': """
Focus on reducing false negatives:
- Broaden detection criteria
- Include more pattern variations
- Cover edge cases explicitly
- Use fuzzy matching where appropriate
""",
            'f1': """
Balance precision and recall:
- Optimize for both false positives and false negatives
- Use layered detection (broad then narrow)
- Include confidence scoring
- Add explicit examples of what to include/exclude
"""
        }

        strategy = optimization_strategies.get(target_metric, optimization_strategies['accuracy'])

        meta_prompt = f"""You are optimizing a prompt for {context.get('requirement_name', 'data analysis')}.

TARGET METRIC: {target_metric.upper()}
Current Value: {current_value:.2%}
Target Value: {target_value:.2%}
Gap: {gap:.2%}

OPTIMIZATION STRATEGY:
{strategy}

TASK CONTEXT:
{json.dumps(context, indent=2)}

Generate an optimized prompt that specifically addresses the {target_metric} gap.
The prompt should be clear, specific, and designed to achieve the target value.

Return ONLY the optimized prompt text.
"""

        result = self.llm_service.generate(meta_prompt)
        
        if result.get('success'):
            optimized_prompt = result['response'].strip()
            
            self.generation_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': 'optimize_for_metric',
                'target_metric': target_metric,
                'target_value': target_value,
                'current_value': current_value,
                'gap': gap,
                'optimized_prompt': optimized_prompt
            })
            
            logger.info(f"Generated prompt optimized for {target_metric}")
            return optimized_prompt
        else:
            logger.error("Failed to generate optimized prompt")
            return ""

    def generate_reasoning_prompt(
        self,
        task_description: str,
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a chain-of-thought reasoning prompt
        
        Args:
            task_description: Description of the analysis task
            examples: Optional examples to include
        """
        meta_prompt = f"""Generate a chain-of-thought reasoning prompt for this task:

TASK: {task_description}

The prompt should:
1. Guide the AI through a step-by-step reasoning process
2. Show its thinking at each step
3. Validate findings before finalizing
4. Explain why each item was included or excluded
5. Achieve high precision (98%) and accuracy (98%)

Format the prompt to encourage the AI to:
- State its reasoning explicitly
- Show intermediate steps
- Question its own assumptions
- Double-check edge cases
- Provide final validated results

"""

        if examples:
            meta_prompt += "\nINCLUDE THESE EXAMPLES:\n"
            for i, ex in enumerate(examples, 1):
                meta_prompt += f"\nExample {i}:\n{json.dumps(ex, indent=2)}\n"

        meta_prompt += "\nReturn ONLY the reasoning prompt text."

        result = self.llm_service.generate(meta_prompt)
        
        if result.get('success'):
            reasoning_prompt = result['response'].strip()
            
            self.generation_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': 'reasoning_prompt',
                'task_description': task_description,
                'reasoning_prompt': reasoning_prompt
            })
            
            logger.info("Generated reasoning prompt")
            return reasoning_prompt
        else:
            logger.error("Failed to generate reasoning prompt")
            return task_description

    def iterate_until_target(
        self,
        initial_prompt: str,
        evaluate_function: callable,
        target_precision: float = 0.98,
        target_accuracy: float = 0.98,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Iteratively improve prompt until targets are met
        
        Args:
            initial_prompt: Starting prompt
            evaluate_function: Function that evaluates a prompt and returns metrics
            target_precision: Target precision (default: 0.98)
            target_accuracy: Target accuracy (default: 0.98)
            max_iterations: Maximum number of improvement iterations
        
        Returns:
            Dictionary with best prompt and metrics
        """
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_metrics = {'precision': 0, 'accuracy': 0, 'f1_score': 0}
        
        logger.info(f"Starting iterative optimization (target: P={target_precision:.0%}, A={target_accuracy:.0%})")
        
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Evaluate current prompt
            metrics = evaluate_function(current_prompt)
            
            precision = metrics.get('precision', 0)
            accuracy = metrics.get('accuracy', 0)
            
            logger.info(f"  Precision: {precision:.2%}, Accuracy: {accuracy:.2%}")
            
            # Update best if improved
            if (precision + accuracy) > (best_metrics['precision'] + best_metrics['accuracy']):
                best_prompt = current_prompt
                best_metrics = metrics
            
            # Check if targets met
            if precision >= target_precision and accuracy >= target_accuracy:
                logger.info(f"✓ Targets met in {iteration + 1} iterations!")
                return {
                    'success': True,
                    'prompt': current_prompt,
                    'metrics': metrics,
                    'iterations': iteration + 1,
                    'target_met': True
                }
            
            # Generate improved prompt for next iteration
            if iteration < max_iterations - 1:
                failures = metrics.get('failures', {
                    'false_positives': [],
                    'false_negatives': []
                })
                
                current_prompt = self.generate_from_failures(
                    failures,
                    current_prompt,
                    metrics
                )
        
        # Max iterations reached
        logger.warning(f"Max iterations reached. Best: P={best_metrics['precision']:.2%}, A={best_metrics['accuracy']:.2%}")
        
        return {
            'success': True,
            'prompt': best_prompt,
            'metrics': best_metrics,
            'iterations': max_iterations,
            'target_met': False,
            'message': 'Maximum iterations reached without meeting targets'
        }

    def generate_fw_specific_prompt(
        self,
        requirement: str,
        threshold: Optional[float] = None
    ) -> str:
        """
        Generate FW requirement-specific prompt
        
        Args:
            requirement: FW requirement code (e.g., 'FW15', 'FW20', etc.)
            threshold: Optional threshold value
        """
        requirement_specs = {
            'FW15': {
                'task': 'Identify all transactions exceeding £{threshold}',
                'focus': 'High-value transaction detection',
                'precision_priority': 'Avoid including transactions at or below threshold',
                'recall_priority': 'Do not miss any transaction above threshold'
            },
            'FW20': {
                'task': 'Identify luxury brand purchases and money transfer transactions',
                'focus': 'Pattern-based categorization',
                'precision_priority': 'Only include confirmed luxury brands and transfer services',
                'recall_priority': 'Check for name variations and misspellings'
            },
            'FW25': {
                'task': 'Identify transactions lacking proper audit trail',
                'focus': 'Missing documentation detection',
                'precision_priority': 'Distinguish between legitimate and missing documentation',
                'recall_priority': 'Check all transaction types for audit gaps'
            },
            'FW30': {
                'task': 'Detect missing months in a 6-month statement sequence',
                'focus': 'Temporal gap analysis',
                'precision_priority': 'Do not flag months with partial data as missing',
                'recall_priority': 'Identify all calendar gaps in the sequence'
            },
            'FW40': {
                'task': 'Detect misspellings, calculation errors, and data inconsistencies',
                'focus': 'Data quality validation',
                'precision_priority': 'Avoid flagging acceptable variations as errors',
                'recall_priority': 'Check all fields for potential errors'
            },
            'FW45': {
                'task': 'Summarize all gambling transactions over 6 months',
                'focus': 'Gambling activity analysis',
                'precision_priority': 'Only include confirmed gambling operators',
                'recall_priority': 'Include all gambling-related transactions'
            },
            'FW50': {
                'task': 'Identify large debt payments (≥£{threshold})',
                'focus': 'Debt repayment tracking',
                'precision_priority': 'Distinguish debt payments from regular purchases',
                'recall_priority': 'Include all payment types (credit cards, loans, mortgages)'
            }
        }

        spec = requirement_specs.get(requirement.upper(), {})
        
        if not spec:
            logger.warning(f"Unknown requirement: {requirement}")
            return ""

        task = spec['task'].format(threshold=threshold) if threshold else spec['task']

        prompt_template = f"""You are a financial analyst with expertise in bank transaction analysis.

REQUIREMENT: {requirement}
TASK: {task}

OBJECTIVE:
{spec['focus']}

PRECISION CRITERIA (to avoid false positives):
{spec['precision_priority']}

RECALL CRITERIA (to avoid false negatives):
{spec['recall_priority']}

INSTRUCTIONS:
1. Analyze each transaction carefully
2. Apply the precision and recall criteria strictly
3. Show your reasoning for edge cases
4. Provide results in a structured format
5. Include transaction IDs, amounts, and relevant details

EXPECTED PERFORMANCE:
- Precision: ≥ 98%
- Accuracy: ≥ 98%

Analyze the provided transaction data according to these specifications.
"""

        logger.info(f"Generated FW-specific prompt for {requirement}")
        
        self.generation_history.append({
            'timestamp': datetime.now().isoformat(),
            'method': 'fw_specific',
            'requirement': requirement,
            'prompt': prompt_template
        })
        
        return prompt_template

    def export_history(self, filepath: str):
        """Export generation history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.generation_history, f, indent=2)
        
        logger.info(f"Exported prompt generation history to {filepath}")
