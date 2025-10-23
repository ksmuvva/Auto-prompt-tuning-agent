"""
Multi-Reasoning Engine System

Implements multiple reasoning strategies for diverse, realistic data generation:
- Monte Carlo Sampling: Statistical distributions
- Beam Search: Multiple candidates, select best
- Chain-of-Thought: Step-by-step reasoning
- Tree-of-Thoughts: Explore multiple paths
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import numpy as np
from .intent_engine import Intent


@dataclass
class GenerationResult:
    """Result from a reasoning engine"""
    data: Any
    score: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ReasoningEngine(ABC):
    """Abstract base class for reasoning engines"""

    def __init__(self, llm_provider):
        self.llm = llm_provider

    @abstractmethod
    def generate(self, intent: Intent, schema: Dict[str, str], count: int = 1) -> List[GenerationResult]:
        """Generate data using this reasoning strategy"""
        pass


class MonteCarloEngine(ReasoningEngine):
    """
    Monte Carlo Sampling Engine

    Uses probabilistic sampling with realistic statistical distributions
    Best for: Realistic distributions, numerical data, demographics
    """

    def generate(self, intent: Intent, schema: Dict[str, str], count: int = 1) -> List[GenerationResult]:
        """
        Generate data using Monte Carlo sampling

        Creates realistic statistical distributions based on domain knowledge
        """
        results = []

        # Build distribution prompt
        distribution_prompt = f"""Define realistic probability distributions for generating {intent.data_type}.

Geography: {intent.geography}
Domain: {intent.domain}
Purpose: {intent.purpose}

Schema:
{schema}

For each field, specify:
1. Distribution type (normal, uniform, categorical, etc.)
2. Parameters (mean, std for normal; categories and probabilities for categorical)
3. Constraints (min, max, valid values)

Respond with JSON:
{{
  "field_name": {{
    "type": "distribution_type",
    "parameters": {{}},
    "constraints": {{}}
  }}
}}"""

        try:
            distributions = self.llm.generate_structured(
                distribution_prompt,
                schema={"field": "distribution"},
                temperature=0.3
            )
        except Exception as e:
            print(f"Warning: Failed to get distributions from LLM: {e}")
            distributions = self._default_distributions(schema, intent)

        # Generate samples using Monte Carlo
        for i in range(count):
            sample = self._monte_carlo_sample(distributions, intent, schema)
            results.append(GenerationResult(
                data=sample,
                score=1.0,  # All samples are equally valid in Monte Carlo
                reasoning=f"Monte Carlo sample {i+1}/{count}",
                metadata={"distribution": "monte_carlo", "index": i}
            ))

        return results

    def _monte_carlo_sample(self, distributions: Dict, intent: Intent, schema: Dict[str, str]) -> Dict[str, Any]:
        """Generate a single Monte Carlo sample"""
        sample = {}

        for field, field_type in schema.items():
            dist_info = distributions.get(field, {})
            dist_type = dist_info.get('type', 'uniform')
            params = dist_info.get('parameters', {})

            # Sample based on distribution type
            if dist_type == 'normal':
                mean = params.get('mean', 0)
                std = params.get('std', 1)
                value = np.random.normal(mean, std)
                # Apply constraints
                min_val = dist_info.get('constraints', {}).get('min')
                max_val = dist_info.get('constraints', {}).get('max')
                if min_val is not None:
                    value = max(value, min_val)
                if max_val is not None:
                    value = min(value, max_val)
                sample[field] = round(value, 2) if field_type == 'number' else int(value)

            elif dist_type == 'categorical':
                categories = params.get('categories', [])
                probabilities = params.get('probabilities', None)
                if categories:
                    sample[field] = np.random.choice(categories, p=probabilities)
                else:
                    sample[field] = f"sample_{field}"

            elif dist_type == 'uniform':
                min_val = params.get('min', 0)
                max_val = params.get('max', 100)
                value = np.random.uniform(min_val, max_val)
                sample[field] = round(value, 2) if field_type == 'number' else int(value)

            else:
                # Default: use LLM to generate value
                sample[field] = self._generate_field_value(field, field_type, intent)

        return sample

    def _default_distributions(self, schema: Dict[str, str], intent: Intent) -> Dict:
        """Provide default distributions when LLM fails"""
        distributions = {}

        for field, field_type in schema.items():
            if field_type == 'number':
                distributions[field] = {
                    'type': 'normal',
                    'parameters': {'mean': 50, 'std': 15},
                    'constraints': {'min': 0, 'max': 100}
                }
            else:
                distributions[field] = {
                    'type': 'categorical',
                    'parameters': {'categories': [f"{field}_value_{i}" for i in range(5)]},
                    'constraints': {}
                }

        return distributions

    def _generate_field_value(self, field: str, field_type: str, intent: Intent) -> Any:
        """Generate a single field value"""
        # Simple generation based on field type
        if field_type == 'number':
            return random.randint(1, 100)
        elif field_type == 'email':
            return f"{field}{random.randint(1, 1000)}@example.com"
        elif field_type == 'date':
            return "2025-01-01"
        else:
            return f"value_{random.randint(1, 1000)}"


class BeamSearchEngine(ReasoningEngine):
    """
    Beam Search Reasoning Engine

    Generates multiple candidates and selects best based on quality criteria
    Best for: Diversity, quality optimization, multi-criteria selection
    """

    def __init__(self, llm_provider, beam_width: int = 5):
        super().__init__(llm_provider)
        self.beam_width = beam_width

    def generate(self, intent: Intent, schema: Dict[str, str], count: int = 1) -> List[GenerationResult]:
        """
        Generate data using beam search

        Creates multiple candidates per record and selects best
        """
        results = []

        for i in range(count):
            # Generate beam_width candidates
            candidates = self._generate_candidates(intent, schema, self.beam_width)

            # Score and rank candidates
            scored_candidates = self._score_candidates(candidates, intent, schema)

            # Select best candidate
            best = max(scored_candidates, key=lambda x: x.score)
            best.reasoning = f"Beam search (width={self.beam_width}), selected best of {len(candidates)}"
            best.metadata = {"beam_width": self.beam_width, "index": i}

            results.append(best)

        return results

    def _generate_candidates(self, intent: Intent, schema: Dict[str, str], num_candidates: int) -> List[Dict[str, Any]]:
        """Generate multiple candidate records"""

        generation_prompt = f"""Generate {num_candidates} diverse candidate records for {intent.data_type}.

Geography: {intent.geography}
Domain: {intent.domain}
Purpose: {intent.purpose}

Schema:
{schema}

Requirements:
- Make each candidate DIFFERENT (vary names, demographics, values)
- Ensure REALISTIC values appropriate for {intent.geography or 'global'} {intent.domain or ''} context
- Include diverse demographics if applicable

Respond with JSON array of {num_candidates} records:
[
  {{ {', '.join(f'"{field}": "value"' for field in schema.keys())} }},
  ...
]"""

        try:
            response = self.llm.generate(generation_prompt, temperature=0.8)  # Higher temp for diversity

            # Parse JSON array
            import json
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                candidates = json.loads(json_match.group(0))
                return candidates[:num_candidates]  # Limit to requested count
            else:
                raise ValueError("Could not parse JSON array")

        except Exception as e:
            print(f"Warning: Failed to generate candidates: {e}")
            # Fallback: generate simple candidates
            return [
                {field: f"candidate_{i}_{field}" for field in schema.keys()}
                for i in range(num_candidates)
            ]

    def _score_candidates(self, candidates: List[Dict], intent: Intent, schema: Dict) -> List[GenerationResult]:
        """Score candidates based on quality criteria"""

        scoring_prompt = f"""Score these candidate records for {intent.data_type} generation.

Criteria:
1. Realism (1-10): How realistic are the values?
2. Diversity (1-10): How different is this from typical entries?
3. Completeness (1-10): Are all fields properly filled?
4. Consistency (1-10): Are field values internally consistent?

Geography: {intent.geography}
Domain: {intent.domain}

Candidates:
{candidates}

Respond with JSON array of scores (0-100 overall):
[
  {{ "index": 0, "score": 85, "reasoning": "Realistic UK customer, diverse demographics" }},
  ...
]"""

        try:
            response = self.llm.generate_structured(
                scoring_prompt,
                schema={"scores": "array"},
                temperature=0.3
            )

            scores = response.get('scores', []) if isinstance(response, dict) else response

            results = []
            for i, candidate in enumerate(candidates):
                score_info = next((s for s in scores if s.get('index') == i), None)
                if score_info:
                    score = score_info.get('score', 50) / 100.0
                    reasoning = score_info.get('reasoning', '')
                else:
                    score = 0.5

                results.append(GenerationResult(
                    data=candidate,
                    score=score,
                    reasoning=reasoning
                ))

            return results

        except Exception as e:
            print(f"Warning: Failed to score candidates: {e}")
            # Fallback: assign random scores
            return [
                GenerationResult(data=c, score=random.uniform(0.5, 1.0), reasoning="")
                for c in candidates
            ]


class ChainOfThoughtEngine(ReasoningEngine):
    """
    Chain-of-Thought Reasoning Engine

    Step-by-step reasoning for complex, interdependent data
    Best for: Related fields, transactions, cause-effect relationships
    """

    def generate(self, intent: Intent, schema: Dict[str, str], count: int = 1) -> List[GenerationResult]:
        """
        Generate data using chain-of-thought reasoning

        Builds records step-by-step with reasoning about dependencies
        """
        results = []

        for i in range(count):
            record, reasoning_chain = self._generate_with_reasoning(intent, schema, i)
            results.append(GenerationResult(
                data=record,
                score=1.0,
                reasoning=reasoning_chain,
                metadata={"method": "chain_of_thought", "index": i}
            ))

        return results

    def _generate_with_reasoning(self, intent: Intent, schema: Dict[str, str], index: int) -> tuple[Dict[str, Any], str]:
        """Generate a single record with step-by-step reasoning"""

        cot_prompt = f"""Generate a realistic {intent.data_type} record using step-by-step reasoning.

Geography: {intent.geography}
Domain: {intent.domain}
Purpose: {intent.purpose}

Schema:
{schema}

Think through each field step by step, considering:
1. What makes sense for this {intent.domain or ''} {intent.data_type}?
2. How do fields relate to each other?
3. What constraints exist (realistic ranges, valid combinations)?

Example reasoning:
"Step 1: Age 32 → professional age group
 Step 2: Age 32 → likely email format: firstname.lastname
 Step 3: UK geography → postcode format SW1A 1AA
 Step 4: Professional age → likely spending £500-2000/month"

Generate ONE record with reasoning. Respond with JSON:
{{
  "reasoning_steps": ["step 1", "step 2", ...],
  "record": {{ {', '.join(f'"{field}": "value"' for field in schema.keys())} }}
}}"""

        try:
            response = self.llm.generate_structured(
                cot_prompt,
                schema={"reasoning_steps": "array", "record": "object"},
                temperature=0.7
            )

            record = response.get('record', {})
            reasoning_steps = response.get('reasoning_steps', [])
            reasoning_chain = " → ".join(reasoning_steps)

            return record, reasoning_chain

        except Exception as e:
            print(f"Warning: Chain-of-thought generation failed: {e}")
            # Fallback
            record = {field: f"value_{index}_{field}" for field in schema.keys()}
            return record, "Fallback generation (LLM failed)"


class TreeOfThoughtsEngine(ReasoningEngine):
    """
    Tree-of-Thoughts Reasoning Engine

    Explores multiple reasoning paths and evaluates branches
    Best for: Complex scenarios, multiple valid options, decision trees
    """

    def __init__(self, llm_provider, num_branches: int = 3, depth: int = 2):
        super().__init__(llm_provider)
        self.num_branches = num_branches
        self.depth = depth

    def generate(self, intent: Intent, schema: Dict[str, str], count: int = 1) -> List[GenerationResult]:
        """
        Generate data using tree-of-thoughts reasoning

        Explores multiple paths through decision tree
        """
        results = []

        for i in range(count):
            record, tree_reasoning = self._explore_tree(intent, schema, i)
            results.append(GenerationResult(
                data=record,
                score=1.0,
                reasoning=tree_reasoning,
                metadata={"method": "tree_of_thoughts", "branches": self.num_branches, "index": i}
            ))

        return results

    def _explore_tree(self, intent: Intent, schema: Dict[str, str], index: int) -> tuple[Dict[str, Any], str]:
        """Explore tree of possibilities"""

        tot_prompt = f"""Generate {intent.data_type} by exploring multiple possibilities.

Geography: {intent.geography}
Domain: {intent.domain}

Schema:
{schema}

Explore {self.num_branches} different paths/scenarios:
- Path 1: [One type of scenario]
- Path 2: [Another type of scenario]
- Path 3: [Third type of scenario]

For each path, consider what values would make sense.
Then select the most realistic path and generate a record.

Example for customer data:
- Path 1: Young urban professional → Age 25-35, city postcode, high tech spending
- Path 2: Retired suburban → Age 65+, suburban postcode, traditional shopping
- Path 3: Student → Age 18-24, university area, budget-conscious

Respond with JSON:
{{
  "explored_paths": ["path 1 description", "path 2 description", "path 3 description"],
  "selected_path": "path X description",
  "record": {{ {', '.join(f'"{field}": "value"' for field in schema.keys())} }}
}}"""

        try:
            response = self.llm.generate_structured(
                tot_prompt,
                schema={"explored_paths": "array", "selected_path": "string", "record": "object"},
                temperature=0.8
            )

            record = response.get('record', {})
            paths = response.get('explored_paths', [])
            selected = response.get('selected_path', '')

            tree_reasoning = f"Explored paths: {', '.join(paths)}. Selected: {selected}"

            return record, tree_reasoning

        except Exception as e:
            print(f"Warning: Tree-of-thoughts generation failed: {e}")
            # Fallback
            record = {field: f"value_{index}_{field}" for field in schema.keys()}
            return record, "Fallback generation (LLM failed)"


class ReasoningEngineFactory:
    """Factory for creating reasoning engines"""

    @staticmethod
    def create(engine_type: str, llm_provider, **kwargs) -> ReasoningEngine:
        """
        Create a reasoning engine

        Args:
            engine_type: Type of engine (monte_carlo, beam_search, chain_of_thought, tree_of_thoughts)
            llm_provider: LLM provider instance
            **kwargs: Additional arguments for specific engines

        Returns:
            ReasoningEngine instance
        """
        engines = {
            'monte_carlo': MonteCarloEngine,
            'beam_search': BeamSearchEngine,
            'chain_of_thought': ChainOfThoughtEngine,
            'cot': ChainOfThoughtEngine,
            'tree_of_thoughts': TreeOfThoughtsEngine,
            'tot': TreeOfThoughtsEngine
        }

        engine_type = engine_type.lower()
        if engine_type not in engines:
            raise ValueError(f"Unknown engine type: {engine_type}. Available: {list(engines.keys())}")

        return engines[engine_type](llm_provider, **kwargs)
