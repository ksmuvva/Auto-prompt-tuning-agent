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
from collections import defaultdict
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
            # Validate that distributions match schema fields
            if not distributions or not any(field in distributions for field in schema.keys()):
                raise ValueError("Distributions don't match schema fields")
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
            if dist_info is None:
                dist_info = {}
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

                # Proper type handling based on semantic meaning
                field_lower = field.lower()
                if field_lower in ['age', 'years', 'count', 'quantity', 'number_of']:
                    sample[field] = int(value)  # Integer for counts/ages
                elif 'price' in field_lower or 'amount' in field_lower or 'salary' in field_lower:
                    sample[field] = round(value, 2)  # 2 decimal places for currency
                elif field_type == 'number':
                    sample[field] = round(value, 2)  # Default for numbers
                else:
                    sample[field] = int(value)

            elif dist_type == 'categorical':
                categories = params.get('categories', [])
                probabilities = params.get('probabilities', None)
                if categories:
                    sample[field] = np.random.choice(categories, p=probabilities)
                else:
                    # No categories provided, use direct field value generation
                    sample[field] = self._generate_field_value(field, field_type, intent)

            elif dist_type == 'uniform':
                min_val = params.get('min', 0)
                max_val = params.get('max', 100)
                value = np.random.uniform(min_val, max_val)

                # Proper type handling
                field_lower = field.lower()
                if field_lower in ['age', 'years', 'count', 'quantity', 'number_of']:
                    sample[field] = int(value)
                elif 'price' in field_lower or 'amount' in field_lower or 'salary' in field_lower:
                    sample[field] = round(value, 2)
                elif field_type == 'number':
                    sample[field] = round(value, 2)
                else:
                    sample[field] = int(value)

            elif dist_type == 'direct':
                # Use direct field value generation (for non-numeric types)
                sample[field] = self._generate_field_value(field, field_type, intent)

            else:
                # Default: use LLM to generate value
                sample[field] = self._generate_field_value(field, field_type, intent)

        return sample

    def _default_distributions(self, schema: Dict[str, str], intent: Intent) -> Dict:
        """Provide default distributions when LLM fails"""
        distributions = {}

        for field, field_type in schema.items():
            field_lower = field.lower()

            # Use proper defaults based on semantic type
            if field_lower in ['age', 'years']:
                distributions[field] = {
                    'type': 'normal',
                    'parameters': {'mean': 38, 'std': 15},
                    'constraints': {'min': 18, 'max': 80}
                }
            elif field_type == 'number' or 'amount' in field_lower or 'price' in field_lower:
                distributions[field] = {
                    'type': 'normal',
                    'parameters': {'mean': 100, 'std': 50},
                    'constraints': {'min': 10, 'max': 5000}
                }
            else:
                # For non-numeric fields, use direct generation
                distributions[field] = {
                    'type': 'direct',  # Special type to trigger _generate_field_value
                    'parameters': {},
                    'constraints': {}
                }

        return distributions

    def _generate_field_value(self, field: str, field_type: str, intent: Intent) -> Any:
        """Generate a single field value with proper type handling"""
        from .uk_standards import UKStandardsGenerator

        uk_gen = UKStandardsGenerator()
        field_lower = field.lower()

        # Semantic type detection
        if 'id' in field_lower and field != 'id':
            # Generate proper ID
            prefix = field.replace('_id', '').replace('id', '').upper() or 'ID'
            return f"{prefix}{random.randint(1, 9999):04d}"
        elif field_lower in ['age', 'years', 'count', 'quantity']:
            return random.randint(18, 80)  # Integer for age
        elif 'name' in field_lower:
            first, last = uk_gen.generate_name()
            return f"{first} {last}" if 'full' in field_lower else (first if 'first' in field_lower else last)
        elif 'email' in field_lower:
            return f"user{random.randint(1, 9999)}@example.com"
        elif 'phone' in field_lower or 'mobile' in field_lower:
            return uk_gen.generate_phone('mobile')
        elif 'postcode' in field_lower or 'postal' in field_lower:
            return uk_gen.generate_postcode()
        elif field_type == 'number' or 'price' in field_lower or 'amount' in field_lower:
            return round(random.uniform(10.0, 1000.0), 2)
        elif field_type == 'date':
            return uk_gen.generate_random_date(2023, 2025)
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
            # Fallback: generate realistic candidates using Monte Carlo-like approach
            monte_carlo = MonteCarloEngine(self.llm)
            fallback_results = monte_carlo.generate(intent, schema, count=num_candidates)
            return [result.data for result in fallback_results]

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
                    reasoning = 'Default score'

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
            reasoning_chain = " → ".join(reasoning_steps) if reasoning_steps else "Direct generation"

            # If record is empty, use fallback
            if not record or len(record) == 0:
                raise ValueError("Empty record returned")

            return record, reasoning_chain

        except Exception as e:
            print(f"Warning: Chain-of-thought generation failed: {e}")
            # Fallback: Use Monte Carlo for realistic data
            monte_carlo = MonteCarloEngine(self.llm)
            fallback_results = monte_carlo.generate(intent, schema, count=1)
            record = fallback_results[0].data if fallback_results else {}
            return record, f"Fallback generation (CoT failed: {str(e)[:50]})"


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

            # If record is empty, use fallback
            if not record or len(record) == 0:
                raise ValueError("Empty record returned")

            tree_reasoning = f"Explored {len(paths)} paths. Selected: {selected[:50]}" if paths else "Direct generation"

            return record, tree_reasoning

        except Exception as e:
            print(f"Warning: Tree-of-thoughts generation failed: {e}")
            # Fallback: Use Monte Carlo for realistic data
            monte_carlo = MonteCarloEngine(self.llm)
            fallback_results = monte_carlo.generate(intent, schema, count=1)
            record = fallback_results[0].data if fallback_results else {}
            return record, f"Fallback generation (ToT failed: {str(e)[:50]})"


class MCTSNode:
    """Node in Monte Carlo Tree Search"""

    def __init__(self, record: Dict[str, Any], parent=None):
        self.record = record
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_features = []

    def uct_value(self, exploration_weight: float = 1.414) -> float:
        """Calculate UCT (Upper Confidence Bound for Trees) value"""
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration = exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        """Check if all child nodes have been explored"""
        return len(self.untried_features) == 0

    def best_child(self, exploration_weight: float = 1.414):
        """Select best child using UCT"""
        return max(self.children, key=lambda c: c.uct_value(exploration_weight))


class MCTSEngine(ReasoningEngine):
    """
    Monte Carlo Tree Search (MCTS) Reasoning Engine

    Uses MCTS to explore the space of possible data generation strategies
    Best for: Complex optimization, exploration-exploitation tradeoff, adaptive generation
    """

    def __init__(self, llm_provider, num_simulations: int = 100, exploration_weight: float = 1.414):
        super().__init__(llm_provider)
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight

    def generate(self, intent: Intent, schema: Dict[str, str], count: int = 1) -> List[GenerationResult]:
        """
        Generate data using MCTS

        Explores generation space using selection, expansion, simulation, backpropagation
        """
        results = []

        for i in range(count):
            # Run MCTS to find optimal generation strategy
            best_record = self._run_mcts(intent, schema, i)

            results.append(GenerationResult(
                data=best_record,
                score=1.0,
                reasoning=f"MCTS with {self.num_simulations} simulations, balanced exploration-exploitation",
                metadata={
                    "method": "mcts",
                    "simulations": self.num_simulations,
                    "exploration_weight": self.exploration_weight,
                    "index": i
                }
            ))

        return results

    def _run_mcts(self, intent: Intent, schema: Dict[str, str], index: int) -> Dict[str, Any]:
        """Run MCTS algorithm"""

        # Initialize root with empty record
        root_record = {field: None for field in schema.keys()}
        root = MCTSNode(root_record)
        root.untried_features = list(schema.keys())

        # Run simulations
        for sim in range(self.num_simulations):
            node = self._select(root)
            reward = self._simulate(node, intent, schema)
            self._backpropagate(node, reward)

        # Select best path from root
        best_record = self._extract_best_record(root, schema)

        return best_record

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCT"""
        while not node.is_fully_expanded():
            if len(node.children) == 0:
                return node
            node = node.best_child(self.exploration_weight)
        return node

    def _simulate(self, node: MCTSNode, intent: Intent, schema: Dict[str, str]) -> float:
        """Simulation phase: generate and evaluate a complete record"""

        # Start with current node's partial record
        record = node.record.copy()

        # Fill in missing fields using Monte Carlo-like approach
        monte_carlo = MonteCarloEngine(self.llm)
        mc_results = monte_carlo.generate(intent, schema, count=1)

        if mc_results:
            mc_record = mc_results[0].data

            # Fill in None values from simulation
            for field, value in record.items():
                if value is None and field in mc_record:
                    record[field] = mc_record[field]

        # Evaluate quality of generated record
        reward = self._evaluate_record(record, intent, schema)

        # Create child node for this simulation
        child = MCTSNode(record, parent=node)
        node.children.append(child)

        return reward

    def _evaluate_record(self, record: Dict[str, Any], intent: Intent, schema: Dict[str, str]) -> float:
        """Evaluate quality of a generated record"""

        score = 0.0
        max_score = len(schema)

        # Check completeness
        for field in schema.keys():
            if field in record and record[field] is not None:
                score += 1.0

        # Check type correctness
        for field, field_type in schema.items():
            if field not in record or record[field] is None:
                continue

            value = record[field]

            if field_type == 'number' and isinstance(value, (int, float)):
                score += 0.5
            elif field_type == 'string' and isinstance(value, str):
                score += 0.5
            elif field_type == 'date' and isinstance(value, str):
                score += 0.5

        # Normalize to 0-1
        return score / (max_score * 1.5)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase: update node statistics"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _extract_best_record(self, root: MCTSNode, schema: Dict[str, str]) -> Dict[str, Any]:
        """Extract best record from MCTS tree"""

        # Find most visited child (best proven strategy)
        if not root.children:
            # Fallback to Monte Carlo if no children
            monte_carlo = MonteCarloEngine(self.llm)
            from .intent_engine import Intent
            fallback_intent = Intent(
                data_type="record",
                count=1,
                format="json",
                purpose="generation",
                domain=None,
                geography=None,
                additional_context=None
            )
            results = monte_carlo.generate(fallback_intent, schema, count=1)
            return results[0].data if results else {}

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.record


class HybridReasoningEngine(ReasoningEngine):
    """
    Hybrid Reasoning Engine

    Combines multiple reasoning strategies for optimal data generation
    Integrates: Monte Carlo, Beam Search, Chain-of-Thought, Tree-of-Thoughts, MCTS

    Best for: Maximum quality, diverse datasets, adaptive generation
    """

    def __init__(
        self,
        llm_provider,
        strategy_weights: Optional[Dict[str, float]] = None,
        adaptive: bool = True
    ):
        super().__init__(llm_provider)

        # Default strategy weights
        self.strategy_weights = strategy_weights or {
            'monte_carlo': 0.3,
            'beam_search': 0.25,
            'chain_of_thought': 0.2,
            'tree_of_thoughts': 0.15,
            'mcts': 0.1
        }

        self.adaptive = adaptive
        self.performance_history = defaultdict(list)

        # Initialize sub-engines
        self.engines = {
            'monte_carlo': MonteCarloEngine(llm_provider),
            'beam_search': BeamSearchEngine(llm_provider, beam_width=5),
            'chain_of_thought': ChainOfThoughtEngine(llm_provider),
            'tree_of_thoughts': TreeOfThoughtsEngine(llm_provider, num_branches=3),
            'mcts': MCTSEngine(llm_provider, num_simulations=50)
        }

    def generate(self, intent: Intent, schema: Dict[str, str], count: int = 1) -> List[GenerationResult]:
        """
        Generate data using hybrid approach

        Combines multiple reasoning strategies based on weights and performance
        """
        results = []

        for i in range(count):
            # Select strategy based on weights
            if self.adaptive and i > 10:
                # Adapt weights based on performance
                self._adapt_weights()

            strategy = self._select_strategy()

            # Generate using selected strategy
            try:
                engine = self.engines[strategy]
                result = engine.generate(intent, schema, count=1)[0]

                # Track performance
                self.performance_history[strategy].append(result.score)

                # Update metadata
                result.reasoning = f"Hybrid: {strategy} (weight: {self.strategy_weights[strategy]:.2f})"
                result.metadata = result.metadata or {}
                result.metadata.update({
                    'hybrid_strategy': strategy,
                    'strategy_weight': self.strategy_weights[strategy],
                    'adaptive': self.adaptive
                })

                results.append(result)

            except Exception as e:
                print(f"Warning: Strategy {strategy} failed, using fallback: {e}")
                # Fallback to Monte Carlo
                fallback = self.engines['monte_carlo'].generate(intent, schema, count=1)[0]
                fallback.reasoning = f"Hybrid: monte_carlo (fallback from {strategy})"
                results.append(fallback)

        return results

    def _select_strategy(self) -> str:
        """Select strategy based on current weights"""
        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())

        # Normalize weights
        total = sum(weights)
        if total == 0:
            # Equal weights if all are zero
            weights = [1.0 / len(strategies)] * len(strategies)
        else:
            weights = [w / total for w in weights]

        return np.random.choice(strategies, p=weights)

    def _adapt_weights(self):
        """Adapt strategy weights based on performance history"""

        if not self.adaptive:
            return

        # Calculate average performance for each strategy
        avg_performance = {}
        for strategy, scores in self.performance_history.items():
            if scores:
                avg_performance[strategy] = sum(scores) / len(scores)
            else:
                avg_performance[strategy] = 0.5  # Default

        # Update weights proportionally to performance
        total_performance = sum(avg_performance.values())
        if total_performance > 0:
            for strategy in self.strategy_weights.keys():
                perf = avg_performance.get(strategy, 0.5)
                self.strategy_weights[strategy] = perf / total_performance

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""

        summary = {}
        for strategy, scores in self.performance_history.items():
            if scores:
                summary[strategy] = {
                    'avg_score': sum(scores) / len(scores),
                    'num_uses': len(scores),
                    'current_weight': self.strategy_weights.get(strategy, 0.0)
                }

        return summary


class ReasoningEngineFactory:
    """Factory for creating reasoning engines"""

    @staticmethod
    def create(engine_type: str, llm_provider, **kwargs) -> ReasoningEngine:
        """
        Create a reasoning engine

        Args:
            engine_type: Type of engine (monte_carlo, beam_search, chain_of_thought,
                        tree_of_thoughts, mcts, hybrid)
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
            'tot': TreeOfThoughtsEngine,
            'mcts': MCTSEngine,
            'hybrid': HybridReasoningEngine
        }

        engine_type = engine_type.lower()
        if engine_type not in engines:
            raise ValueError(f"Unknown engine type: {engine_type}. Available: {list(engines.keys())}")

        return engines[engine_type](llm_provider, **kwargs)
