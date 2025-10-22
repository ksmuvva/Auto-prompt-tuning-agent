"""
Multi-Objective Optimization Module

This module implements multi-objective optimization for prompt tuning using:
1. Pareto frontier calculation
2. NSGA-II inspired non-dominated sorting
3. Multiple objective optimization (accuracy, latency, cost, etc.)
4. Hypervolume calculation
5. Trade-off analysis

Program of Thoughts:
1. Define multiple objectives (maximize accuracy, minimize cost, minimize latency)
2. Calculate Pareto dominance relationships
3. Find Pareto optimal solutions (non-dominated set)
4. Rank solutions by dominance level
5. Calculate hypervolume metric for solution quality
6. Provide trade-off recommendations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json


@dataclass
class Objective:
    """Represents a single optimization objective"""
    name: str
    value: float
    maximize: bool  # True to maximize, False to minimize
    weight: float = 1.0  # Importance weight

    def __post_init__(self):
        """Validate objective parameters"""
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")


@dataclass
class Solution:
    """Represents a solution in multi-objective space"""
    id: str  # Prompt template name or ID
    objectives: Dict[str, Objective]
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_objective_value(self, name: str) -> float:
        """Get value for a specific objective"""
        if name not in self.objectives:
            raise KeyError(f"Objective '{name}' not found")
        return self.objectives[name].value

    def dominates(self, other: 'Solution') -> bool:
        """
        Check if this solution dominates another (Pareto dominance)

        Solution A dominates B if:
        - A is no worse than B in all objectives
        - A is strictly better than B in at least one objective
        """
        at_least_one_better = False

        for name, obj in self.objectives.items():
            if name not in other.objectives:
                continue

            self_val = obj.value
            other_val = other.objectives[name].value

            if obj.maximize:
                # For maximization: higher is better
                if self_val < other_val:
                    return False  # Worse in this objective
                if self_val > other_val:
                    at_least_one_better = True
            else:
                # For minimization: lower is better
                if self_val > other_val:
                    return False  # Worse in this objective
                if self_val < other_val:
                    at_least_one_better = True

        return at_least_one_better

    def to_dict(self) -> Dict:
        """Convert solution to dictionary"""
        return {
            'id': self.id,
            'objectives': {
                name: {
                    'value': obj.value,
                    'maximize': obj.maximize,
                    'weight': obj.weight
                }
                for name, obj in self.objectives.items()
            },
            'metadata': self.metadata
        }


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization engine using Pareto optimality

    Key concepts:
    - Pareto frontier: Set of non-dominated solutions
    - Dominance: A solution is dominated if another is better in all objectives
    - Hypervolume: Quality metric for solution set
    """

    def __init__(self, reference_point: Optional[Dict[str, float]] = None):
        """
        Initialize optimizer

        Args:
            reference_point: Reference point for hypervolume calculation
                           (worst acceptable values for each objective)
        """
        self.reference_point = reference_point or {}
        self.solutions: List[Solution] = []
        self.pareto_fronts: List[List[Solution]] = []

    def add_solution(self, solution: Solution):
        """Add a solution to the optimization set"""
        self.solutions.append(solution)

    def add_from_metrics(self, prompt_id: str, metrics: Dict,
                        objective_config: Dict[str, Dict]) -> Solution:
        """
        Create and add solution from prompt metrics

        Args:
            prompt_id: Identifier for the prompt
            metrics: Dictionary of metric values
            objective_config: Configuration for each objective:
                {
                    'accuracy': {'maximize': True, 'weight': 0.3},
                    'latency': {'maximize': False, 'weight': 0.2},
                    ...
                }

        Returns:
            Created Solution object
        """
        objectives = {}
        for name, config in objective_config.items():
            if name in metrics:
                objectives[name] = Objective(
                    name=name,
                    value=float(metrics[name]),
                    maximize=config.get('maximize', True),
                    weight=config.get('weight', 1.0)
                )

        solution = Solution(
            id=prompt_id,
            objectives=objectives,
            metadata={'raw_metrics': metrics}
        )
        self.add_solution(solution)
        return solution

    def _fast_non_dominated_sort(self) -> List[List[Solution]]:
        """
        NSGA-II inspired non-dominated sorting

        Program of Thoughts:
        1. For each solution, find which solutions it dominates
        2. Count how many solutions dominate it
        3. Solutions with zero domination count form first Pareto front
        4. Recursively find subsequent fronts

        Returns:
            List of Pareto fronts (list of lists of solutions)
        """
        # Track domination relationships
        domination_count = {i: 0 for i in range(len(self.solutions))}
        dominated_solutions = {i: [] for i in range(len(self.solutions))}

        # Calculate dominance relationships
        for i, sol_i in enumerate(self.solutions):
            for j, sol_j in enumerate(self.solutions):
                if i == j:
                    continue

                if sol_i.dominates(sol_j):
                    dominated_solutions[i].append(j)
                elif sol_j.dominates(sol_i):
                    domination_count[i] += 1

        # Build fronts
        fronts = []
        current_front_indices = [i for i, count in domination_count.items() if count == 0]

        while current_front_indices:
            fronts.append([self.solutions[i] for i in current_front_indices])
            next_front_indices = []

            for i in current_front_indices:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front_indices.append(j)

            current_front_indices = next_front_indices

        return fronts

    def calculate_pareto_frontier(self) -> List[Solution]:
        """
        Calculate the Pareto frontier (first non-dominated front)

        Returns:
            List of Pareto optimal solutions
        """
        if not self.solutions:
            return []

        self.pareto_fronts = self._fast_non_dominated_sort()
        return self.pareto_fronts[0] if self.pareto_fronts else []

    def calculate_hypervolume(self, front: List[Solution] = None) -> float:
        """
        Calculate hypervolume indicator for a Pareto front

        Hypervolume measures the volume of objective space dominated by the front.
        Higher hypervolume = better quality solution set.

        Simplified 2D calculation for now (can be extended to n-dimensions)

        Args:
            front: List of solutions (uses Pareto frontier if not specified)

        Returns:
            Hypervolume value
        """
        if front is None:
            front = self.calculate_pareto_frontier()

        if not front:
            return 0.0

        # For simplicity, calculate 2D hypervolume for first two objectives
        objective_names = list(front[0].objectives.keys())
        if len(objective_names) < 2:
            return 0.0

        obj1_name, obj2_name = objective_names[:2]

        # Get reference point
        ref1 = self.reference_point.get(obj1_name, 0.0)
        ref2 = self.reference_point.get(obj2_name, 0.0)

        # Extract points and sort
        points = []
        for sol in front:
            obj1 = sol.objectives[obj1_name]
            obj2 = sol.objectives[obj2_name]

            # Normalize for maximization
            val1 = obj1.value if obj1.maximize else -obj1.value
            val2 = obj2.value if obj2.maximize else -obj2.value

            points.append((val1, val2))

        # Sort by first objective
        points.sort(reverse=True)

        # Calculate hypervolume using step function
        hypervolume = 0.0
        prev_x = points[0][0]

        for i, (x, y) in enumerate(points):
            if i == 0:
                width = x - ref1
                height = y - ref2
            else:
                width = x - ref1
                height = max(y - ref2, points[i-1][1] - ref2)
                width = points[i-1][0] - x

            hypervolume += width * height if width > 0 and height > 0 else 0

        return max(0.0, hypervolume)

    def get_best_compromise(self, weights: Optional[Dict[str, float]] = None) -> Optional[Solution]:
        """
        Find best compromise solution using weighted sum

        Args:
            weights: Optional custom weights for objectives

        Returns:
            Solution with best weighted score from Pareto frontier
        """
        pareto_frontier = self.calculate_pareto_frontier()
        if not pareto_frontier:
            return None

        best_solution = None
        best_score = float('-inf')

        for solution in pareto_frontier:
            score = 0.0
            total_weight = 0.0

            for name, obj in solution.objectives.items():
                w = weights.get(name, obj.weight) if weights else obj.weight
                normalized_value = obj.value if obj.maximize else -obj.value
                score += w * normalized_value
                total_weight += w

            # Normalize by total weight
            if total_weight > 0:
                score /= total_weight

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def analyze_tradeoffs(self) -> Dict:
        """
        Analyze trade-offs between objectives in Pareto frontier

        Returns:
            Dictionary with trade-off analysis
        """
        pareto_frontier = self.calculate_pareto_frontier()

        if not pareto_frontier:
            return {
                'pareto_frontier_size': 0,
                'total_solutions': len(self.solutions),
                'trade_offs': []
            }

        # Get objective names
        objective_names = list(pareto_frontier[0].objectives.keys())

        # Calculate ranges in Pareto frontier
        ranges = {}
        for obj_name in objective_names:
            values = [sol.get_objective_value(obj_name) for sol in pareto_frontier]
            ranges[obj_name] = {
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }

        # Identify trade-offs (negative correlations in Pareto frontier)
        trade_offs = []
        for i, obj1 in enumerate(objective_names):
            for j, obj2 in enumerate(objective_names[i+1:], i+1):
                values1 = [sol.get_objective_value(obj1) for sol in pareto_frontier]
                values2 = [sol.get_objective_value(obj2) for sol in pareto_frontier]

                if len(values1) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]

                    if abs(correlation) > 0.3:  # Significant correlation
                        trade_offs.append({
                            'objective_1': obj1,
                            'objective_2': obj2,
                            'correlation': float(correlation),
                            'type': 'trade-off' if correlation < 0 else 'synergy'
                        })

        return {
            'pareto_frontier_size': len(pareto_frontier),
            'total_solutions': len(self.solutions),
            'total_fronts': len(self.pareto_fronts),
            'hypervolume': self.calculate_hypervolume(pareto_frontier),
            'objective_ranges': ranges,
            'trade_offs': trade_offs,
            'pareto_solutions': [sol.id for sol in pareto_frontier]
        }

    def get_recommendations(self, user_preferences: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Get solution recommendations based on user preferences

        Args:
            user_preferences: Optional preference weights for objectives

        Returns:
            List of recommended solutions with explanations
        """
        pareto_frontier = self.calculate_pareto_frontier()

        if not pareto_frontier:
            return []

        recommendations = []

        # 1. Best compromise solution
        best_compromise = self.get_best_compromise(user_preferences)
        if best_compromise:
            recommendations.append({
                'type': 'best_compromise',
                'solution': best_compromise.to_dict(),
                'explanation': 'Best overall balance across all objectives'
            })

        # 2. Best for each individual objective
        for obj_name in pareto_frontier[0].objectives.keys():
            obj_config = pareto_frontier[0].objectives[obj_name]

            if obj_config.maximize:
                best_sol = max(pareto_frontier,
                             key=lambda s: s.get_objective_value(obj_name))
            else:
                best_sol = min(pareto_frontier,
                             key=lambda s: s.get_objective_value(obj_name))

            recommendations.append({
                'type': f'best_for_{obj_name}',
                'solution': best_sol.to_dict(),
                'explanation': f'Best performance for {obj_name}'
            })

        # 3. Most balanced solution (minimal variance across normalized objectives)
        normalized_variances = []
        for solution in pareto_frontier:
            normalized_values = []
            for obj_name, obj in solution.objectives.items():
                # Get range from all solutions
                all_values = [s.get_objective_value(obj_name) for s in self.solutions]
                val_range = max(all_values) - min(all_values)

                if val_range > 0:
                    normalized = (obj.value - min(all_values)) / val_range
                else:
                    normalized = 0.5

                normalized_values.append(normalized)

            variance = np.var(normalized_values)
            normalized_variances.append((variance, solution))

        if normalized_variances:
            most_balanced = min(normalized_variances, key=lambda x: x[0])[1]
            recommendations.append({
                'type': 'most_balanced',
                'solution': most_balanced.to_dict(),
                'explanation': 'Most balanced performance across objectives'
            })

        return recommendations

    def export_results(self, filepath: str):
        """Export optimization results to JSON"""
        results = {
            'solutions': [sol.to_dict() for sol in self.solutions],
            'pareto_fronts': [
                [sol.id for sol in front]
                for front in self.pareto_fronts
            ],
            'analysis': self.analyze_tradeoffs(),
            'recommendations': self.get_recommendations()
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

    def visualize_pareto_frontier(self, obj1: str, obj2: str) -> Dict:
        """
        Get data for visualizing Pareto frontier in 2D

        Args:
            obj1: First objective name (x-axis)
            obj2: Second objective name (y-axis)

        Returns:
            Dictionary with plot data
        """
        pareto_frontier = self.calculate_pareto_frontier()

        # All solutions
        all_points = []
        for sol in self.solutions:
            if obj1 in sol.objectives and obj2 in sol.objectives:
                all_points.append({
                    'id': sol.id,
                    'x': sol.get_objective_value(obj1),
                    'y': sol.get_objective_value(obj2),
                    'is_pareto': sol in pareto_frontier
                })

        # Pareto frontier points
        pareto_points = []
        for sol in pareto_frontier:
            if obj1 in sol.objectives and obj2 in sol.objectives:
                pareto_points.append({
                    'id': sol.id,
                    'x': sol.get_objective_value(obj1),
                    'y': sol.get_objective_value(obj2)
                })

        return {
            'all_solutions': all_points,
            'pareto_frontier': pareto_points,
            'obj1': {
                'name': obj1,
                'maximize': pareto_frontier[0].objectives[obj1].maximize if pareto_frontier else True
            },
            'obj2': {
                'name': obj2,
                'maximize': pareto_frontier[0].objectives[obj2].maximize if pareto_frontier else True
            }
        }
