"""
Comprehensive tests for multi-objective optimization module
"""

import pytest
import numpy as np
from agent.multi_objective import (
    Objective, Solution, MultiObjectiveOptimizer
)


class TestObjective:
    """Test Objective class"""

    def test_objective_creation(self):
        """Test creating an objective"""
        obj = Objective(name="accuracy", value=0.85, maximize=True, weight=0.5)
        assert obj.name == "accuracy"
        assert obj.value == 0.85
        assert obj.maximize is True
        assert obj.weight == 0.5

    def test_objective_negative_weight(self):
        """Test that negative weight raises error"""
        with pytest.raises(ValueError):
            Objective(name="test", value=1.0, maximize=True, weight=-0.1)


class TestSolution:
    """Test Solution class"""

    def test_solution_creation(self):
        """Test creating a solution"""
        objectives = {
            'accuracy': Objective('accuracy', 0.9, True),
            'latency': Objective('latency', 100, False)
        }
        sol = Solution(id="prompt_1", objectives=objectives)
        assert sol.id == "prompt_1"
        assert len(sol.objectives) == 2

    def test_get_objective_value(self):
        """Test getting objective value"""
        objectives = {
            'accuracy': Objective('accuracy', 0.9, True)
        }
        sol = Solution(id="test", objectives=objectives)
        assert sol.get_objective_value('accuracy') == 0.9

    def test_get_nonexistent_objective(self):
        """Test getting nonexistent objective raises error"""
        sol = Solution(id="test", objectives={})
        with pytest.raises(KeyError):
            sol.get_objective_value('nonexistent')

    def test_dominance_simple(self):
        """Test Pareto dominance comparison"""
        # Solution A: accuracy=0.9, latency=100
        sol_a = Solution(
            id="a",
            objectives={
                'accuracy': Objective('accuracy', 0.9, True),
                'latency': Objective('latency', 100, False)
            }
        )

        # Solution B: accuracy=0.8, latency=120
        sol_b = Solution(
            id="b",
            objectives={
                'accuracy': Objective('accuracy', 0.8, True),
                'latency': Objective('latency', 120, False)
            }
        )

        # A dominates B (higher accuracy, lower latency)
        assert sol_a.dominates(sol_b)
        assert not sol_b.dominates(sol_a)

    def test_dominance_tradeoff(self):
        """Test dominance with trade-offs"""
        # Solution A: accuracy=0.9, latency=150
        sol_a = Solution(
            id="a",
            objectives={
                'accuracy': Objective('accuracy', 0.9, True),
                'latency': Objective('latency', 150, False)
            }
        )

        # Solution B: accuracy=0.8, latency=100
        sol_b = Solution(
            id="b",
            objectives={
                'accuracy': Objective('accuracy', 0.8, True),
                'latency': Objective('latency', 100, False)
            }
        )

        # Neither dominates (trade-off)
        assert not sol_a.dominates(sol_b)
        assert not sol_b.dominates(sol_a)

    def test_solution_to_dict(self):
        """Test converting solution to dict"""
        objectives = {
            'accuracy': Objective('accuracy', 0.9, True)
        }
        sol = Solution(id="test", objectives=objectives, metadata={'foo': 'bar'})
        d = sol.to_dict()
        assert d['id'] == "test"
        assert 'objectives' in d
        assert d['metadata']['foo'] == 'bar'


class TestMultiObjectiveOptimizer:
    """Test MultiObjectiveOptimizer"""

    def test_optimizer_initialization(self):
        """Test initializing optimizer"""
        optimizer = MultiObjectiveOptimizer()
        assert len(optimizer.solutions) == 0
        assert len(optimizer.pareto_fronts) == 0

    def test_add_solution(self):
        """Test adding solutions"""
        optimizer = MultiObjectiveOptimizer()
        sol = Solution(
            id="test",
            objectives={'accuracy': Objective('accuracy', 0.9, True)}
        )
        optimizer.add_solution(sol)
        assert len(optimizer.solutions) == 1

    def test_add_from_metrics(self):
        """Test creating solution from metrics"""
        optimizer = MultiObjectiveOptimizer()

        metrics = {
            'accuracy': 0.9,
            'latency': 100,
            'cost': 0.05
        }

        objective_config = {
            'accuracy': {'maximize': True, 'weight': 0.5},
            'latency': {'maximize': False, 'weight': 0.3},
            'cost': {'maximize': False, 'weight': 0.2}
        }

        sol = optimizer.add_from_metrics("prompt_1", metrics, objective_config)
        assert sol.id == "prompt_1"
        assert len(sol.objectives) == 3
        assert sol.objectives['accuracy'].value == 0.9

    def test_pareto_frontier_simple(self):
        """Test calculating Pareto frontier"""
        optimizer = MultiObjectiveOptimizer()

        # Add 3 solutions
        # Sol1: accuracy=0.9, latency=100 (Pareto optimal)
        optimizer.add_solution(Solution(
            id="sol1",
            objectives={
                'accuracy': Objective('accuracy', 0.9, True),
                'latency': Objective('latency', 100, False)
            }
        ))

        # Sol2: accuracy=0.8, latency=120 (dominated by sol1)
        optimizer.add_solution(Solution(
            id="sol2",
            objectives={
                'accuracy': Objective('accuracy', 0.8, True),
                'latency': Objective('latency', 120, False)
            }
        ))

        # Sol3: accuracy=0.85, latency=80 (Pareto optimal - trade-off)
        optimizer.add_solution(Solution(
            id="sol3",
            objectives={
                'accuracy': Objective('accuracy', 0.85, True),
                'latency': Objective('latency', 80, False)
            }
        ))

        frontier = optimizer.calculate_pareto_frontier()

        # Should have sol1 and sol3 in frontier
        assert len(frontier) == 2
        frontier_ids = {sol.id for sol in frontier}
        assert 'sol1' in frontier_ids
        assert 'sol3' in frontier_ids
        assert 'sol2' not in frontier_ids

    def test_pareto_fronts_multiple_levels(self):
        """Test multiple Pareto front levels"""
        optimizer = MultiObjectiveOptimizer()

        # First front
        optimizer.add_solution(Solution(
            id="f1_1",
            objectives={
                'accuracy': Objective('accuracy', 0.95, True),
                'latency': Objective('latency', 50, False)
            }
        ))

        # Second front (dominated by f1_1)
        optimizer.add_solution(Solution(
            id="f2_1",
            objectives={
                'accuracy': Objective('accuracy', 0.85, True),
                'latency': Objective('latency', 80, False)
            }
        ))

        # Third front (dominated by f2_1)
        optimizer.add_solution(Solution(
            id="f3_1",
            objectives={
                'accuracy': Objective('accuracy', 0.75, True),
                'latency': Objective('latency', 100, False)
            }
        ))

        optimizer.calculate_pareto_frontier()

        assert len(optimizer.pareto_fronts) == 3
        assert len(optimizer.pareto_fronts[0]) == 1  # First front
        assert optimizer.pareto_fronts[0][0].id == "f1_1"

    def test_best_compromise(self):
        """Test finding best compromise solution"""
        optimizer = MultiObjectiveOptimizer()

        optimizer.add_solution(Solution(
            id="sol1",
            objectives={
                'accuracy': Objective('accuracy', 0.9, True, weight=0.7),
                'latency': Objective('latency', 100, False, weight=0.3)
            }
        ))

        optimizer.add_solution(Solution(
            id="sol2",
            objectives={
                'accuracy': Objective('accuracy', 0.85, True, weight=0.7),
                'latency': Objective('latency', 50, False, weight=0.3)
            }
        ))

        best = optimizer.get_best_compromise()
        assert best is not None
        assert best.id in ['sol1', 'sol2']

    def test_analyze_tradeoffs(self):
        """Test trade-off analysis"""
        optimizer = MultiObjectiveOptimizer()

        # Add solutions with clear trade-off
        for i in range(5):
            accuracy = 0.95 - (i * 0.05)
            latency = 50 + (i * 20)
            optimizer.add_solution(Solution(
                id=f"sol{i}",
                objectives={
                    'accuracy': Objective('accuracy', accuracy, True),
                    'latency': Objective('latency', latency, False)
                }
            ))

        analysis = optimizer.analyze_tradeoffs()

        assert 'pareto_frontier_size' in analysis
        assert 'objective_ranges' in analysis
        assert 'trade_offs' in analysis
        assert analysis['pareto_frontier_size'] > 0

    def test_get_recommendations(self):
        """Test getting solution recommendations"""
        optimizer = MultiObjectiveOptimizer()

        # Add diverse solutions
        optimizer.add_solution(Solution(
            id="high_acc",
            objectives={
                'accuracy': Objective('accuracy', 0.95, True),
                'cost': Objective('cost', 0.10, False)
            }
        ))

        optimizer.add_solution(Solution(
            id="low_cost",
            objectives={
                'accuracy': Objective('accuracy', 0.80, True),
                'cost': Objective('cost', 0.02, False)
            }
        ))

        recommendations = optimizer.get_recommendations()

        assert len(recommendations) > 0
        assert any(r['type'] == 'best_compromise' for r in recommendations)
        assert all('explanation' in r for r in recommendations)

    def test_hypervolume_calculation(self):
        """Test hypervolume metric calculation"""
        optimizer = MultiObjectiveOptimizer(
            reference_point={'accuracy': 0.0, 'latency': 200}
        )

        optimizer.add_solution(Solution(
            id="sol1",
            objectives={
                'accuracy': Objective('accuracy', 0.9, True),
                'latency': Objective('latency', 100, False)
            }
        ))

        hypervolume = optimizer.calculate_hypervolume()
        assert hypervolume >= 0

    def test_export_results(self, tmp_path):
        """Test exporting results to JSON"""
        optimizer = MultiObjectiveOptimizer()

        optimizer.add_solution(Solution(
            id="test",
            objectives={'accuracy': Objective('accuracy', 0.9, True)}
        ))

        filepath = tmp_path / "results.json"
        optimizer.export_results(str(filepath))

        assert filepath.exists()

        import json
        with open(filepath) as f:
            data = json.load(f)

        assert 'solutions' in data
        assert 'analysis' in data
        assert 'recommendations' in data

    def test_visualize_pareto_frontier(self):
        """Test getting visualization data"""
        optimizer = MultiObjectiveOptimizer()

        optimizer.add_solution(Solution(
            id="sol1",
            objectives={
                'accuracy': Objective('accuracy', 0.9, True),
                'latency': Objective('latency', 100, False)
            }
        ))

        viz_data = optimizer.visualize_pareto_frontier('accuracy', 'latency')

        assert 'all_solutions' in viz_data
        assert 'pareto_frontier' in viz_data
        assert 'obj1' in viz_data
        assert 'obj2' in viz_data


def test_integration_realistic_scenario():
    """Test realistic multi-objective optimization scenario"""
    optimizer = MultiObjectiveOptimizer()

    # Simulate 10 prompt variants with different characteristics
    prompts_data = [
        {'id': 'prompt_1', 'accuracy': 0.92, 'latency': 1.2, 'cost': 0.08, 'tokens': 1500},
        {'id': 'prompt_2', 'accuracy': 0.88, 'latency': 0.8, 'cost': 0.05, 'tokens': 1000},
        {'id': 'prompt_3', 'accuracy': 0.95, 'latency': 1.8, 'cost': 0.12, 'tokens': 2000},
        {'id': 'prompt_4', 'accuracy': 0.85, 'latency': 0.6, 'cost': 0.03, 'tokens': 800},
        {'id': 'prompt_5', 'accuracy': 0.90, 'latency': 1.0, 'cost': 0.06, 'tokens': 1200},
        {'id': 'prompt_6', 'accuracy': 0.87, 'latency': 0.9, 'cost': 0.055, 'tokens': 1100},
        {'id': 'prompt_7', 'accuracy': 0.93, 'latency': 1.4, 'cost': 0.09, 'tokens': 1600},
        {'id': 'prompt_8', 'accuracy': 0.86, 'latency': 0.7, 'cost': 0.04, 'tokens': 900},
        {'id': 'prompt_9', 'accuracy': 0.91, 'latency': 1.1, 'cost': 0.07, 'tokens': 1300},
        {'id': 'prompt_10', 'accuracy': 0.89, 'latency': 0.95, 'cost': 0.06, 'tokens': 1150},
    ]

    objective_config = {
        'accuracy': {'maximize': True, 'weight': 0.4},
        'latency': {'maximize': False, 'weight': 0.3},
        'cost': {'maximize': False, 'weight': 0.2},
        'tokens': {'maximize': False, 'weight': 0.1}
    }

    # Add all solutions
    for data in prompts_data:
        optimizer.add_from_metrics(data['id'], data, objective_config)

    # Calculate Pareto frontier
    frontier = optimizer.calculate_pareto_frontier()

    # Should have multiple solutions in frontier (trade-offs)
    assert len(frontier) > 1
    assert len(frontier) < len(prompts_data)

    # Get analysis
    analysis = optimizer.analyze_tradeoffs()
    assert analysis['pareto_frontier_size'] == len(frontier)
    assert 'objective_ranges' in analysis

    # Get recommendations
    recommendations = optimizer.get_recommendations()
    assert len(recommendations) > 0

    # Best compromise should be in Pareto frontier
    best = optimizer.get_best_compromise()
    assert best in frontier
