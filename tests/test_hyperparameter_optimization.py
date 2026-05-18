"""Tests for Bayesian optimization utilities."""

from hyperparameter_optimization import BayesianOptimizer, HyperparameterSpace


def test_random_sample_respects_bounds():
    """Random samples must stay inside the declared search space."""
    optimizer = BayesianOptimizer(
        space=[
            HyperparameterSpace("threshold", 0.1, 0.9),
            HyperparameterSpace("depth", 1, 5, param_type="discrete"),
        ],
        n_initial=2,
        n_iterations=2,
    )

    sample = optimizer._random_sample()

    assert 0.1 <= sample["threshold"] <= 0.9
    assert 1 <= sample["depth"] <= 5
    assert float(sample["depth"]).is_integer()


def test_optimizer_finds_reasonable_point_for_simple_objective():
    """Optimization should improve on a simple smooth objective."""
    optimizer = BayesianOptimizer(
        space=[HyperparameterSpace("x", -2.0, 2.0)],
        n_initial=3,
        n_iterations=8,
    )

    def objective(point):
        return 1.0 - (point["x"] - 0.5) ** 2
