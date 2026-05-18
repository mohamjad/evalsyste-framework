"""Tests for Bayesian optimization utilities."""

from hyperparameter_optimization import BayesianOptimizer, HyperparameterSpace


def test_random_sample_respects_bounds():
    """Random samples must stay inside the declared search space."""
    optimizer = BayesianOptimizer(
        space=[
            HyperparameterSpace("threshold", 0.1, 0.9),
            HyperparameterSpace("depth", 1, 5, param_type="discrete"),
