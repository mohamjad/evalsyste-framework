"""
Bayesian Hyperparameter Optimization

Optional Gaussian-process-based search utilities for tuning thresholds
and model parameters when scikit-learn is available.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

# Optional: use scikit-learn for GP if available
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GaussianProcessRegressor = None


@dataclass
class HyperparameterSpace:
    """Define search space for hyperparameters."""
    name: str
    min_val: float
    max_val: float
    param_type: str = 'continuous'  # 'continuous' or 'discrete'
    log_scale: bool = False  # Use log scale for optimization


class BayesianOptimizer:
    """
    Bayesian optimization using Gaussian Process regression.

    Optimizes expensive black-box functions by:
    1. Building GP model of objective function
    2. Using acquisition function (Expected Improvement) to select next point
    3. Evaluating and updating GP model
    4. Repeating until convergence
    """
    
    def __init__(self, 
                 space: List[HyperparameterSpace],
                 acquisition_function: str = 'ei',
                 n_initial: int = 5,
                 n_iterations: int = 50):
        self.space = space
        self.acquisition_function = acquisition_function
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        
        # Storage
        self.X = []  # Evaluated points
        self.y = []  # Objective values
        
        # GP model
        self.gp = None
        if SKLEARN_AVAILABLE:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=1e-6
