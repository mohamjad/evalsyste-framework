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
            )
    
    def _random_sample(self) -> Dict[str, float]:
        """Sample random point from space."""
        point = {}
        for param in self.space:
            if param.log_scale:
                min_log = math.log(param.min_val)
                max_log = math.log(param.max_val)
                val = math.exp(random.uniform(min_log, max_log))
            else:
                val = random.uniform(param.min_val, param.max_val)
            
            if param.param_type == 'discrete':
                val = round(val)
            
            point[param.name] = val
        return point
    
    def _point_to_vector(self, point: Dict[str, float]) -> np.ndarray:
        """Convert point dict to vector for GP."""
        return np.array([point[p.name] for p in self.space])
    
    def _vector_to_point(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert vector to point dict."""
        point = {}
        for i, param in enumerate(self.space):
            val = float(vector[i])
            # Clip to bounds
            val = max(param.min_val, min(param.max_val, val))
            if param.param_type == 'discrete':
                val = round(val)
            point[param.name] = val
        return point
    
    def _expected_improvement(self, X_candidate: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function.
        
        EI(x) = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)
        where Z = (μ(x) - f_best - ξ) / σ(x)
        """
        if not self.gp or len(self.y) == 0:
            return np.ones(len(X_candidate))
