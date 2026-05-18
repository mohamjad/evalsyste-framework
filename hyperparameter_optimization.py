"""
Bayesian Hyperparameter Optimization

Optional Gaussian-process-based search utilities for tuning thresholds
and model parameters when scikit-learn is available.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
