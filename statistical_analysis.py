"""
Statistical Analysis Module

Provides basic confidence intervals, significance checks, and trend
analysis helpers for framework scores.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import statistics


class StatisticalAnalyzer:
    """
    Statistical analysis for stability metrics.
    
    Provides:
    - Confidence intervals
    - Significance testing
    - Trend analysis with p-values
    - Bootstrap resampling
