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
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95 = 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
