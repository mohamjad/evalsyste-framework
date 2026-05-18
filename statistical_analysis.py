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
    
    def calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for a sample.
        
        Uses t-distribution for small samples, normal for large.
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        if not values:
            return 0.0, 0.0, 0.0
        
        n = len(values)
        mean = statistics.mean(values)
        
        if n == 1:
            return mean, mean, mean
        
        # Sample standard deviation
        if n > 1:
            std_dev = statistics.stdev(values)
        else:
            std_dev = 0.0
        
        # Standard error
        std_error = std_dev / math.sqrt(n)
        
        # Critical value (approximate t-distribution)
        # For 95% CI: ~1.96 for large n, higher for small n
        if n >= 30:
            t_critical = 1.96  # Normal approximation
        elif n >= 10:
            t_critical = 2.262  # t-distribution, df=9, alpha=0.05
        elif n >= 5:
            t_critical = 2.776  # t-distribution, df=4
        else:
            t_critical = 3.182  # t-distribution, df=3
        
        margin = t_critical * std_error
        
        return mean, mean - margin, mean + margin
    
    def test_significance(self, sample1: List[float], sample2: List[float]) -> Tuple[bool, float, str]:
        """
        Test if two samples are significantly different.
        
        Uses Welch's t-test (unequal variances).
        
        Returns:
            (is_significant, p_value, interpretation)
        """
        if not sample1 or not sample2:
            return False, 1.0, "insufficient data"
        
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        
        if n1 < 2 or n2 < 2:
            return False, 1.0, "insufficient data for test"
        
        # Sample variances
        var1 = statistics.variance(sample1) if n1 > 1 else 0.0
        var2 = statistics.variance(sample2) if n2 > 1 else 0.0
        
        # Welch's t-test statistic
