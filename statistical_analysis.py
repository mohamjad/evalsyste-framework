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
        std_error = math.sqrt(var1/n1 + var2/n2)
        if std_error == 0:
            return False, 1.0, "zero variance"
        
        t_statistic = (mean1 - mean2) / std_error
        
        # Degrees of freedom (Welch-Satterthwaite equation)
        df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        df = max(1, int(df))
        
        # Approximate p-value using t-distribution
        # Simplified: use critical values
        critical_value_95 = 1.96 if df >= 30 else 2.0
        
        is_significant = abs(t_statistic) > critical_value_95
        
        # Approximate p-value (simplified)
        if abs(t_statistic) > 3.0:
            p_value = 0.001
        elif abs(t_statistic) > 2.5:
            p_value = 0.01
        elif abs(t_statistic) > 2.0:
            p_value = 0.05
        elif abs(t_statistic) > 1.96:
            p_value = 0.05
        else:
            p_value = 0.1
        
        interpretation = f"t={t_statistic:.3f}, df={df}, p≈{p_value:.3f}"
        
        return is_significant, p_value, interpretation
    
    def analyze_trend(self, values: List[float], timestamps: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze trend in a time series.
        
        Uses linear regression and correlation analysis.
        
        Returns:
            Dictionary with trend statistics
        """
        if len(values) < 2:
            return {
                "trend": "insufficient_data",
                "slope": 0.0,
                "correlation": 0.0,
                "p_value": 1.0,
                "is_significant": False
            }
        
        n = len(values)
        
        # Use indices as time if timestamps not provided
        if timestamps is None:
            x = list(range(n))
        else:
            x = timestamps
        
        # Linear regression: y = ax + b
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(values)
        
        # Calculate slope and intercept
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        intercept = mean_y - slope * mean_x
        
        # Calculate correlation coefficient
        std_x = statistics.stdev(x) if len(x) > 1 else 1.0
        std_y = statistics.stdev(values) if n > 1 else 1.0
        
        if std_x == 0 or std_y == 0:
            correlation = 0.0
        else:
            correlation = (numerator / n) / (std_x * std_y)
        
        # Test significance of correlation
        if n > 2:
            t_stat = correlation * math.sqrt((n - 2) / (1 - correlation**2)) if abs(correlation) < 0.999 else 10.0
            # Approximate p-value
            if abs(t_stat) > 3.0:
                p_value = 0.001
            elif abs(t_stat) > 2.0:
                p_value = 0.05
            else:
                p_value = 0.1
        else:
            p_value = 1.0
        
        is_significant = p_value < 0.05
        
        # Determine trend direction
        if abs(slope) < 1e-6:
            trend = "stable"
        elif slope > 0:
            trend = "increasing" if is_significant else "slightly_increasing"
        else:
            trend = "decreasing" if is_significant else "slightly_decreasing"
        
        return {
            "trend": trend,
            "slope": slope,
            "intercept": intercept,
            "correlation": correlation,
