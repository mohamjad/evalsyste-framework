"""Tests for statistical helper behavior."""

from statistical_analysis import StatisticalAnalyzer


def test_confidence_interval_singleton_is_exact():
    """A single sample should return a zero-width interval."""
    analyzer = StatisticalAnalyzer()

    mean, lower, upper = analyzer.calculate_confidence_interval([0.8])

