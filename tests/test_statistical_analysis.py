"""Tests for statistical helper behavior."""

from statistical_analysis import StatisticalAnalyzer


def test_confidence_interval_singleton_is_exact():
    """A single sample should return a zero-width interval."""
    analyzer = StatisticalAnalyzer()

    mean, lower, upper = analyzer.calculate_confidence_interval([0.8])

    assert mean == 0.8
    assert lower == 0.8
    assert upper == 0.8


def test_significance_requires_enough_data():
    """Welch testing should refuse undersized samples."""
    analyzer = StatisticalAnalyzer()

    significant, p_value, interpretation = analyzer.test_significance([0.8], [0.6])

