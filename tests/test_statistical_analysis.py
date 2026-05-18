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

    assert significant is False
    assert p_value == 1.0
    assert "insufficient data" in interpretation


def test_trend_analysis_identifies_increase():
    """A clear upward series should be recognized as increasing."""
    analyzer = StatisticalAnalyzer()

    result = analyzer.analyze_trend([0.2, 0.4, 0.6, 0.8, 1.0])

    assert result["trend"] in {"increasing", "slightly_increasing"}
    assert result["slope"] > 0


def test_bootstrap_interval_returns_ordered_bounds():
    """Bootstrap output should return mean and ordered interval bounds."""
    analyzer = StatisticalAnalyzer()

    mean, lower, upper = analyzer.bootstrap_confidence_interval([0.3, 0.4, 0.5, 0.6], n_bootstrap=50)

    assert lower <= mean <= upper
