"""Core framework behavior tests."""

from ai_system_stability_framework import AIStabilityFramework


def test_framework_self_verifies_and_initializes(tmp_path):
    """Framework should initialize only after its self-checks pass."""
    log_path = tmp_path / "stability.log"
    framework = AIStabilityFramework(log_file=str(log_path), log_level="ERROR")

    assert framework.verified is True
    assert framework.operation_count == 0
    assert framework.thresholds["coherence_minimum"] == 0.75


def test_process_operation_detects_contradictions(tmp_path):
    """Obvious contradictions should reduce coherence and register in metrics."""
    framework = AIStabilityFramework(log_file=str(tmp_path / "contradiction.log"), log_level="ERROR")

    result = framework.process_operation(
        operation_description="contradiction check",
        statements=[
            "The system is stable",
            "The system is unstable",
            "Processing continues",
        ],
        context_size=200,
    )

    assert result["metrics"]["contradictions_count"] > 0
    assert result["metrics"]["coherence"] < 1.0


def test_custom_thresholds_are_applied(tmp_path):
    """Caller-supplied thresholds should override defaults."""
    framework = AIStabilityFramework(
        log_file=str(tmp_path / "custom.log"),
        log_level="ERROR",
        custom_thresholds={"coherence_minimum": 0.9},
    )

    assert framework.thresholds["coherence_minimum"] == 0.9


def test_report_includes_recent_operations(tmp_path):
    """Generated reports should summarize recent monitored work."""
    framework = AIStabilityFramework(log_file=str(tmp_path / "report.log"), log_level="ERROR")
    framework.process_operation(
        operation_description="reportable operation",
        statements=["System initialized", "Generated response"],
        context_size=100,
    )

    report = framework.generate_stability_report()

