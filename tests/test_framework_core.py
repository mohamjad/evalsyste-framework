"""Core framework behavior tests."""

from ai_system_stability_framework import AIStabilityFramework


def test_framework_self_verifies_and_initializes(tmp_path):
    """Framework should initialize only after its self-checks pass."""
    log_path = tmp_path / "stability.log"
    framework = AIStabilityFramework(log_file=str(log_path), log_level="ERROR")

    assert framework.verified is True
