"""Tests for intent drift scoring."""

from evalsyste.intent import score_intent_drift
from evalsyste.models import ModelTurn, SessionTrace


def test_intent_drift_detects_focus_loss():
    trace = SessionTrace(
        id="drift",
        target_intent="separate observed facts from unresolved claims",
        turns=(
