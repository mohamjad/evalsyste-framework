"""Intent drift scoring for multi-turn model sessions."""

from __future__ import annotations

from dataclasses import dataclass

from evalsyste.models import SessionTrace
from evalsyste.text import jaccard_similarity, tokenize


@dataclass(frozen=True)
class IntentDriftReport:
    session_id: str
    anchor_similarity: float
    declared_intent_stability: float
    response_focus: float
    drift_score: float

    @property
    def stable(self) -> bool:
        return self.drift_score <= 0.35


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_intent_drift(trace: SessionTrace) -> IntentDriftReport:
    if not trace.turns:
        return IntentDriftReport(trace.id, 0.0, 0.0, 0.0, 1.0)
    anchor = trace.target_intent
    anchor_similarity = _mean(
        [jaccard_similarity(anchor, turn.prompt + " " + turn.response) for turn in trace.turns]
