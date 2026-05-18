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
    )
    declared = [turn.declared_intent for turn in trace.turns if turn.declared_intent]
    declared_stability = (
        _mean([jaccard_similarity(anchor, intent) for intent in declared]) if declared else 0.0
    )
    anchor_tokens = set(tokenize(anchor))
    focus_values = []
    for turn in trace.turns:
        response_tokens = set(tokenize(turn.response))
        focus_values.append(len(anchor_tokens & response_tokens) / max(len(anchor_tokens), 1))
    response_focus = _mean(focus_values)
