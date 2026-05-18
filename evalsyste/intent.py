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

