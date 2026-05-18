"""Scoring primitives for verifiable and non-verifiable eval work."""

from __future__ import annotations

from evalsyste.models import Criterion, EvidenceItem
from evalsyste.text import contradiction_flags, coverage_score, lexical_entropy


def weighted_mean(scores: dict[str, float], criteria: tuple[Criterion, ...]) -> float:
    total_weight = sum(max(criterion.weight, 0.0) for criterion in criteria)
    if total_weight == 0:
        return 0.0
    weighted = 0.0
    for criterion in criteria:
        weighted += criterion.normalize(scores.get(criterion.name, 0.0)) * max(criterion.weight, 0.0)
    return weighted / total_weight


def evidence_support_score(answer: str, evidence: tuple[EvidenceItem, ...]) -> float:
    if not evidence:
        return 0.0
    support = 0.0
