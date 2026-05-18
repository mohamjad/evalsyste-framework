"""Scoring primitives for verifiable and non-verifiable eval work."""

from __future__ import annotations

from evalsyste.models import Criterion, EvidenceItem
from evalsyste.text import contradiction_flags, coverage_score, lexical_entropy


def weighted_mean(scores: dict[str, float], criteria: tuple[Criterion, ...]) -> float:
    total_weight = sum(max(criterion.weight, 0.0) for criterion in criteria)
    if total_weight == 0:
