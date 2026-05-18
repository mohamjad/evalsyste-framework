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
    total = 0.0
    answer_lower = answer.lower()
    for item in evidence:
        weight = max(item.weight, 0.0)
        total += weight
        claim_present = item.claim.lower() in answer_lower
        if item.supports and claim_present:
            support += weight
        if not item.supports and not claim_present:
            support += weight
    return support / total if total else 0.0


def rubric_proxy_scores(
    answer: str,
    references: tuple[str, ...],
    evidence: tuple[EvidenceItem, ...],
) -> dict[str, float]:
    flags = contradiction_flags(answer)
    return {
        "reference_coverage": coverage_score(answer, references),
        "evidence_support": evidence_support_score(answer, evidence),
