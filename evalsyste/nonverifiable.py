"""Evaluation path for work that cannot be reduced to one gold answer."""

from __future__ import annotations

from evalsyste.models import Criterion, EvalCase, EvalResult
from evalsyste.scoring import rubric_proxy_scores, uncertainty_penalty, weighted_mean


DEFAULT_NONVERIFIABLE_CRITERIA = (
    Criterion("reference_coverage", 0.25, "Answer covers known reference points."),
    Criterion("evidence_support", 0.25, "Answer uses supporting evidence without laundering conflicts."),
    Criterion("specificity", 0.15, "Answer is concrete enough to audit."),
    Criterion("non_contradiction", 0.2, "Answer does not contradict itself."),
    Criterion("information_density", 0.15, "Answer carries signal instead of filler."),
)


def score_nonverifiable_answer(case: EvalCase, answer: str, threshold: float = 0.65) -> EvalResult:
    criteria = case.criteria or DEFAULT_NONVERIFIABLE_CRITERIA
    metrics = rubric_proxy_scores(answer, case.references, case.evidence)
    penalty = uncertainty_penalty(answer)
    metrics["uncertainty_penalty"] = penalty
    raw_score = weighted_mean(metrics, criteria)
    score = max(0.0, raw_score - 0.15 * penalty)
    notes = []
    if metrics["evidence_support"] < 0.5 and case.evidence:
        notes.append("weak evidence support")
    if metrics["non_contradiction"] < 1.0:
        notes.append("internal contradiction risk")
    return EvalResult(
        case_id=case.id,
        score=score,
        passed=score >= threshold,
        metrics=metrics,
        notes=tuple(notes),
    )
