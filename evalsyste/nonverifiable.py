"""Evaluation path for work that cannot be reduced to one gold answer."""

from __future__ import annotations

from evalsyste.models import Criterion, EvalCase, EvalResult
from evalsyste.scoring import rubric_proxy_scores, uncertainty_penalty, weighted_mean


DEFAULT_NONVERIFIABLE_CRITERIA = (
    Criterion("reference_coverage", 0.25, "Answer covers known reference points."),
    Criterion("evidence_support", 0.25, "Answer uses supporting evidence without laundering conflicts."),
