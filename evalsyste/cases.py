"""Built-in eval cases that exercise ambiguity, evidence, and drift."""

from __future__ import annotations

from evalsyste.models import Criterion, EvalCase, EvidenceItem


NONVERIFIABLE_CRITERIA = (
    Criterion("reference_coverage", 0.25, "Covers the known reference facts."),
    Criterion("evidence_support", 0.30, "Keeps evidence and uncertainty separate."),
    Criterion("specificity", 0.15, "Gives an auditable answer."),
