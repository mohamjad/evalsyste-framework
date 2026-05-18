"""Built-in eval cases that exercise ambiguity, evidence, and drift."""

from __future__ import annotations

from evalsyste.models import Criterion, EvalCase, EvidenceItem


NONVERIFIABLE_CRITERIA = (
    Criterion("reference_coverage", 0.25, "Covers the known reference facts."),
    Criterion("evidence_support", 0.30, "Keeps evidence and uncertainty separate."),
    Criterion("specificity", 0.15, "Gives an auditable answer."),
    Criterion("non_contradiction", 0.20, "Avoids internal conflict."),
    Criterion("information_density", 0.10, "Avoids filler."),
)


def load_builtin_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            id="ambiguous-incident-summary",
            prompt=(
                "Summarize the incident. Evidence is partial: one service log shows "
