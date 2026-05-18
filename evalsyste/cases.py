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
                "timeouts, the payment queue recovered, and no customer refund log is present."
            ),
            intent="separate observed facts from unresolved claims",
            criteria=NONVERIFIABLE_CRITERIA,
            references=(
                "service logs show timeouts",
                "payment queue recovered",
                "refund status is unverified",
            ),
            evidence=(
                EvidenceItem("service-log", "timeouts"),
                EvidenceItem("queue-monitor", "recovered"),
                EvidenceItem("refund-ledger", "refund issued", supports=False),
            ),
        ),
        EvalCase(
            id="policy-answer-with-conflict",
            prompt=(
                "Answer whether a deployment is safe when tests passed but the rollback "
                "plan is missing."
            ),
            intent="avoid calling a partially verified deployment fully safe",
