"""Tests for the package-level eval scorers."""

from evalsyste.cases import load_builtin_cases
from evalsyste.nonverifiable import score_nonverifiable_answer
from evalsyste.scoring import evidence_support_score


def test_evidence_support_rewards_supported_claims():
    case = load_builtin_cases()[0]

    score = evidence_support_score("timeouts occurred and the queue recovered", case.evidence)
