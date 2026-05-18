"""Tests for suite runner behavior."""

from evalsyste.cases import load_builtin_cases
from evalsyste.runner import EvalRunner


def test_runner_returns_report_for_builtin_cases():
    def agent(prompt, metadata):
        return prompt + " Unknown claims remain unverified and evidence must be separated."

    report = EvalRunner(threshold=0.3).run_suite(agent, load_builtin_cases())
