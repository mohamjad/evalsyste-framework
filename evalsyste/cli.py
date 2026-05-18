"""Command line interface for evalsyste."""

from __future__ import annotations

import argparse
import json

from evalsyste.cases import load_builtin_cases
from evalsyste.runner import EvalRunner


def baseline_agent(prompt: str, metadata: dict) -> str:
    _ = metadata
    return (
        "Observed facts: service logs show timeouts; the payment queue recovered; "
        "tests passed when tests are mentioned; a rollback plan can still be missing. "
        "Unresolved claims stay unverified. Do not claim refunds were issued or that "
        "a deployment is fully safe without direct evidence. "
        f"Task context: {prompt}"
    )


def run_builtin(args: argparse.Namespace) -> int:
    report = EvalRunner(threshold=args.threshold).run_suite(baseline_agent, load_builtin_cases())
    payload = {
        "score": report.score,
        "pass_rate": report.pass_rate,
        "results": [
            {
                "case_id": result.case_id,
                "score": result.score,
                "passed": result.passed,
                "metrics": result.metrics,
