"""Small runner for repeatable eval suites."""

from __future__ import annotations

from typing import Any

from evalsyste.models import AgentFn, EvalCase, EvalReport
from evalsyste.nonverifiable import score_nonverifiable_answer


class EvalRunner:
