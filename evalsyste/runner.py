"""Small runner for repeatable eval suites."""

from __future__ import annotations

from typing import Any

from evalsyste.models import AgentFn, EvalCase, EvalReport
from evalsyste.nonverifiable import score_nonverifiable_answer


class EvalRunner:
    """Run an agent function across eval cases."""

    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = threshold

    def run_case(self, agent: AgentFn, case: EvalCase) -> dict[str, Any]:
        answer = agent(case.prompt, case.metadata)
        result = score_nonverifiable_answer(case, answer, threshold=self.threshold)
        return {"answer": answer, "result": result}

    def run_suite(self, agent: AgentFn, cases: tuple[EvalCase, ...]) -> EvalReport:
        results = tuple(self.run_case(agent, case)["result"] for case in cases)
        return EvalReport(results=results)
