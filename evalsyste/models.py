"""Typed models for task, evidence, and session-level evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


AgentFn = Callable[[str, dict[str, Any]], str]


@dataclass(frozen=True)
class Criterion:
    """One explicit thing an answer is judged on."""

    name: str
    weight: float
    description: str
    min_score: float = 0.0
    max_score: float = 1.0

    def normalize(self, score: float) -> float:
