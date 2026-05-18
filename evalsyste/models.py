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
        if self.max_score <= self.min_score:
            raise ValueError("max_score must be greater than min_score")
        clipped = min(max(score, self.min_score), self.max_score)
        return (clipped - self.min_score) / (self.max_score - self.min_score)


@dataclass(frozen=True)
class EvidenceItem:
    """Evidence that can support or attack a model answer."""

    source: str
    claim: str
    supports: bool = True
    weight: float = 1.0


@dataclass(frozen=True)
class EvalCase:
    """Single eval task with optional references and rubric criteria."""

    id: str
    prompt: str
