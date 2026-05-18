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
    intent: str
    criteria: tuple[Criterion, ...]
    references: tuple[str, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelTurn:
    """One model answer in a session."""

    prompt: str
    response: str
    declared_intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionTrace:
    """Multi-turn trace used for intent-drift checks."""

    id: str
    turns: tuple[ModelTurn, ...]
    target_intent: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """Score payload for one eval case."""

    case_id: str
    score: float
    passed: bool
    metrics: dict[str, float]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalReport:
    """Collection-level report."""

    results: tuple[EvalResult, ...]

    @property
    def score(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.score for result in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.passed) / len(self.results)
