"""Typed models for task, evidence, and session-level evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


AgentFn = Callable[[str, dict[str, Any]], str]


