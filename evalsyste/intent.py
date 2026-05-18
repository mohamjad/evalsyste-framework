"""Intent drift scoring for multi-turn model sessions."""

from __future__ import annotations

from dataclasses import dataclass

from evalsyste.models import SessionTrace
from evalsyste.text import jaccard_similarity, tokenize


@dataclass(frozen=True)
