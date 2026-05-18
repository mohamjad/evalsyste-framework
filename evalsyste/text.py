"""Small text utilities used by deterministic eval scorers."""

from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in TOKEN_RE.finditer(text))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def coverage_score(answer: str, references: tuple[str, ...]) -> float:
    if not references:
        return 0.0
    return max(jaccard_similarity(answer, reference) for reference in references)


def lexical_entropy(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy


def contradiction_flags(text: str) -> tuple[str, ...]:
    tokens = set(tokenize(text))
    flags: list[str] = []
    pairs = (
        ("always", "never"),
        ("safe", "unsafe"),
        ("stable", "unstable"),
        ("complete", "incomplete"),
        ("verified", "unverified"),
        ("certain", "uncertain"),
    )
    for left, right in pairs:
        if left in tokens and right in tokens:
            flags.append(f"{left}/{right}")
    return tuple(flags)
