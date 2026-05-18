"""Repository quality gate for evalsyste-framework."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    readme = read("README.md").lower()
    docs = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "docs").glob("*.md")).lower()
