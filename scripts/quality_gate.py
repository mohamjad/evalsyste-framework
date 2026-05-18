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
    checks = {
        "typed_eval_models": exists("evalsyste/models.py"),
        "nonverifiable_scorer": exists("evalsyste/nonverifiable.py"),
        "intent_drift_scorer": exists("evalsyste/intent.py"),
        "suite_runner": exists("evalsyste/runner.py"),
        "builtin_cases": exists("evalsyste/cases.py"),
        "legacy_monitor_kept": exists("ai_system_stability_framework.py"),
        "tests_cover_package": len(list((ROOT / "tests").glob("test_evalsyste_*.py"))) >= 3,
        "thesis_visible": "non-verifiable" in readme and "intent drift" in readme,
        "scenario_matrix": "evidence laundering" in docs and "context decay" in docs,
        "no_placeholder_language": "lorem ipsum" not in docs and "placeholder implementation" not in docs,
    }
    payload = {"repo": "evalsyste-framework", "passed": all(checks.values()), "checks": checks}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
