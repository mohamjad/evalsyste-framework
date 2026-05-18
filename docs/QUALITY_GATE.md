# Quality Gate

Run:

```bash
pytest
python -m evalsyste.cli builtin
python scripts/quality_gate.py
```

The gate checks that the repo is not just generic eval plumbing:

- typed eval models
- non-verifiable scorer
- intent drift scorer
- suite runner
- built-in hard cases
- legacy stability monitor preserved
- package tests
- thesis visible in README
- scenario matrix present
