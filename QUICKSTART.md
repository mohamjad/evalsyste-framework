# Quick Start

This repo can be used immediately as a small monitoring harness for statement-level consistency checks.

## 1. Install

```bash
pip install -e .
```

## 2. Run the built-in smoke test

```bash
python test_framework.py
```

## 3. Use the framework

```python
from ai_system_stability_framework import AIStabilityFramework

framework = AIStabilityFramework(log_file="stability.log", log_level="WARNING")

result = framework.process_operation(
    operation_description="answer user question",
    statements=[
        "The system is stable",
        "The system is processing the request",
    ],
    context_size=100,
)

print(result["metrics"]["coherence"])
```

## 4. Interpret the output carefully

- `coherence` is a normalized contradiction count, not a semantic truth score.
- `clarity` and `redundancy` are heuristic indicators.
- `signal_quality` is a coarse label meant for debugging, not for governance decisions.

For scope and limitations, read [`METHODS_AND_LIMITATIONS.md`](./METHODS_AND_LIMITATIONS.md).
