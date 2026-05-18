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
