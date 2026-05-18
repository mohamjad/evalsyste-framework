# evalsyste-framework

Evaluation code for messy model work.

The thesis is simple:

```text
most useful evals are not just "match the answer key"
```

This repo scores the parts that usually get hand-waved:

- evidence use
- contradiction risk
- non-verifiable work
- intent drift across turns
- clarity under growing context
- stability signals over repeated operations

It keeps the implementation small and inspectable. No giant platform. No fake
research claims. Just structured eval objects, scorers, runners, fixtures, and
tests.

## Run

```bash
pip install -e .[dev]
pytest
evalsyste builtin
```

## Core Package

```python
from evalsyste import EvalRunner
from evalsyste.cases import load_builtin_cases


def agent(prompt, metadata):
    return prompt + " unresolved claims stay unverified"


report = EvalRunner().run_suite(agent, load_builtin_cases())
print(report.score)
print(report.pass_rate)
```

## What Is In Here

```text
evalsyste/
  models.py          typed eval cases, evidence, traces, reports
  scoring.py         rubric, evidence, uncertainty scores
  nonverifiable.py   scorer for work without one gold answer
  intent.py          session intent drift
  runner.py          small suite runner
  cases.py           built-in ambiguity cases

ai_system_stability_framework.py
  older stability monitor for contradiction, clarity, redundancy, context scale
```

## Useful Commands

```bash
evalsyste builtin
pytest
python examples/basic_usage.py
```

## Limits

Heuristic scores are not truth. They are pressure tests.

Use this to expose weak behavior, compare runs, and force eval assumptions into
code.

