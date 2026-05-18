# Scenario Matrix

## Evidence laundering

Partial evidence becomes a clean answer.

Scored by:

- evidence support
- trap resistance through unsupported claims
- contradiction flags

## Intent drift

The system keeps talking but leaves the task.

Scored by:

- target intent overlap
- declared intent stability
- response focus

## Context decay

More context reduces coherence instead of improving it.

Scored by:

- contradiction count
- coherence trend
- redundancy and clarity signals

## Non-verifiable synthesis

There is no single answer key, but the work still has structure.

Scored by:

- reference coverage
- evidence support
- specificity
- information density

## Tool trace collapse

The final answer hides a bad process.

Tracked by downstream runners through trace metrics; this repo keeps the
scoring objects small enough to plug into those runners.
