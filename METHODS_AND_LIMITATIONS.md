# Methods And Limitations

This repository is intentionally scoped as a lightweight monitoring and experimentation toolkit.

## Core Methods

- Coherence is computed from pairwise contradiction checks across collected statements.
- The default contradiction detector is mostly lexical and pattern-based.
- Clarity uses simple text features and information-content heuristics.
- Redundancy is estimated from overlap and optional semantic similarity.
- Trend analysis summarizes whether scores improve or degrade as context size changes.

## Optional Methods

- Sentence-transformer embeddings can be used when available.
- Neural models can be trained separately and loaded as optional add-ons.
- Gaussian-process search utilities are available for parameter sweeps.

## What This Repo Does Not Claim

- It is not a benchmark standard like `openai/evals`.
- It is not a production-grade LLM safety system.
- It does not provide calibrated scientific measures of coherence or clarity.
- It does not establish new evaluation research on its own.

## Recommended Use

- debugging prompt or system behavior
- comparing heuristic stability trends across runs
- instrumenting experiments where traceability matters more than leaderboard rigor

## Not Recommended As

- a sole decision-maker for deployment readiness
- a substitute for human review
- evidence of research-grade semantic evaluation by itself
