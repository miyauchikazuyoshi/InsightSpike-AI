---
layout: home
title: geDIG — Unified Gauge Control
---

# geDIG — Unified Gauge Control for Dynamic Knowledge Graphs

> A discrete free-energy perspective bridging FEP, MDL, and information thermodynamics to govern When/What in dynamic knowledge graphs.

- Paper (PDF): [geDIG_onegauge_improved_v4.pdf](paper/geDIG_onegauge_improved_v4.pdf)
- Quick start: `python examples/public_quick_start.py`
- CLI: `python -m insightspike.cli.spike --help`
- Smoke tests: `make codex-smoke`

## What is geDIG?

- Single gauge F: `F = ΔEPC_norm − λ (ΔH_norm + γ ΔSP_rel)`
- Two-stage gate: AG (0-hop ambiguity) → DG (multi-hop confirmation)
- Phase 1 PoC: Maze + RAG under equal-resources, with observable decision logs

See the paper for the short-form and sign conventions recap and PSZ/SLO definition.

## Reproduce (Maze 25×25, 500 steps)

- L3 batch (60 seeds): scripts/run_maze_batch_and_update.py `--mode l3 --seeds 60 --workers 4 --update-tex`
- Eval batch (60 seeds): scripts/run_maze_batch_and_update.py `--mode eval --seeds 60 --workers 4 --update-tex`
- Aggregates land in `docs/paper/data/` and update the 25×25 table.

## Call for Reviewers / Collaborators

We’re looking for collaborators on:
- Information thermodynamics, Active Inference (FEP), MDL bridges
- Graph RAG, multi-hop reasoning, dynamic knowledge bases
- Phase 2: offline rewiring (global consistency) — design to implementation

How to engage:
- Open an Issue with “Review” label, or PR small fixes
- Or DM on X (Twitter): @your_handle (replace in README)

## Links
- API quick start: docs/api-reference/quick_start.md
- Public API surface: docs/api-reference/public_api.md
- Getting started: docs/getting-started/ENVIRONMENT_SETUP.md

