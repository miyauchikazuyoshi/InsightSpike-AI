---
layout: home
title: geDIG — Unified Gauge Control
---

# geDIG — Unified Gauge Control for Dynamic Knowledge Graphs

> **式の位置づけ（簡約式） / Formula Status (Simplified)**: この文書の数式は説明用の簡約式です。正準定義（Canonical）は `docs/gedig_spec.md` です。


> A discrete free‑energy perspective bridging FEP, MDL, and information thermodynamics to govern When/What in dynamic knowledge graphs.

- Landing (JA, overview slide): [geDIG unified gauge (JA overview)](images/gedig_unified_gauge_ja_landing.pdf)
- Paper v6.1 (JA): [geDIG_onegauge_improved_v6_1.pdf](paper/v6.1/geDIG_onegauge_improved_v6_1.pdf)
- Paper v6.1 (EN): [geDIG_onegauge_improved_v6_1_en.pdf](paper/v6.1/arxiv_en/geDIG_onegauge_improved_v6_1_en.pdf)
- Quick start: `python examples/public_quick_start.py`
- CLI: `python -m insightspike.cli.spike --help`
- Smoke tests: `make codex-smoke`

## What is geDIG?

- Single gauge F: `F = ΔEPC_norm − λ (ΔH_norm + γ ΔSP_rel)`
- Two‑stage gate: AG (0‑hop) → DG (multi‑hop)
- Phase‑1 PoC: Maze + RAG under equal‑resources, with observable decision logs

See Theory and Phase‑1 pages for sign conventions, PSZ/SLO, and reproduction.

## Reproduce (Maze 25×25, 500 steps)

- L3 batch (60 seeds): `python scripts/run_maze_batch_and_update.py --mode l3 --seeds 60 --workers 4 --update-tex`
- Eval batch (60 seeds): `python scripts/run_maze_batch_and_update.py --mode eval --seeds 60 --workers 4 --update-tex`
- Aggregates land in `docs/paper/data/`; the 25×25 table updates automatically.

## Call for Reviewers / Collaborators

We’re looking for collaborators on:
- Information thermodynamics, Active Inference (FEP), MDL
- Graph RAG, multi‑hop reasoning, dynamic knowledge bases
- Phase‑2: offline rewiring (global consistency)

How to engage:
- Open an Issue with “Review” label, or PR small fixes
- Or DM on X (Twitter): @kazuyoshim5436

## Links
- Spec: [geDIG spec](gedig_spec.md)
- Phase‑1: [maze & RAG](phase1.md)
- Tutorial (trace): [trace a spike](tutorials/trace.md)
- GitHub repo: [InsightSpike-AI](https://github.com/miyauchikazuyoshi/InsightSpike-AI)
