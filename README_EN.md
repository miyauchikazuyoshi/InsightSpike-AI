# InsightSpike-AI — geDIG

**A structural fitness score for knowledge graphs — can one scalar decide *when* to restructure?**

[![CI (Lite)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml)
[![Paper](https://img.shields.io/badge/paper-PDF-blue)](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-green)](https://miyauchikazuyoshi.github.io/InsightSpike-AI)

> **Status:** active research by an individual researcher, with AI-assisted implementation.
> This is a **position paper + proof-of-concept**, *not* a production library and *not* a state-of-the-art system.
> Most quantitative results below are **single-seed** and should be read as preliminary; statistical validation (multiple seeds, significance tests) is planned for v7.

> *(This is the English README. It mirrors the primary [`README.md`](README.md). An earlier version of this file described an isomorphism / "scientific-discovery" prototype; those demos used synthetic data and have been archived — see [What this is **not**](#what-this-is-not).)*

---

## The Equation

```
F = ΔEPC_norm − λ ( ΔH_norm + γ · ΔSP_rel )
```

| Term | Mathematical structure | What it captures |
|------|------------------------|------------------|
| ΔEPC | Metric (distance) | Cost of restructuring the knowledge graph (graph-edit-path cost) |
| ΔH | Measure (probability) | Change in entropy / uncertainty (Shannon) |
| ΔSP_rel | Structure potential (paths) | Relative shortest-path gain |

**F < 0** means information gain exceeds structural cost — the system should commit the change.

The established gauge and v6 paper use ΔSP. Ongoing v7 research evaluates
Δβ₁ (the first Betti number) as an explicit alternative research mode; it is
not the default structural term.

Two-stage gating: **AG** (0-hop ambiguity detection → explore) and **DG** (multi-hop confirmation → integrate). The neurotransmitter analogy (AG↔noradrenaline, DG↔dopamine) and the thermodynamic reading (F = E − TS, FEP/MDL) are **operational metaphors**, not literal claims.

---

## Project Status (honest)

| Component | Status |
|-----------|--------|
| **Unified geDIG Core** | 80 core regression tests (all 80 pass locally; a CI-compatible torch import-block simulation produces 57 passes and 23 expected skips). Transformer values/gradients are compared with the frozen independent oracle; RAG has independent formula checks; Maze has adapter contracts plus an active-legacy golden trace, not a full equivalence claim. See [`src/gedig/`](src/gedig/). |
| geDIG theory (v6 paper) | Pre-print — position paper + PoC. [PDF](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf) |
| Maze PoC (single episode) | 98% goal-reach (15×15, n=100) — **on par with greedy-DFS / oracle (99%)**; the distinctive result is ~98% map compression. PoC only. |
| Maze stage-2 (Wake-Sleep-Wake) | Pre-registered v2/v3 established and replicated a trajectory-Q replay benefit for self-navigation; v5/v6 rejected unconditional budget splitting. Single-cycle remains the default. See [`experiments/maze/graph_persistent_dg/`](experiments/maze/graph_persistent_dg/). |
| HotpotQA dual-process (v3) | GPT-4o EM 51.2% vs IRCoT 47.6% (500q) — **p=0.086, not significant**; the ordering reverses on GPT-4o-mini (p=0.008). The most carefully evaluated experiment. |
| BRIGHT retrieval (AGHT) | nDCG@10 = 0.439 on **biology, 50 queries, single seed** (full 3-domain ≈ 0.19). **Not competitive with SOTA (≈ 0.63).** |
| Transformer Flash-profile regularization (Exp4) | **Single-seed, preliminary**: single-state profile maximize > baseline under the SP profile only — *not* confirmed under β₁ or vs a random-regularization control. |

---

## Repository Maintenance Status (2026-07-25)

The P0–P6 technical-debt programme is complete. The public API remains
compatible, while formerly implicit behavior is now covered by explicit
contracts.

| Area | Current contract |
|------|------------------|
| Public runtime | `create_agent()` returns an initialized agent and injects the configured `DataStore`; direct `MainAgent(...)` construction remains lazy. |
| Configuration | Strict nested validation, source-aware legacy migration, Pydantic v1/v2 compatibility, canonical examples, and safe YAML round trips. |
| Persistence | Memory, filesystem, and SQLite share one factory and episode codec; state saves are exact snapshots and indexes rebuild after load. |
| `MainAgent` | The facade and replaceable L1–L4 components remain; lifecycle, persistence, result aggregation, and config access are focused services. |
| geDIG and adapters | Canonical delta F is separate from the legacy Flash profile. Transformer validation compares against an independent frozen oracle, the active AGHT RAG path delegates to `RAGFEval`, and active Maze remains legacy-fixed with a golden trace. |
| Message passing | `weighted_mean`, `mean`, and `max` are explicit; legacy `attention` warns and aliases `mean`; unknown values raise. |

Validation evidence:

- 143 targeted tests passed across public API, config, persistence, Flash,
  message passing, and `MainAgent`.
- All 80 unified-core tests passed locally; the torch-less simulation produced
  57 passes and 23 expected skips.
- The `MainAgent` integration file moved from the unchanged-HEAD baseline of
  11 failures / 11 passes to 22 passes in both PyG and torch-less environments.
- The cloud-safe smoke suite passed 16/16. The standard entry collected 1,078
  tests with 33 collection-time skips; this is not a claim that all 1,078 were run.

References:

- [Completed debt-repayment plan](docs/development/debt_repayment_refactoring_plan_20260724.md)
- [Canonical implementation map](docs/CANONICAL.md)
- [Adapter migration ledger](src/gedig/docs/MIGRATION_PROGRESS.md)
- [`MainAgent` behavior contract](docs/architecture/mainagent_behavior.md)

Deliberate boundaries: active Maze cutover waits for the ongoing v7 work;
Transformer T4/Exp4 and historical RAG R4–R6 E2E runs were not rerun; a true
attention aggregator requires a separate scientific specification.

---

## Experiments (with caveats)

### Maze Navigation — proof-of-concept
A partial-observation agent builds a persistent graph and uses F + AG/DG to decide when to explore vs. integrate.
- **15×15 (n=100):** geDIG 98% (Wilson 95% CI [0.93, 0.99]). A plain greedy-DFS reaches **99%** and oracle BFS **99%** — geDIG does **not** beat the simplest complete baseline on success rate.
- **What is distinctive:** emergent AG/DG control (no hard-coded "if dead-end" rule) + ~98% map compression (keeps only the topological skeleton).
- **25×25:** preliminary (success ≈ 0.42–0.75 across configs); statistics not yet finalized.

### HotpotQA dual-process (v3) — strongest result
F (via Betti numbers) routes between System 1 (answer now) and System 2 (reason step-by-step).
- **GPT-4o (500q):** Hybrid-E1 EM 51.2% / F1 0.667 vs IRCoT EM 47.6% / F1 0.653, at fewer LLM calls — but **McNemar p = 0.086 (not significant)**.
- **GPT-4o-mini (500q):** IRCoT wins (50.4% vs 45.2%, p = 0.008).
- Call counts (~2.2 vs ≤8) are approximate: Hybrid's value is 1 + mean(CoT steps); IRCoT's "8" is a `max_steps` ceiling, not a measured count.
- Negative results from v4/v5 (adaptive depth, topology routing) are reported in [`REPORT_v3_dual_process.md`](experiments/hotpotqa_v2/REPORT_v3_dual_process.md).

### BRIGHT / AGHT — early zero-shot, not competitive
Zero-shot heterogeneous graph transformer (10 analytical parameters).
- nDCG@10 = 0.439 on **biology, 50q, single seed** (CI ≈ ±13pt). Full 3-domain (323q) ≈ **0.19**.
- For context on BRIGHT: BM25 ≈ 0.145, BM25 + GPT-4 + rerank ≈ 0.30, SOTA INF-X-Retriever ≈ 0.63. This is an early PoC, well below these.
- HotpotQA paragraph selection (100q, zero-shot): R@2 = 0.405, **+170% over an internal PageRank baseline** (not a supervised SOTA).

### Transformer single-state Flash-profile regularization (Exp4) — preliminary
SST-2 / DistilBERT, single run, β = 0.1.
- SP profile: profile-maximize 89.4% vs baseline 88.1% (baseline peaked at 89.4% at epochs 1–2, then overfit to 88.1%).
- **β₁ profile: the same condition reached 85.5% < baseline 88.5%** — the effect does not hold.
- A negative control found profile regularization (66.5%) did **not** beat random-value regularization (69.5%). This does not establish the direction of canonical before/after delta F; multi-seed replication is required before any `negative_better` claim.

---

## Theoretical Background

geDIG hypothesizes that intelligence acts to minimize F: balancing the cost of restructuring knowledge against the information gained. The correspondence to Free Energy (FEP) and Minimum Description Length (MDL) is presented as an **operational** proposition (proportionality under stated assumptions, with O(1/N) residuals), not as mathematical equivalence or a physiological claim. See [`docs/gedig_spec.md`](docs/gedig_spec.md) and the [v6 pre-print](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf).

---

## What this is **not**

To set expectations honestly:
- **Not SOTA.** On BRIGHT/HotpotQA the system is below standard strong baselines; comparisons are mostly against internal baselines (e.g. PageRank).
- **Mostly single-seed PoCs.** Reported numbers are not yet backed by multi-seed significance testing (planned for v7).
- **Archived earlier claims.** Earlier "scientific-discovery" demos — reproducing Bohr/Kekulé/Darwin structural similarities, automatic cross-domain analogy discovery, "+60%" cross-domain QA — were exploratory results on **synthetic data**, and their demo scripts have been archived (not part of the current validated codebase). They are intentionally **not** repeated here.

---

## Quick Start

```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
make install-dev
make test
```

For a runtime-only installation, use `pip install -e .`.

### Ready-to-use Agent API

```python
from insightspike import create_agent

agent = create_agent(provider="mock", datastore__type="memory")
assert agent.initialized

agent.add_knowledge("geDIG balances graph-edit cost and information gain.")
result = agent.process_question("What does geDIG balance?", max_cycles=1)
print(result.response)
```

### Canonical delta F

The example below requires the optional PyTorch dependency for your platform.

```python
import torch
from insightspike.gedig import compute_delta_f_score

before = torch.softmax(torch.rand(1, 12, 64, 64), dim=-1)
after = torch.softmax(torch.rand(1, 12, 64, 64), dim=-1)
result = compute_delta_f_score(before, after)
print(f"Mean delta F: {result.F_mean:.4f}")  # lower is better
```

---

## Citation, License, and Patent

- **Paper:** [geDIG v6 (pre-print)](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf)
- **DOI:** [10.5281/zenodo.19454110](https://zenodo.org/record/19454110)
- **License:** Apache-2.0
- **Patents:** JP 2025-082988, JP 2025-082989 (pending)
- **Contact:** miyauchikazuyoshi@gmail.com

> *All theoretical contributions and experimental design are by the author. Implementation is AI-assisted (Codex, Claude, GitHub Copilot).*
