# CANONICAL — what is authoritative in this repository

> One page answering four questions that the layered history of this project
> otherwise forces every reader to reverse-engineer: **which F, which
> implementation, which tests, which claims.** Created 2026-07-05 in response
> to an external review that (correctly) flagged the ambiguity. If this page
> and any other document disagree, fix the other document.

## 1. The canonical gauge

```
F = ΔEPC − λ(ΔH + γ·ΔB)
```

There is exactly one gauge. **B is a pluggable "structure potential" slot**,
and the two instantiations have different evidential status:

| B = | Status | Where |
|---|---|---|
| **ΔSP_rel** (relative shortest-path gain) | **Established.** Every confirmed result to date (maze PoC, maze stage-2 sleep line, BRIGHT/RAG, Exp4's positive arm) was obtained in this form | Formal spec: [gedig_spec.md](gedig_spec.md) (v4 One-Gauge Spec) |
| **Δβ₁** (Betti-1, topological cycles) | **In-progress generalization** (v7 line). Not yet confirmed anywhere: Exp4 under β₁ is preliminary/negative so far | [paper/v7/plan.md](paper/v7/plan.md) |

So: when a document says "the gauge" without qualification, it means the
**ΔSP form**. β₁ is a research direction, not the incumbent.

**Sign convention** (canonical, as implemented in `src/gedig/core/f_eval.py`):
**F < 0 = structural gain exceeds cost = commit / accept. Lower is better.**
AG/DG means the two-stage **Attention Gate / Decision Gate** event flow:
AG reacts to the hop-0 signal and opens exploration; DG evaluates the
multi-hop signal and confirms a commit. Historical RAG/Transformer code also
called a low/high edge-score partition “AG/DG”; that partition is now a
separate type and is not a gate event.
Derived uses may point the other way locally — e.g. Exp4 rewards *raising* F
as a training regularizer — but those are per-experiment loss constructions,
defined in the experiment's own README; they do not change the judgment
convention here.

## 2. The canonical implementation

| Implementation | Path | Role |
|---|---|---|
| **Unified Core — the reference** | [`src/gedig/`](../src/gedig/) | numpy/networkx. EPC / Entropy / StructurePotential injected as Protocols; one `FEval` for all three domains. **New code imports this.** |
| Flash-geDIG | [`src/insightspike/gedig/`](../src/insightspike/gedig/) | torch-native transformer API. `compute_delta_f_score` delegates to `TransformerFEval`; `compute_structural_profile` is a distinct single-state diagnostic. Public forwarding and gradients are checked by `test_flash_gedig_api.py` |
| Full/legacy geDIG | [`src/insightspike/algorithms/gedig/`](../src/insightspike/algorithms/gedig/) | Historical application implementation. The active maze evaluator still uses `GeDIGCore`; its behavior is frozen by an active-path golden trace, not claimed equivalent to `MazeFEval` |
| Experiment-local code | `experiments/**` | Frozen for reproducibility of published runs. Never canonical |

Both `insightspike` variants predate the unified core; they remain because
published experiments ran on them. Transformer T1–T3/T5 now compare the
adapter with the frozen pre-refactor implementation under
`experiments/refactor_transformer/`; an independence guard prevents the old
and new arms from resolving to the same adapter. T4/Exp4 E2E was not rerun.
RAG R1–R3 are independent formula-level comparisons. Maze M1–M3 are adapter
contracts only; the active maze evaluator remains on the historical
application core and has no full-equivalence claim.

## 3. The canonical tests

- **Core correctness**: [`src/gedig/tests/`](../src/gedig/tests/) — the unified
  core suite
  (F composition, Betti, SP, AG/DG, independent Transformer/RAG comparisons,
  adapter contracts, and the active-maze golden trace).
  Run on every push by `ci-unit.yml`; the light CI has no torch, so the
  torch-only tests skip in the light job and run in a full local environment.
  `make test` executes both the application suite and this core suite.
- **Experimental claims**: verified only by pre-registered experiments in
  [`docs/prereg/`](prereg/) (analysis scripts committed before execution).
  CI passing says the code computes what it says; it says nothing about
  whether a scientific claim holds.

## 4. The canonical claim ledger

Three maturity levels, used consistently: **confirmed** (pre-registered +
replicated) / **preliminary** (single seed or single condition) /
**hypothesis** (design note only).

- Win/loss ledger across all pre-registered experiments: [prereg/README.md](prereg/README.md)
- Per-claim status table with lineage diagram (sleep line): [experiments/maze/graph_persistent_dg/README.md](../experiments/maze/graph_persistent_dg/README.md)
- Repo-wide status snapshot: [README Project Status](../README.md#project-status)

## 5. Document roles (where to look for what)

| Document | Role |
|---|---|
| [README.md](../README.md) | Current status summary, honest per-result caveats |
| [docs/MAP.md](MAP.md) | Navigation: directory map, terminology traps ("Phase" has four meanings), known debt |
| [docs/gedig_spec.md](gedig_spec.md) | Formal math of the ΔSP-form gauge (v4 One-Gauge Spec) |
| **docs/CANONICAL.md** | This page — the tie-breaker |
| [docs/prereg/](prereg/) | Pre-registrations and their outcomes (wins *and* defeats) |
