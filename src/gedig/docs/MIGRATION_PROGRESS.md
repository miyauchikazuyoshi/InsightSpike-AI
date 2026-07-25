# Adapter Migration Progress

> **Status date:** 2026-07-24
> This is the factual migration ledger. `MIGRATION_PLAN.md` is the historical
> pre-migration plan and is not evidence that an E2E run occurred.

## Current boundary

| Domain | Active experiment path | State | What is verified | What is not claimed |
|---|---|---|---|---|
| Transformer delta F | `experiments/transformer/thermodynamic_gedig.py` | **Delegated** to `TransformerFEval`; `use_unified` is an inert compatibility input | Unified values and gradients match the frozen pre-refactor oracle | T4/Exp4 E2E was not rerun in this debt-repayment phase |
| RAG / AGHT | `experiments/hotpotqa_v2/src/unified_graph.py` | **Delegated** to `RAGFEval`; `use_unified_feval` and `use_unified` are inert compatibility inputs | R1–R3 independently check edge F, partition, and propagation formulas | R4–R6 are historical experiment records, not runnable P6 regression tests |
| Maze | `experiments/maze/qhlib/evaluator.py` | **Legacy fixed** on `insightspike.algorithms.gedig_core.GeDIGCore` | `MazeFEval` contracts plus a golden trace of the active SP evaluator | No drop-in equivalence, active cutover, gate equivalence, sleep equivalence, M4, or M5 |

`src/gedig/` is the canonical implementation for new code. Archived
pre-refactor implementations under `experiments/refactor_*` are read-only
oracles and provenance records, not alternate production paths.

## Transformer

The active `DifferentiableGeDIG` wrapper delegates every before/after delta
evaluation to `gedig.adapters.transformer.TransformerFEval`. Since commit
`eed1ae4`, passing `use_unified=False` to that active wrapper does not select
the old algorithm.

The independent comparison now imports:

```text
old: experiments/refactor_transformer/thermodynamic_gedig_legacy.py
new: experiments/transformer/thermodynamic_gedig.py
     → src/gedig/adapters/transformer.py
```

The regression suite prevents another self-comparison by asserting that the
old class comes from `thermodynamic_gedig_legacy`, keeps
`use_unified=False`, and never constructs `_unified_adapter`.

Verified locally:

- SP and β₁ component values and full F tensors;
- 100 deterministic random samples;
- masked tensors and non-default λ, γ, percentile, and temperature;
- direct gradient tensor equality (within explicit numerical tolerance);
- a true old/new forward-speed guard.

T4 is not present as an automated E2E test and was not rerun. Historical
artifacts produced after `eed1ae4` may record `"use_unified": false` even
though the active wrapper actually used the unified adapter; that field
records the requested CLI value, not reliable backend provenance.

The separate `experiments/transformer/train_f_regularized.py` calculator is a
single-state experimental profile. It is not the canonical before/after delta
API and is not part of this adapter migration claim.

## RAG / AGHT

The active AGHT edge evaluation, percentile partition, and relevance
propagation all call `RAGFEval`. The retained config/function/CLI flags do not
switch implementations; their help and comments mark them as deprecated
no-ops.

R1–R3 remain independent formula-level comparisons:

- R1 compares per-edge F with a manual QK dot-product formula;
- R2 compares the complete edge partition with a manual percentile split;
- R3 compares every propagated node value with a local legacy loop.

The prior HotpotQA and BRIGHT E2E numbers remain experiment records. P6 did
not rerun datasets or external-model evaluation, so it does not upgrade their
evidential status or claim R4–R6 as current executable tests.

## Maze

The active maze evaluator is intentionally not switched in this phase. It
combines experiment-specific Cmax normalization, linkset IG, scoped/sampled
shortest-path evaluation, hop selection, and runner-side gates. `MazeFEval`
uses the generic NetworkX snapshot backends and is therefore not a drop-in
replacement.

The M1–M3 tests are adapter contracts only:

- F composition on small NetworkX graphs;
- two-stage gate threshold semantics;
- canonical synchronous Q-style propagation.

They do not import the active evaluator. The active evaluator is instead
protected by
`src/gedig/tests/fixtures/maze_active_sp_trace.json` and
`test_maze_active_trace.py`, which freeze g0, gmin, best hop, selected edge,
and the hop-series components for one deterministic SP-mode graph. The
fixture explicitly makes no adapter-equivalence claim.

Active maze sleep propagation also has different update semantics from the
canonical propagator. Cutover requires a separate post-v7 design and E2E
phase. The ongoing v7 files are protected from this refactor.

## Test entry points

```bash
# Canonical core, adapters, frozen Transformer oracle, active maze trace
PYTHONPATH=src .venv/bin/python -m pytest -q src/gedig/tests

# Public Flash profile/delta boundary
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/unit/test_flash_gedig_api.py
```
