# geDIG Graph-IG Retirement Plan (Linkset-First)

Status: Draft v1 (for internal review)
Owner: Core/Experiments
Scope: Whole repo (Core API, experiments, docs, CI)

## Background

- Today there are two IG sources in geDIG:
  - Linkset IG (paper-aligned): entropy over candidate weights before/after adding query. ΔH = H_after − H_before, norm by log K.
  - Graph IG (approximation): mean local entropy from node features in graph neighborhoods. ΔH = after − before.
- Behavior differences (sign/scale) are expected because linkset measures distribution “spread” while graph-IG measures local “ordering/compactness”. This leads to practical confusion and diverging knobs.
- We will retire graph-IG from Core’s decision path and make linkset the single IG source, across all call sites.

## Goals & Non‑Goals

Goals
- Linkset IG becomes the only IG path used by `GeDIGCore.calculate(...)` and the experiment runners.
- Keep low‑level entropy utilities for research/tests, but decouple them from `GeDIGCore` decisions.
- Preserve performance/regression safety with telemetry and a staged rollout.

Non‑Goals
- We do not change SP gain semantics, GED normalization semantics, or spike detection modes aside from IG source.

## Affected Components

- Core
  - `src/insightspike/algorithms/gedig_core.py`
    - Single/multihop `calculate()` branches using graph-IG.
    - `ig_mode` (raw/norm/z) & `_ig_nonneg` application sites.
  - `src/insightspike/algorithms/core/metrics.py`
    - `entropy_ig()` remains as utility but is not wired into `GeDIGCore` decisions.
- Experiments
  - `experiments/maze-query-hub-prototype/run_experiment_query.py`
  - Other experiment runners using `core.calculate()` without `linkset_info`.
- Tests/CI
  - Unit tests that assert graph-IG through `GeDIGCore`.
  - CI smoke/unit selecting graph paths.
- Docs/Examples
  - Any quickstarts relying on graph-IG behavior or terminology.

## Rollout Plan

Phase 0 — Freeze & Defaults (now)
- Default maze runner to linkset (already supported via `--linkset-mode`).
- Add explicit `--ig-hop-apply` (all|hop0) to control hop scope. Default remains `all` for paper parity.
- Telemetry/logging: warn once when graph-IG path is taken in `GeDIGCore.calculate()`.

Phase 1 — Adapter & Coverage (1–2 PRs)
- Introduce a light “linkset adapter” helper for callers that currently lack linkset info:
  - Given candidate pool, chosen index, base mode (link/mem/pool), and query vector → build `linkset_info` consistently.
  - Provide per-hop variant (used internally by evaluators) when hop series are required.
- Update experiments to pass `linkset_info` everywhere they call into `GeDIGCore.calculate()`.
- Add a repo‑wide config preset (e.g., `paper`) that forces linkset IG and candidate‑base GED normalization.

Phase 2 — Deprecation Window (2–3 PRs)
- Mark graph-IG in `GeDIGCore` as deprecated (runtime warning + docs).
- CI: add a deprecation test ensuring the warning is emitted when graph-IG is forced.
- Docs: migration guide for moving to linkset (examples + code snippets).

Phase 3 — Removal (breaking PR)
- Remove graph-IG decision path in `GeDIGCore.calculate()` (single & multihop).
- Keep `entropy_ig()` in `core/metrics.py` as a utility with a clear “not used by Core decisions” docstring.
- Remove/modify tests that relied on graph-IG through Core; keep pure metric tests for `entropy_ig()`.
- Update CI workflows (drop graph-specific shards).

## API Changes (draft)

- `GeDIGCore.calculate(...)`
  - Require `linkset_info` for IG (no silent fallback to graph-IG). If absent, IG defaults to 0 unless the caller explicitly opts into a “hybrid” experimental flag (not recommended).
  - `ig_source_mode` becomes either `linkset` or `none` (remove `graph`, `hybrid`).
  - Preserve `ig_hop_apply` (all|hop0), `ig_mode` (raw|norm|z), `_ig_nonneg`.

## Migration Guide (for callers)

1) Build `linkset_info`:
- Inputs: `s_link` (or `candidate_pool`), `decision.index/similarity`, `query_entry{vector, similarity}`, `base_mode` (link|mem|pool).
- If `s_link` is empty and “forced candidates” exist, you may choose `base_mode=mem` and fall back to forced as base.

2) Call `GeDIGCore.calculate(g_prev, g_now, ..., linkset_info=...)`.

3) Optional per-hop series: evaluators build per-hop `linkset_info` or reuse hop0 ΔH depending on `ig_hop_apply`.

## Testing Strategy

- Unit
  - Ensure Core raises/warns when `ig_source_mode='graph'` is requested.
  - Keep metric tests for `entropy_ig()` with explicit expectations (sign/orientation = after−before).
- Experiments
  - Maze runner: verify ΔH sign, hop series, PSZ summary fields under linkset mode.
  - L3 integration (hop0‑only IG path) parity checks.
- CI
  - Update smoke/unit to run purely with linkset.

## Risk & Mitigations

- Risk: Callers without candidate pools cannot form `linkset_info`.
  - Mitigation: provide adapter with documented fallbacks (e.g., mem‑only pool). Define explicit behavior when pool is unknown (IG=0 vs error).
- Risk: Thresholds (θ_AG/θ_DG) drift under linkset.
  - Mitigation: provide recommendation fields (p95(gmin), g0 quantiles) and optional auto‑rerun to assist re‑tuning.
- Risk: Performance regressions.
  - Mitigation: measure eval latency distributions; keep `--ig-hop-apply hop0` option.

## Timeline

- P0 (this week): adopt linkset default in maze; add deprecation warnings; draft adapter.
- P1 (next): migrate primary experiments to linkset adapter; update docs; CI parity.
- P2 (following): remove graph-IG in Core; finalize tests; publish migration notes.

## Success Criteria

- No Core decision path uses graph-IG.
- All experiments pass with linkset IG and provide equivalent or improved acceptance/latency trade‑offs.
- CI green; docs updated; deprecation warnings removed.

## Open Questions

- Should we keep a hidden hybrid mode for research (off by default), or enforce a strict single‑source policy?
- How to handle cases with no meaningful `candidate_pool` (e.g., synthetic/no‑LLM pipelines)?

