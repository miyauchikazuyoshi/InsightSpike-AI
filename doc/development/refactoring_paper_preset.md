# Refactoring Plan — "Paper Preset" Alignment (Layer3 ↔ Experiment)

Goal: Make Layer3 (L3GraphReasoner) produce metrics consistent with the paper’s operational definitions and the Query‑Hub experiment route, so both routes agree numerically under the same inputs.

## Background

We observed metric drift between:
- Experiment route (Query‑Hub): geDIGCore + evaluator in `experiments/maze-query-hub-prototype` following paper definitions strictly.
- Layer3 route: generic engine defaults (after-based GED normalization, flexible IG sources, connected SP scope, hop-wise IG recomputation), which diverge from the paper.

Key differences causing drift:
- GED normalization: candidate-base (paper) vs edges_after (generic)
- IG (ΔH) normalization denominator: fixed log k★ (paper) vs fallback (generic)
- SP gain: fixed-before-pairs (paper) vs connected-pairs (generic)
- SP scope: union-of-k-hop (paper) vs auto (generic)
- λ (information temperature) and γ (SP weight) not synchronized

## Target behaviors (Paper Preset)

When using the “paper preset”:
- GED: `ged_norm_scheme='candidate_base'` (Cmax = c_node + |S_link|·c_edge)
- IG: `ig_source_mode='linkset'`, ΔH orientation after−before, denominator fixed to log k★ (from candidate_selection)
- SP: `sp_eval_mode='fixed_before_pairs'`, scope `sp_scope_mode='union'`
- λ/γ: matched to experiment route (env/explicit)
- Metrics local normalization enabled to stabilize decision-time thresholds

## Changes implemented (Progress)

1) L3 adapter alignment (experiments/qhlib/l3_adapter.py)
- Instantiate L3GraphReasoner with a config overlay:
  - graph: `sp_scope_mode='union'`, `ig_source_mode='linkset'`, `ged_norm_scheme='candidate_base'`, `sp_eval_mode='fixed_before_pairs'`, `linkset_query_weight=1.0`
  - metrics: `ig_denominator='fixed_kstar'`, `use_local_normalization=True`
- Pass `candidate_selection` into Layer3 context with `{k_star, log_k_star}` so ΔH denominator is fixed.
- Allow λ injection via `INSIGHTSPIKE_GEDIG_LAMBDA` when provided by the caller.

2) Experiment driver (run_experiment_query.py)
- In the `--use-main-l3` path, construct a minimal `candidate_selection` (k★ from linkset size) and pass to the adapter, plus optional λ.

3) Config centralization
- Added GraphConfig fields to manage paper knobs in one place:
  - `sp_scope_mode`, `sp_eval_mode`, `sp_hop_expand`, `sp_boundary_mode`
  - `ig_source_mode`, `ged_norm_scheme`, `linkset_query_weight`
  - `lambda_weight`, `sp_beta`
- Added `ConfigPresets.paper()` to produce a paper-aligned InsightSpikeConfig.

Note: The adapter overlay remains for call-site enforcement; modules can now also consume `ConfigPresets.paper()` or set fields via standard config plumbing.

4) First-class preset in L3 (done)
- `INSIGHTSPIKE_PRESET=paper` now enforces paper-aligned knobs inside `L3GraphReasoner` at initialization.
  - graph: `sp_scope_mode='union'`, `sp_eval_mode='fixed_before_pairs'`, `ig_source_mode='linkset'`, `ged_norm_scheme='candidate_base'`, `linkset_query_weight=1.0`, λ/γ kept from config (defaults 1.0/1.0)
  - metrics: `ig_denominator='fixed_kstar'`, `use_local_normalization=True`
  - This removes the need for the adapter overlay when running L3 in isolation under the paper preset.

## Progress log

- Baseline hardening (done)
  - Codex smoke narrowed to explicit safe files、ホーム直書きログ→repo内`results/logs`、MPLCONFIGDIR適用
  - 重依存テストに importorskip/skip を追加（収集エラー解消）
- Paper preset bridge (done)
  - L3 adapter overlay 実装（union/linkset/candidate_base/fixed_before_pairs + fixed_kstar + local_norm）
  - candidate_selection（k★/log k★/l1_candidates）と λ の同期注入
  - run_experiment_query の use-main-l3 パスから selection_summary と λ を渡す
- Config 一元化（done）
  - GraphConfig に paper ノブ追加（sp_scope_mode, sp_eval_mode, ig_source_mode, ged_norm_scheme, linkset_query_weight, lambda_weight, sp_beta ほか）
  - `ConfigPresets.paper()` 追加、CLI preset=paper で使用可
- L3 内部の挙動寄せ（partial, hop0）
  - query-centric 経路で `ged_norm_scheme=='candidate_base'` 時に k★ を使って Cmax を `norm_override` に適用（ΔGED_normの整合）

## Next steps / Decisions

- 迷路実験を paper preset で再走し、hop0 の一致度（g0差分、DG発火一致率）を短尺で評価（テスト計画参照）
- adapter overlay は「安全な上書き」用途に残しつつ、不要経路の削除は迷路実験での再現確認後に段階的に実施

## Operational improvements

- Long-run resilience: `run_experiment_query.py` supports `--checkpoint-interval N` to periodically write partial steps to `--step-log`.
- Fallback policy: added `--link-forced-as-base/--no-link-forced-as-base` to control whether forced Top‑L is used as base when `S_link` is empty (default: enabled).


## Rollout & Compatibility

- Default behavior remains unchanged unless `--use-main-l3` is used from the experiment or the adapter is called explicitly.
- The adapter overlay only affects the specific call-site.

## Risks & Mitigations

- If `k_star` is unavailable, ΔH denominator falls back. We ensure `k_star` is propagated from the experiment’s linkset size.
- λ mismatch: the adapter sets `INSIGHTSPIKE_GEDIG_LAMBDA` when provided; otherwise, Layer3 defaults apply.

## Next steps (optional)

- Promote a first-class `preset='paper'` in L3 (config/env) to make this alignment available across modules.
- Add tiny unit checks comparing experiment evaluator vs Layer3 “paper” output on a frozen micro‑graph (hop0 only) with fixed seeds.
