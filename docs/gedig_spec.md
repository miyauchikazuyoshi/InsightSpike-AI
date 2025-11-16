# geDIG (Generalized Differential Information Gain) — v4 One‑Gauge Spec

このドキュメントは、論文 v4（One‑Gauge + 二段ゲート）および実装の現行仕様に整合する正準定義を示します。

## 正準ゲージ定義（論文 v4）

- 記号対応
  - ΔEPC_norm: 正規化済み編集パスコスト（構造コスト）。実装の `ged_norm_scheme` で規定。
  - ΔH_norm: 固定分母（log K）でのエントロピー差（after − before）。
  - ΔSP_rel: 相対最短路ゲイン（(L_before − L_after)/L_before）。
  - λ: 情報温度（`lambda_weight`）。
  - γ: SPゲイン重み（`sp_beta`）。

- ゲージ（F）
```
F = ΔEPC_norm − λ ( ΔH_norm + γ · ΔSP_rel )
```

### 正規化と分母
- ΔH_norm は `log K` で正規化（K は after の候補数）。K < 2 の場合は ε でガード。
- ΔEPC_norm は上界で割る近似。論文プリセットでは `ged_norm_scheme = "candidate_base"` を既定（候補集合サイズに基づく上界）。

### 二段ゲート（AG/DG）
- 0-hop: `g0 = ΔEPC_norm − λ·ΔH_norm`
- multi-hop: `gmin = min_h { ΔEPC_norm − λ(ΔH_norm + γ·ΔSP_rel^(h)) }`
- 受理規則（例）: AG が高い新規性を示し、かつ `min{g0, gmin} ≤ θ_DG` のとき採択。閾値は設定依存。

## 実装マッピング（主要パラメータ）
- `src/insightspike/algorithms/gedig_core.py`
  - γ: `sp_beta`
  - λ: `lambda_weight`
  - ΔH_norm: `delta_h_norm`（after−before で負が秩序化）
  - ΔSP_rel: `delta_sp_rel`
  - 正規化戦略: `ig_norm_strategy`, `ged_norm_scheme`
- `src/insightspike/implementations/layers/layer3_graph_reasoner.py`
  - 合成IG: `_ig_norm = ΔH_norm + sp_beta * ΔSP_rel`
  - プリセット適用（paper向け）: scope=`union`, eval=`fixed_before_pairs`
- `src/insightspike/config/presets.py` の `paper()`
  - `graph.sp_scope_mode = "union"`
  - `graph.sp_eval_mode = "fixed_before_pairs"`
  - `graph.ged_norm_scheme = "candidate_base"`
  - `graph.ig_source_mode = "linkset"`
  - `graph.lambda_weight = 1.0`, `graph.sp_beta = 1.0`
  - `metrics.ig_denominator = "fixed_kstar"`, `metrics.use_local_normalization = True`

## 関連アルゴリズム
- 情報利得（ΔH）: `src/insightspike/algorithms/information_gain.py`
- 構造コスト/改善: `src/insightspike/algorithms/graph_structure_analyzer.py`
- ゲージ中核: `src/insightspike/algorithms/gedig_core.py`

## PSZ 指標（動的 RAG）
- 受理率/偽受理/レイテンシを集約: `src/insightspike/metrics/psz.py`

## 互換ノート
- 旧式スパイク検出（ΔGED/ΔIGしきい値）はレガシー互換のため残置: `src/insightspike/detection/eureka_spike.py`

## 付録: 旧・教育用の簡易式（非推奨）
```
geDIG ≈ GED − IG
```
学習用途の近似式。実装/論文 v4 の正式ゲージとは異なるため、新規コードや評価では本ドキュメント冒頭の定義を使用してください。

## 更新履歴
- 2025-11-14: v4（One‑Gauge）に整合するよう全面更新。実装マッピングを追記。
