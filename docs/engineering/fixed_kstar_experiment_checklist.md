# 固定分母（log K⋆）適用チェックリスト

## 対象
- Maze オンライン実験 (`experiments/maze-online-phase1*`)
- RAG v3-lite (`experiments/rag-dynamic-db-v3-lite`)

## 手順概要
1. 設定ファイルで `metrics.ig_denominator` を `fixed_kstar` に設定し、`metrics.use_local_normalization` を `true` にする。
2. 候補選択しきい値を `metrics.theta_cand`, `metrics.theta_link`, `metrics.candidate_cap`, `metrics.top_m` で指定する。
3. 実行前に `query-recorder` / `graph_analysis` 出力に `candidate_selection` サマリ（`k_star`, `theta_cand`, `theta_link`, `k_cap`, `top_m`, `log_k_star`）が含まれているか確認する。
4. 実験実行後、AG/DG 閾値と λ を既存分位に基づいて再調整し、差分ログを `results/.../gedig_metrics.csv` に追記する。

## RAG v3-lite サンプル
```yaml
metrics:
  theta_cand: 0.45
  theta_link: 0.35
  candidate_cap: 32
  top_m: 32
  ig_denominator: fixed_kstar
  use_local_normalization: true
```

## Maze オンライン簡易設定
`experiments/maze-online-phase1-querylog/src/gedig_adapter.py`
- `GeDIGAdapter` 初期化時に `ig_norm_strategy` を `"before"` から `"before"` + `k_star` 供給へ移行予定。
- `maze_online_config.yaml` に上記 `metrics` セクションを追加し、Navigator 経由で MainAgent に渡す。

## 確認事項
- [ ] `selection_summary["k_star"]` が 2 以上にならないケースのハンドリング（既に安全値 0 → IG=0）。
- [ ] `candidate_selection` がログに含まれない場合のフォールバック（旧挙動と同じ）。
- [ ] `use_local_normalization=true` で Cmax=1+K⋆ が意図した通りに作用しているか、テンプレ `tests/unit/test_gedig_core_local_norm.py` をベースに現場グラフでも sanity check。
