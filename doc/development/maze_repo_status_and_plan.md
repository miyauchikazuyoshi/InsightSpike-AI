# 迷路実験の改善・残タスクとリポジトリ改善ポイント（バックアップ/プッシュ方針含む）

目的: 迷路PoC→RAG実証の順でやり直す計画に合わせ、現状の改善点・残タスク・push前の整理/バックアップ方針を一本化して記録する。

---

## 迷路PoC 改善点（適用済み/適用予定）
- SP計算の効率化（DistanceCache + PairSetレジストリ）
  - 実装: `src/insightspike/algorithms/sp_distcache.py`
  - メモリ/Fileレジストリ切替（ENV: `INSIGHTSPIKE_SP_REGISTRY`）
  - between-graphs ΔSP 推定と固定ペア再利用
- L3 SPエンジン切替（`core`/`cached`/`cached_incr`）
  - 実装: `src/insightspike/implementations/layers/layer3_graph_reasoner.py`
  - `cached_incr` は候補エッジ単位の貪欲・逐次採用（sequential、相互作用考慮）
    - 予算制御: `graph.cached_incr_budget` または ENV `INSIGHTSPIKE_SP_BUDGET`
  - 候補が無い場合は自動で `cached` にフォールバック
  - 追加: 候補未提供時は L3 内で `graph.x` と `centers` から自動提案（Top‑K/θ_link）
  - 追加: L3（global, 非‐union）経路で `sp_before`/`sp_after` を metrics に格納（UIの H/SP 列と整合）
- NormSpec（ノルム仕様）の受け渡し
  - NormalizedConfig で `graph.norm_spec` を提供し、ExplorationLoop が `context['norm_spec']` に格納
  - L3 は metrics に `norm_spec` をエコー（再現性のため）
- テスト
  - `tests/unit/test_l3_cached_incr_basic.py`（候補エッジ＋NormSpec の基本動作）
  - `tests/unit/test_sp_distcache_registry.py`（レジストリの保存/再利用）
  - `tests/unit/test_gating_and_psz.py`（AG/DG用のPSZ基準）

---

## 残タスク（短期）
- candidate_edges 自動提案のパラメータ最適化
  - `centers`/`top_k`/`theta_link` の既定値の調整と config 露出
  - 重複/無効エッジ、空集合時の堅牢化（ログ・メトリクス）
- `cached_incr` 逐次適用の最適化（計算効率の改善）
  - 現状は greedy sequential（各候補を一時追加→評価）。pair_samples/候補数/予算に応じた高速化（SSSP再利用）
- 追加テスト
  - 小規模グラフで `core` vs `cached_incr` の順位相関/劣化しないことの保証
  - 空/重複/無効候補、budget > |Ecand| 等のエッジケース
  - L2→L3 結合テスト（centers/candidate_edges/norm_spec 伝播のE2E）
  - L3‐union デバッグ（cand_used/cand_total）を JSON に反映して HTML の hop 表に表示（現在は列のみ追加済み）

---

## リポジトリ全体の改善ポイント
- config 設計の明確化
  - `graph.sp_engine`, `graph.norm_spec`, `graph.cached_incr_budget` を `README`/APIリファレンスに反映
  - ENV > config > 既定 の優先順位を明記
- ドキュメント更新（仕様・アーキテクチャ・図）
  - `doc/development/l1l2_l3_sp_engine_spec.md` へ進捗反映済み（NormSpec 伝播/自動候補）
  - `docs/api-reference/*` と `docs/architecture/*` に SPエンジン切替と NormSpec データフローを追記
  - 迷路 HTML（`query_interactive_template.html`）に `cand_used`/`cand_total` 列を追加（L3‐union デバッグ可視化の受け皿）
- 実行ログの軽量化
  - DIAGログ（import markers）と通常ログの切り分け、既定 INFO の静穏化

---

## バックアップ/プッシュ方針（main のクリーン化）

合意内容:
- main を履歴書き換えでクリーン化 → 強制 push
- 例: `snapshot/2025-10-30-clean` の状態に main をリセット
- 直前に main に入れていた小実装（GateDecider/PSZ/hello_gating/—no-link-autowire-all/docstring 強化）はクリーン後に再適用済み（現状 main 反映済み）

実施手順（再現用メモ）:
```
# 1) 念のためローカルスナップショット作成
git branch backup/$(date +%F)-pre-clean

# 2) main をスナップショットにリセット
git checkout main
git reset --hard snapshot/2025-10-30-clean

# 3) 必要実装を再適用（反映済み: GateDecider/PSZ/L3 cached_incr/DistanceCache/Docs/Tests）
#    → 反映済みブランチとの差分確認
git log --oneline -n 10

# 4) テスト実行（最小セット→拡大）
pytest -q tests/unit/test_l3_cached_incr_basic.py \
          tests/unit/test_sp_distcache_registry.py \
          tests/unit/test_gating_and_psz.py

# 5) 強制 push（同意済み）
git push origin main --force
```

注意:
- 破壊的操作（`--force`）のため、共同作業者の追随手順を別途告知
- `INSIGHTSPIKE_SP_REGISTRY` を運用環境で設定（ペア集合をプロセス間で再利用する場合）

---

## 仕様とテスト計画の更新フロー
1) 仕様（本ドキュメントと `l1l2_l3_sp_engine_spec.md`）の「進捗/未達」を更新
2) config と ENV ノブの API リファレンス追記
3) 図（centers/Ecand/norm_spec フロー）更新
4) ユニット/結合テストの拡充（優先度順）

---

## 補足QA
- Q: 迷路の SP 効率化は近似？
  - A: `cached` は between-graphs の固定ペア再計算（厳密な before/after 比較上の近似）。`cached_incr` は候補ごと貪欲で相互作用を無視する近似（初期統合に十分）。逐次適用（相互作用考慮）は今後拡張。
- Q: Layer3 でも候補エッジ単位の最適化を使いたい
  - A: `graph.sp_engine: cached_incr` を指定。`candidate_edges` が無い場合でも L3 が `graph.x` と `centers` から自動提案（Top‑K/θ_link）。
- Q: NormSpec の統一と受け渡しは？
  - A: `NormalizedConfig` の `graph.norm_spec` を ExplorationLoop から `context['norm_spec']` に渡し、L3 metrics にエコー。Layer 間で一貫した半径/閾値を担保。
