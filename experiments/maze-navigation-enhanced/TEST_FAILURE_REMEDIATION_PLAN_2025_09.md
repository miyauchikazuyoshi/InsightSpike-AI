---
title: Test Failure Remediation Plan (Sept 2025)
status: active
created: 2025-09-01
owner: quality-fix
scope: unit/integration normalization
---

## 1. Objective

短期で pytest 失敗を系統別に潰し、後続の geDIG/GED キャリブレーション作業を安全に進められるグリーン基盤を確立する。

## 2. Failure Clusters & Target Milestones

| Milestone | 範囲 | 内容 | 期待 Fail 減 | 目標期間 |
|-----------|------|------|--------------|-----------|
| M1 | Spike/Knowledge/Sphere/Edge | (Spike bool ✅ 実装済) / Knowledge counters / SphereSearch 境界 inclusive ✅ / EdgeReevaluator 空ガード ✅ | 15-20 | Day 1 |
| M2 | IG / Config | LocalInformationGainV2 正方向化 (variance ベース TODO) + MemoryConfig 属性デフォルト (faiss_index_type 等) ✅ (デフォルト挿入 + テスト) | 5-8 | Day 2 |
| M3 | GED/GeDIG Calibration | GED 正規化係数調整 + 効率/改善指標期待値整合 | 残り大半 | Day 3-4 |
| M4 | Cleanup | Deprecation / Warning 整理 (低優先) | 0 (品質向上) | Day 5 |

## 3. RICE Priority (概算)

| Task | Reach | Impact | Confidence | Effort (d) | RICE | 順位 |
|------|-------|--------|------------|------------|------|------|
| Spike 戻り値正規化 | 100% (該当全テスト) | High | High | 0.1 | 1000 | 1 |
| Knowledge counters 修正 | 60% | High | High | 0.15 | 600 | 2 |
| SphereSearch 境界修正 | 30% | Med | High | 0.05 | 300 | 3 |
| EdgeReevaluator 安全化 | 25% | Med | High | 0.1 | 250 | 4 |
| Local IG クリップ/符号 | 40% | Med | Med | 0.3 | 160 | 5 |
| MemoryConfig デフォルト | 40% | Med | High | 0.1 | 400 | 3.5 (再優先) |
| GED 指標キャリブレーション | 100% | High | Med | 1.5 | 333 | 5 (後ろ倒し) |
| Warnings Cleanup | 100% | Low | High | 0.5 | 200 | 6 |

(注) MemoryConfig は Effort 極小のため順位再調整で 3.5 扱い。

## 4. Implementation Plan (Detail)

### M1 詳細

1. Spike: `Layer3GraphReasoner.detect_eureka_spike` ラッパー/adapter で tuple -> bool 返却; `context_spike_score` を内部 `_last_spike_context` に格納 ✅ Flag: `SPIKE_BOOL_WRAPPER=1`。
2. KnowledgeManagement: `add_*` 系で内部カウンタ (total_facts, total_relations, updates) を更新; `get_stats()` / `__repr__` 拡張; ゼロから正数への遷移を debug ログ。
3. SphereSearch: 半径比較 `dist < radius` → `<=` に統一 ✅ Flag: `SPHERE_INCLUSIVE=1` (デフォルト ON)。
4. EdgeReevaluator: `edge_index` サイズ / None チェックで早期 return; metrics: `skipped_empty=True` ✅

### M2 詳細

1. LocalInformationGainV2: `ig_raw = before_var - after_var`; `ig = max(0, ig_raw) + bonus_factor*new_nodes`; Flag: `IG_CLIP_EXPERIMENTAL=1` (A/B)。ログ: raw/clip/bonus。

	- 実装状況: metrics_selector 内で `IG_CLIP_EXPERIMENTAL` フラグ有効時に `max(0, raw) + 0.01*new_nodes` を適用し debug ログ出力対応 (簡易クリップ版 ✅)。 before/after variance 測定による `ig_raw` 算出は未実装 (TODO)。
2. MemoryConfig: `faiss_index_type`, `metric` 不在時にデフォルト挿入 (e.g. 'FlatL2'); Adapter / normalizer / L2 dict init で fallback 実装済 ✅; `config_summary` に `defaults_applied` 配列 (TODO 暫定)。

### M3 詳細

1. GED Calibration: 収集フィールド raw_ged, possible_ged, efficiency = raw/(possible+eps); 正規化係数 α 調整用 env: `GED_ALPHA`; 改善/効率をテスト期待値帯域 (例: 0<=eff<=1, structural_improvement 非負) に収束。実装手順: (a) ログ収集 (b) 分位点比較 (c) 係数更新 (d) 再テスト。
2. geDIG Efficiency: 既存 negative spike 誤検知率ログ; optional smoothing EMA (env: `GEDIG_EFF_EWMA=window`) 実験。

### M4 詳細

1. Warning Cleanup: Deprecation (Pydantic Config), noisy debug の段階的削除; `--no-warn` モード検討。

## 5. Flags / Env Vars

| Flag | Default | Purpose |
|------|---------|---------|
| SPIKE_BOOL_WRAPPER | 1 | Spike 戻り値 bool 安全化 |
| SPHERE_INCLUSIVE | 1 | SphereSearch 境界 inclusive |
| IG_CLIP_EXPERIMENTAL | 1 | IG クリップ + ボーナス適用 |
| GED_ALPHA | 1.0 | GED 正規化調整係数 |
| INSIGHT_DEBUG_METRICS | 0 | GED/IG 詳細ログ有効化 |
| GEDIG_EFF_EWMA | 0 | geDIG 効率 EMA 平滑ウィンドウ |

## 6. Logging Additions

- Spike: `spike_detected`, `spike_score_raw`, `wrapped_bool`。
- Knowledge: `before_counts`, `after_counts`, `delta`。
- IG: `ig_raw`, `ig_clipped`, `new_node_bonus`。
- GED: `raw_ged`, `possible_ged`, `efficiency`, `alpha_used`。
- EdgeReevaluator: `skipped_empty`, `edges_before`, `edges_after`。

## 7. Test Update Strategy

- 既存失敗テストはインターフェースのみ変更 (期待 bool)。
- 新規 quick tests: sphere inclusive, knowledge counters increment, IG clip non-negative.
- Calibration: 暫定 failing を xfail マーク (GED 調整期間中) → 安定後解除。

## 8. Risk Mitigation Recap

- 破壊的変更は flag で可逆化。
- A/B ログ並列出力でオーバーフィッティング防止。
- 一括 commit 回避 (M1/M2/M3 分割)。

## 9. Acceptance Criteria

- M1 完了後: Spike/Knowledge/Sphere/Edge 関連失敗 0。
- M2 完了後: IG/Config 関連失敗 0、IG 負値 0。
- M3 完了後: GED/GeDIG 指標テスト期待値帯域内 (全緑 or xfail=0)。
- M4 完了後: pytest warnings (own code) < 5。

## 10. Timeline (Target)

- Day1 午前: M1 実装 + 部分テスト
- Day1 午後: フルテスト → 残クラスタ再分類
- Day2: M2 実装 & 緑化
- Day3-4: M3 ログ→調整→確定
- Day5: M4 & 文書更新

## 11. Out-of-Scope

- 性能最適化 (後でベンチ)
- ANN 実装本体 (別計画) / DataStore 永続化
- geDIG Phase 3 高度式再定義 (本計画では触れない)

## 12. Archive Note

旧 `REFactor_PLAN.md`, `SIMPLE_MODE_REFACTOR_PLAN_2025_08.md` などは archive/ に移動済み (参照用)。

## 13. Backlog (Consolidated from Archived Plans)

優先: Remediation M1-M4 完了後に順次再評価。RICE/依存で再スコア予定。

### 13.1 Simple Mode 残 (Phase B)

- [ ] B2 README 更新 (Simple vs Legacy 差分 / 移行ガイド)
- [ ] B3 テスト追加 (4 ケース: query_single_generation, backtrack_simple_static_only, select_action_query_reuse, statistics_simple_mode_block)
- [ ] 指標計測: backtrack_trigger_rate ログ安定化 (A/B) ※ B1 部分実装後集計
- [ ] query_generated_per_step メトリクス検証 (期待=1.0±0.01) → Evidence パイプラインに統合

### 13.2 Refactor Persistence / Index 拡張

- [ ] DataStoreIndex 実装 (sqlite/fs) + parity test (linear fallback superset)
- [ ] Rehydration Hardening: duplicate 防止 / catalog compaction / integrity counters
- [ ] Perf Regression Guard: baseline wiring_ms 計測 + <5% 劣化検知 (warning)
- [ ] Pydantic v2 移行 (Config -> ConfigDict) & deprecation 除去
- [ ] ANN Backend Prototype (hnswlib/faiss) + oversample refine strategy
- [ ] CLI 拡張: --enable_flush, --max_in_memory, --max_in_memory_positions, --persistence_dir
- [ ] Docs: memory_control.md & rehydration 図追加

### 13.3 geDIG Enhancement Phase 2

- [ ] P2-1 ExperimentRecorder (JSONL/summary)
- [ ] P2-2 wiring dispatch + gedig 戦略 stub
- [ ] P2-3 geDIG 分解ログ (gain/penalty)
- [ ] P2-4 dead-end ground truth (BFS degree=1 内部ノード)
- [ ] P2-5 backtrack precision/recall 指標集計
- [ ] P2-6 batch runner + sweep (threshold × backtrack × strategy)
- [ ] P2-7 Pareto 抽出 & best_config.json
- [ ] P2-8 README 結果反映

### 13.4 Evidence 強化

- [ ] サイズスケーリング: generator + grid 実験 + 図 (line / heatmap)
- [ ] geDIG Leave-One-Out (LOO) 特徴寄与算出 + loo_stats.json + feature_contribution.md
- [ ] 外れ値診断: P95 抽出 + outliers.json / outliers.md
- [ ] パイプライン自動化: Makefile target maze_evidence + collect_maze_evidence.py
- [ ] 総合レポート evidence_report.md 生成
- [ ] metrics_utils ユニットテスト (冗長度 / effect size / AUC)
- [ ] smoke run (n=5) → フル run (n=30) 検証

### 13.5 CI / テスト インフラ (残差)

- [ ] Canonical spec uniqueness CI チェック (PLAN_INDEX 自動検証)
- [ ] PLAN_INDEX 自動生成スクリプト (front matter 走査) + pre-commit hook

### 13.6 Future (Post-Stability)

- [ ] geDIG Phase 3 数式再定義 & 係数 sweep (α..ζ)
- [ ] Pruning 戦略 (低 impact node 圧縮)
- [ ] Adaptive thresholds (EWMA + 管理図) wiring/backtrack
- [ ] ANN index ベンチ (速度/精度カーブ)
- [ ] Long-run memory leak regression test (5k steps)

Backlog 総数: Simple(4) + Refactor(7) + geDIG(8) + Evidence(7) + CI(2) + Future(5) = 33 項目。

優先再評価基準:
 
1. 失敗削減 / 安定化寄与 (直近 2 日)
2. 計測基盤整備 (ExperimentRecorder / Evidence)
3. 構造最適化 & メモリ (Persistence / Pruning)
4. 研究的改善 (Phase 3 / ANN)

M1 完了後に 13.3 (Recorder + dispatch) から着手するか再確認する。

---
Generated automatically.
