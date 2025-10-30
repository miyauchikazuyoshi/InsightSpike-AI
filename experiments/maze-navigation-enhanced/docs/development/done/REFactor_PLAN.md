# Maze Navigation Refactoring & Persistence Plan

## 1. Goals

- Scale wiring/query cost for larger mazes and long runs.
- Introduce persistence & memory control (backups + pruning).
- Externalize recall/backtrack tuning via CLI.
- Reduce geDIG metric overhead while retaining analytic signal.
- Prepare abstraction for future ANN / DataStore integration.

## 2. KPIs

| Metric | Current | Target |
|--------|---------|--------|
| Wiring step time (N≈2000) | O(N log N) full sort | O(N log k) or indexed |
| Memory peak (1000 steps) | Unbounded snapshots | ≤60% of current via cap |
| geDIG overhead ratio | 1.0 | ≤0.6 |
| Long run stability (5k steps) | Unverified | No OOM / <2× baseline time |
| Recall/BT A/B CLI | Missing | Available |

## 3. Phases

- P0: Lightweight perf (heapq, vector cache, snapshot cap).
- P1: Instrumentation (timers & stats exposure).
- P2: geDIG thinning (interval + idle skip).
- P3: CLI flags (recall, thresholds, k, snapshot limits, intervals).
- P4: Vector index abstraction + DataStore adapter.
- P5: Episode flush + lazy reload + memory thresholds.
- P6: Optional ANN backend (hnswlib/faiss) & benchmarks.

## 4. Tasks (Selected) ✔ Progress Checklist

Legend: [x] 完了 / [~] 部分 (PARTIAL) / [ ] 未着手 or 保留

- [x] T0-1 heapq top-k candidate episodes
- [x] T0-2 Weighted vector per-episode cache
- [x] T0-3 GraphManager max_snapshots + prune
- [x] T1-1 Timing instrumentation hooks
- [x] T1-2 Capture wiring_ms, gedig_ms, snapshot_ms, recall_ms
- [x] T2-1 Dense metric interval param (basic; diameter thinning 未)
- [x] T2-2 Skip snapshot when no growth (idle=2 heuristic)
- [x] T3-1 CLI flags (--global_recall, --recall_threshold, --wiring_top_k, --max_graph_snapshots, --dense_metric_interval, --snapshot_skip_idle)
- [x] T3-2 Extend summary with recall/backtrack event counts & latencies (records.json / summary.md)
- [x] T4-1 Add interfaces/vector_index.py (search(query, top_k))
- [x] T4-2 Implement InMemoryIndex
- [ ] T4-2 DataStoreIndex integration (STUB) ※保留
- [x] T4-3 Navigator uses injected index for query wiring (auto add & search with heap fallback)
- [x] T5-1 Episode flush_interval + max_in_memory (eviction scoring, dual-mode removal, events, counters, flush_ms timing, metadata catalog) ※ 永続リハイド pending
- [x] T5-1b Position-based capacity enforcement (`test_position_capacity_clamp`)
- [~] T5-2 Lazy load episodes (in-memory catalog rehydration 完 / persistence 未)
- [ ] T6-1 ANN backend plugin registration & evaluation script

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Stale weighted vectors | Track weight version & invalidate cache |
| Excessive IO on flush | Batch + configurable interval |
| ANN precision loss | Oversample (k*2) then refine distances |
| Snapshot pruning hides needed history | Keep sampled tail (every M steps) |
| Recall over-trigger | Cooldown param + growth-based gating |

## 6. Testing Strategy

- Unit: wiring produces ≤k new edges; cache invalidation.
- Perf micro: compare wiring time before/after (synthetic N=3000).
- Memory: snapshot cap enforced (len(history) ≤ max_snapshots).
- A/B: recall enabled vs disabled metrics diff stored.
- Index parity: DataStoreIndex results superset check vs linear.

## 7. Rollout Order

1. Phase 0 + 1 (safe, immediate benefit).
2. Validate metrics (multi-seed short runs).
3. Phase 2 + 3 (control + observability).
4. Stabilize then Phase 4 (flagged).
5. Phase 5/6 on demand.

## 8. Rollback Strategy

- Feature flags & fallbacks (None index → legacy path).
- Separate commits per phase; revert granularly.

## 9. Immediate Next Commits / Remaining Items (Checklist)

- [x] T4-1 / T4-3 and InMemory portion of T4-2 完了
- [x] T5-1 / T5-1b core eviction + position cap + timing + tests 完了
- [~] T5-2 in-memory rehydration (catalog-driven) ※ persistence 未

Next actions (優先順):

1. [ ] Persistence: DataStoreIndex real implementation (sqlite / fs) + evicted vector & metadata write + parity tests
2. [ ] Rehydration hardening (duplicate episode防止, catalog compaction (LRU/size cap), integrity counters)
3. [ ] Perf regression guard (baseline wiring_ms capture + pytest marker + <5% warn)
4. [ ] Pydantic migration (class Config → ConfigDict) deprecation除去
5. [ ] ANN backend prototype (hnswlib optional) + parity test + graceful fallback
6. [ ] CLI: expose --enable_flush, --max_in_memory, --max_in_memory_positions, --persistence_dir
7. [ ] Docs: memory control & rehydration guide (docs/guides/memory_control.md) + architecture diagram更新

Notes (latest fix batch): added flush_ms timing; dual-mode eviction stabilized & tested; eviction metadata catalog + on-demand rehydration integrated; position capacity clamp verified; next focus = persistence + ANN.

---
Archive Notice: 残タスク (Persistence 実装 / Rehydration hardening / ANN backend / CLI expose / Docs) は 新しい TEST_FAILURE_REMEDIATION_PLAN_2025_09.md の後続バックログへ移管予定。本ドキュメントはアーカイブ対象。

## 10. Future Extensions

- Branch priority queue for recall order.
- geDIG RLE compression.
- Adaptive top-k (dynamic by exploration progress).

## 11. Detailed Test Plan (テスト計画)

### 11.1 Categories

- Unit (単体): 個別メソッド/小粒度ロジックの正当性
- Integration (結合): Navigator 全体ステップ進行とメトリクス生成
- Performance (性能): 配線・geDIG・スナップショットの時間/メモリ
- Persistence (永続化): バックアップ & リセット動作とデータ整合
- Regression (退行防止): 既存 CLI / 出力フォーマット互換性

### 11.2 Core Test Cases

| ID | Scope | Description | Success Criteria |
|----|-------|-------------|------------------|
| U01 | wiring | `_wire_episodes_query_based` が k 上限以内で重複無くエッジ作成 | 追加エッジ数 ≤ k, 重複0 |
| U02 | cache | 重み付きベクトルキャッシュ: 同一 weights で再計算無し | call 回数 < 2 (初回のみ) |
| U03 | prune | `GraphManager.save_snapshot` が上限超過時に古い分を削除 | len(history) == max_snapshots |
| U04 | recall | `_maybe_global_recall` 閾値未満時のみプラン生成 | gedig > threshold で plan 生成0 |
| U05 | stagnation | `_detect_backtrack_stagnation` 条件境界テスト | しきい値ギリギリで True/False 適切 |
| I01 | full-run | 小さな迷路 (10x10) で goal 到達または max_steps 正常終了 | 例外なし & 統計 dict キー存在 |
| I02 | seeds-consistency | 同シード同設定で deterministic (simple/dfs) | path hash 一致 |
| I03 | cli-flags | 新 CLI フラグ指定で Navigator パラメータ反映 | stats で値一致 |
| I04 | timing-export | wiring_ms 等 timing キー出力 | stats['timing'] 全キー存在 |
| P01 | wiring-perf | N≈2000 で wiring 時間 baseline 比 < 0.7 | 平均 ms 減少 |
| P02 | snapshot-mem | max_snapshots=100 設定でメモリ上昇勾配抑制 | RSS 勾配減少 |
| P03 | dense-metrics | dense interval >1 で高コスト指標呼出削減 | 呼出回数 / step < baseline/interval |
| P04 | snapshot-idle-skip | growth 無し2連続で snapshot スキップ | snapshot 回数 < step 数 (期待差) |
| S01 | backup-basic | バックアップ実行で metadata.json & logs & results コピー | 必須ファイル存在 |
| S02 | prune-results | keep-results=2 で古い結果削除 | 残数 2 |
| R01 | output-shape | summary.md / records.json 既存キー保持 | 差分キー=追加のみ |
| R02 | fallback | index 未注入時に従来フルスキャン動作 | 例外なし & エッジ生成あり |

### 11.3 Edge / Boundary Cases

- k = 1, k = 0 (防御的処理) / k > N
- max_snapshots = 0 (保存抑制) / 1 / 大きな値
- recall_threshold = 0 / 非常に高値 (常時発火防止確認)
- 動的 backtrack と simple_mode の同時無効化
- 迷路全壁近似パターン（移動不能連続）

### 11.4 Tooling & Automation

- Add pytest markers: unit, perf, persistence
- Performance: 計測用 fixture で平均/分位数出力 (閾値失敗で warning → 将来 strict)
- Github Actions (将来): fast tier (unit+integration) / nightly (perf)

### 11.5 Metrics Collection

- wiring_ms, gedig_ms, snapshot_ms, recall_ms (移動平均)
- memory_est (len(history), node/edge counts)
- recall/backtrack event rate vs geDIG_low_frac 相関 (csv)

### 11.6 Pass/Fail Gate

- Unit & Integration 失敗 0
- Regression: 破壊的キー削除なし
- Perf: wiring_ms 改善が 5% 未満なら WARN (ログ) だが失敗扱いはしない (初期)

## 12. Backward Impact Checklist (後方影響確認リスト)

| Area | Risk | Check Method | Status (init) |
|------|------|-------------|---------------|
| CLI 互換 | 既存引数破壊 | 旧コマンドライン再実行 | 未実施 |
| 出力フォーマット | summary.md / records.json キー欠損 | スナップショット diff | 未実施 |
| ログ語彙 | 解析スクリプトが期待する event type 変更 | grep 既存 event 種別 | 未実施 |
| 再現性 | シード結果変動 | 同シード比較 (hash/path) | 未実施 |
| メモリ使用 | snapshot pruning バグで漏れ | 長尺 run RSS 監視 | 未実施 |
| パフォーマンス | heapq 導入でオーバーヘッド | プロファイル計測 | 未実施 |
| 回想/BT | 過剰発火で探索遅延 | event rate モニタ | 未実施 |
| DataStore 統合 (将来) | フォールバック不整合 | index=None テスト | 未実施 |
| ライセンス/依存 | 新規 ANN ライブラリ | 依存ライセンス確認 | 未実施 |
| CI | テスト時間増 | 測定 & キャッシュ導入 | 未実施 |

### 12.1 Verification Steps

1. Capture baseline metrics before Phase 0 commit.
2. Apply Phase 0; run seed set; compare wiring_ms / edge count.
3. Run backup script; verify metadata completeness.
4. Execute rollback dry-run (revert commit) to ensure clean reversal.

### 12.2 Sign-off Criteria

- All checklist rows marked OK or documented waiver.
- Baseline vs post-refactor: no >5% degradation in path_length, unique_coverage for same seeds (variance normalized) unless justified.

---
Added sections 11 & 12 to extend original plan.

(Generated and stored automatically)
