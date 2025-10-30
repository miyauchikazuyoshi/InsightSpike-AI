# Maze Navigation 50x50 実験 & パフォーマンス最適化 チェックリスト (2025-09)

目的: 50x50 迷路ナビゲーション実験において、探索精度 (成功率 / カバレッジ / 新規ノード比) を維持しつつ 1 ステップ当たり計算コストを現在のほぼ線形 O(N) から O(log N) 〜 O(k)（k=候補数上限）へ近づける。合わせて再現性・可観測性を強化し、スイープ比較をしやすくする。

---

## 0. メタ情報

- 作成日: 2025-09-02
- 更新担当: （更新時に追記）
- 対象コード: `examples/maze50_experiment.py`, `maze_navigator.py` 他
- 主要ログ: `logs/maze_sweep_*.log`
- 主要メトリクス: coverage, novelty_ratio, revisit_ratio, nodes/edges growth, success, step_time_ms

---

## 1. 現状把握 (Baseline) 状態

| 項目 | 状態 | メモ |
|------|------|------|
| graph_growth イベント出力 | ✅ | verbosity=1 でも表示可能にパッチ済み |
| visit_weight_scale=1.0 ラン稼働 | ✅(進行中) | 完了後 summary 抽出必要 |
| run summary f-string バグ修正 | ✅ | 再現なし |
| メトリクス (novelty/revisit/coverage) ログ化 | ✅ | interval growth 行に含む |
| ANN 自動アップグレード発火確認 | ⬜ | 発火ログ行 (e.g. HNSW init) 未確認 |
| スイープ (1.6,1.8) 未実行 | ⬜ | baseline 1.0 完了後 |
| 高スケール追加検討 (2.0,2.5) | ⬜ | 1.6/1.8 の差分見て決定 |

---

## 2. 完了済タスク

- [x] graph_growth 出力制御パッチ (emit_event)
- [x] ログ unbuffered 実行運用
- [x] 進捗 / growth interval ログ整備
- [x] 実験メタデータ (commit, seed, sweep list) 出力
- [x] run summary フォーマット修正

---

## 3. 未完了 / 実施予定タスク (チェックしながら運用)

### 3.1 Baseline 実験完遂

- [ ] (B1) scale=1.0 ラン完了を検出し summary を保存 (`results/` へ JSON も可)
- [ ] (B2) scale=1.0 の step_time_ms 推移を抽出 (開始/中盤/終盤)
- [ ] (B3) scale=1.6 ラン実行 & summary 保存
- [ ] (B4) scale=1.8 ラン実行 & summary 保存
- [ ] (B5) 1.0 / 1.6 / 1.8 比較レポート (成長曲線重ね合わせ)
- [ ] (B6) 追加スケール (2.0 / 2.5) 要否判定 & 必要なら実行

### 3.2 計測強化 (Instrumentation)

- [ ] (I1) 1 ステップ処理時間計測 (壁時計 + CPU) を `maze_navigator` にオプション追加
- [ ] (I2) wiring 部分と geDIG 部分の分離計測
- [ ] (I3) スナップショットコピー有無のコスト計測フラグ
- [ ] (I4) metrics 出力に step_time_ms (avg / p95) 追加

### 3.3 最適化 Phase 1 (安全・ノーリスク)

- [ ] (P1-1) coverage 計算を増分更新 (set 再生成 → カウンタ)
- [ ] (P1-2) novelty/revisit も増分更新（累積カウンタ）
- [ ] (P1-3) ANN 早期初期化閾値を CLI で指定 (例: `--ann-threshold 200`)
- [ ] (P1-4) ANN 初期化ログ行追加 (旧→新 index 種別, N)
- [ ] (P1-5) 変更後の正当性テスト (小規模迷路 10x10) で前後 diff なし確認

### 3.4 最適化 Phase 2 (候補集合縮小 + geDIG 軽量化)

- [ ] (P2-1) wiring 候補 sampling 上限 k (例: k=64) 導入 + ランダム/スコア優先戦略
- [ ] (P2-2) geDIG 呼び出しをトリガ方式へ (成長Δ≥X or stagnation≥Y steps)
- [ ] (P2-3) スナップショット取得間隔を指数バックオフ (1,2,4,8,... or growth-trigger)
- [ ] (P2-4) diameter 計算を常時無効化 or 近似 (オプション化)
- [ ] (P2-5) 精度回帰テスト (baseline vs P2) 実行

### 3.5 最適化 Phase 3 (高度 / 条件付き)

- [ ] (P3-1) HNSW パラメータチューニング (M, ef_construction, ef_search) 最小化
- [ ] (P3-2) キャッシュ: 直近ノード間距離 (LRU) で再計算抑制
- [ ] (P3-3) wiring バッチ化 (複数 step 分の近傍を先読み)
- [ ] (P3-4) geDIG 指標の差分更新 (density など再計算除去)
- [ ] (P3-5) プロファイラ (pyinstrument or cProfile) ショートラン結果保存

### 3.6 検証 & ガードレール

- [ ] (V1) 小迷路 (10x10) で全 Phase 適用後: 結果差分 0 (成功率/coverage)
- [ ] (V2) 25x25 で成長曲線オーバーレイ (baseline vs P1 vs P2 vs P3)
- [ ] (V3) 50x50 で step_time_ms 減少率 >= 40% (目標) を達成
- [ ] (V4) 50x50 で coverage / success 低下 2% 未満確認
- [ ] (V5) 再現性: 同 seed 3 回の分散許容範囲設定 (±5% novelty_ratio)

### 3.7 結果整理 / ドキュメント

- [ ] (D1) 結果サマリ Markdown: before/after 指標表
- [ ] (D2) グラフ: step vs time, step vs nodes, novelty vs steps
- [ ] (D3) 設定差分表 (ann-threshold, k, triggers)
- [ ] (D4) リグレッションテスト手順 README 追記
- [ ] (D5) 主要パラメータを config に外出し (yaml)

### 3.8 リスク & フォールバック

- [ ] (R1) ANN 早期化により初期過学習(偏り)が出ないか統計確認
- [ ] (R2) sampling によりレア経路喪失がないか coverage 差分解析
- [ ] (R3) geDIG 間引きで重要シグナル遅延しないか (stagnation 検知遅延)

### 3.9 RAG インテグレーション準備 (再利用抽象)

- [ ] (RAG-1) AdaptiveMetricTrigger 抽象クラス化 (growth/stagnation/cost) → `src/insightspike/adaptive/trigger.py`
- [ ] (RAG-2) IncrementalStats (coverage/novelty/revisit 汎用) → `src/insightspike/metrics/incremental.py`
- [ ] (RAG-3) VectorIndex アダプタに `batch_add`, `adaptive_tune(params)` フック追加
- [ ] (RAG-4) EmbeddingCache + AsyncEmbeddingQueue スケルトン (LRU + 背景生成)
- [ ] (RAG-5) Rerank Early-Exit ハーネス (閾値打ち切り) 試験
- [ ] (RAG-6) Hybrid Retrieval インターフェイス (ANN + keyword) Draft
- [ ] (RAG-7) Retrieval ベンチ (QPS / p95 / recall@k) スクリプト追加
- [ ] (RAG-8) 迷路最適化コンポーネントを RAG で利用する PoC 配線図作成

---

## 4. 受け入れ基準 (Acceptance Criteria)

- Phase 1 適用後: 精度指標 (success/coverage/novelty) ±0.5% 以内
- Phase 2 適用後: 平均 step_time_ms 30% 以上短縮 + 精度低下 <2%
- Phase 3 適用後: 平均 step_time_ms さらに 15% 以上追加短縮 or メモリ削減 20% 以上
- 50x50 完走時間 (同 seed) が baseline 比 50% 近似で短縮 (Stretch Goal)

---

## 5. 計測テンプレート

```text
RUN_ID | scale | phase | steps | success | coverage | novelty_ratio | revisit_ratio | avg_step_ms | p95_step_ms | nodes | edges | ann_init_step | k_candidates | gedig_calls
```

---

## 6. ログ抽出コマンド例 (参考)

(必要に応じて README へ再配置)

```text
# growth 行抽出
grep 'GRAPH_GROWTH' logs/maze_sweep_41.log | head

# step time 統計 (導入後)
grep 'STEP_TIME' logs/maze_sweep_41.log | awk '{print $NF}' | Rscript -e 'x<-scan(file="stdin",quiet=TRUE);cat(mean(x),quantile(x,0.95),"\n")'
```

---

## 7. メモ / 次回更新予定

- ANN 発火ログ未確認 → (B1) 後にチェック
- diameter 計算頻度 現状: 条件的 (ノード閾値?). P2 で完全オプション化予定
- スナップショットコピー: メモリ使用量計測まだ

---

## 8. 更新履歴

| 日付 | 変更者 | 変更概要 |
|------|--------|----------|
| 2025-09-02 | 初版 | 文書作成 |
| 2025-09-02 | 追記 | RAG インテグレーション準備 (3.9) / セクション9 追加 |

---

（以後、完了したら [ ] → [x] に更新しコミット）

---

## 9. RAG 連携観点メモ

| Maze 最適化要素 | RAG 対応先 | 期待効果 | 補足 |
|------------------|------------|----------|------|
| Snapshot gating / exponential backoff | インデックス統計再集計間引き | 再集計コスト削減 | コーパス静的時は長期停止 |
| geDIG trigger (growth/stagnation) | 動的関連スコア/ランキング再計算トリガ | 不要 rerank 抑制 | growth=新チャンク率 |
| ANN 早期初期化 + adaptive param | Retrieval ANN ef_search 動的制御 | レイテンシ安定 | コスト/recall トレード最適化 |
| Candidate sampling (top-k + random) | 再ランキング前プリフィルタ | rerank モデル呼び出し削減 | ランダムで recall 補完 |
| IncrementalStats | クエリ成功率 / doc hit coverage 追跡 | O(1) 更新 | 指標可観測性向上 |
| Landmark diameter 近似 | 連結性/クラスタ拡散度 監視 | 異常検知 | オフライン解析用 |
| Cost model (EWMA) | 適応的パラメータ (k, ef_search) 調整 | 自動チューニング | SLA 満たす範囲で最大 recall |
| Dirty-set 差分更新 | インデックス内メタ特性更新 | スループット向上 | 更新頻度高い場合有効 |

RAG 追加課題(別管理): Embedding 生成遅延, 再ランキング early-exit, Chunking policy, Hybrid query rewrite.
