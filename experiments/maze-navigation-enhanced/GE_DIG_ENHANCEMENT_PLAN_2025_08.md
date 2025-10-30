---
title: Maze Navigation geDIG 強化計画 (Phase2/3)
status: proposal
decision_due: 2025-09-15
next_step: "Approve scope → implement Phase 2 incremental tasks"
created: 2025-08-25
owner: maze-nav
tags: [maze, gedig, exploration, experiment]
---

<!-- markdownlint-disable MD041 -->

## geDIG 強化 & 迷路ナビゲーション最適化 計画 (2025-08)

## 目的 (Why)

探索型迷路ナビゲーション実験において geDIG (Graph Edit Distance - Information Gain) を単なるヒューリスティック指標から、以下 4 役割を担う統合メトリクスとして再定義し、再現性あるベンチマークで既存戦略 (simple wiring 等) に対する優位性を提示できる状態へ引き上げる。

1. 探索進捗評価  
1. dead-end / shortcut (短絡) 検出  
1. エッジ配線制御 (疎密最適化)  
1. バックトラックトリガ

上記を一貫性ある数式とログ構造で裏付ける。

## 成功基準 (Definition of Done)

| 区分 | 指標 | 目標値 (Phase 2) | 目標値 (Phase 3) |
|------|------|------------------|------------------|
| 成功率 | 25x25 迷路達成率 (30 seeds) | ≥ 90% | ≥ 95% |
| 効率 | 平均 (steps / 最短路長) | ≤ 2.0 | ≤ 1.8 |
| dead-end 検出 | backtrack precision | ≥ 0.80 | ≥ 0.88 |
| dead-end 検出 | backtrack recall | ≥ 0.70 | ≥ 0.80 |
| グラフ効率 | エッジ数(simple比) | -10% 以上 | -25% 以上 |
| 安定性 | geDIG 負スパイク誤検知率 | ≤ 15% | ≤ 10% |
| パフォーマンス | 1 step 処理時間 (25x25) | < 40ms | < 30ms |
| 再現性 | config + seed 追跡率 | 100% | 100% |

## 現状ギャップ

- geDIG 実装: ノード/エッジ増分 + 密度変化のみ (GED/IG の分離なし)
- wiring_strategy: `simple` のみ / `gedig` 未実装
- dead-end 評価: ground truth / precision 指標未整備
- 重み/閾値調整: 手動試行止まり (自動 sweep なし)
- ログ: 画像 + サマリ Markdown のみ (解析しやすい JSONL 不足)
- 再現性: seed 管理 / config snapshot 不統一
- スケール: pruning / memory 制御未定義

## アーキ拡張方針

```text
[Episode Layer] --(vector)-> [GraphManager] --(structural delta)-> [GeDIGEvaluator]
      |                                                |            
      +--(visit stats)--> [DecisionEngine] <----------- +--(gedig score streams)
                                          |            
                               [Backtrack Controller]
```

## geDIG 仕様方針（Phase 2 固定 / Phase 3 探索）

Phase 2 では「現行メインコードのシンプルな GED - k*IG（実装上はノード/エッジ増分 + 密度系ヒューリスティックに相当）を変更しない」ことを明示。以下の再定義案は Phase 3 以降の探索的オプションであり、行動決定や wiring のスコア計算コアを差し替えない。

### Phase 2 固定事項
 
1. geDIG コア式: 現実装そのまま（線形/減算構造）
2. Retrieval パイプライン: 二段階 (a) 距離類似度トップ候補抽出 → (b) geDIG による re-rank（線形結合は使用しても良いが“距離フィルタ後”）
3. 重みベクトル: 手動固定（学習/更新なし）
4. Pruning / "睡眠" 相当の統合フェーズ: 未実装（Phase 3 以降）
5. 次元動的拡張（ベクトル圧縮/追加）: 検討保留（ブレ回避）

### Phase 3 探索案（参考 / 実装前提ではない）
 
```text
geDIG = structural_gain - k * consolidation_penalty

structural_gain:
  α * new_node_ratio + β * new_edge_ratio + γ * component_reduction_gain
consolidation_penalty:
  δ * densification_spike + ε * redundant_edge_ratio + ζ * shortcut_intensity
```

#### 用語 (Phase 3 再考メモ)

- shortcut_intensity: (エッジ急増 / 既存エッジ) 正規化
- densification_spike: density(t) - EMA_density(t)
- new_node_ratio: added_nodes / (prev_nodes + 1)
- 係数 (α..ζ) 初期値: 0.6 / 0.3 / 0.1 / 0.5 / 0.3 / 0.2 （後で sweep）

### ログ戦略（Phase 2）
 
- コア geDIG 値は従来計算をそのまま保存
- 追加で "参考列" として簡易 gain/penalty 分解を“副次的”に出力する場合でも行動スコアリングに未使用であることを記録（混同防止）
- predictive geDIG（前向き推定）はインタフェース予約（未計算は None）とし本線ロジック非依存

### 二段階 Retrieval の最小仕様
 
1. 距離ランキング上位 M（例: 4〜8）方向のみ残す
2. その集合内で geDIG（または負値リスク回避指標）で優先度再計算
3. タイブレークに距離最小を採用
4. 記録: {candidate_id, distance_rank, distance_norm, gedig_value, rerank_score, chosen_flag}

この方針により Phase 2 では “構造的予測の難しい前向き補正” を避け、ベースライン＋再ランク導入効果を定量化してから Phase 3 の再定義に進む。

## フェーズ分割

### Phase 2 (実装重視 / 速い検証)

1. Logging 基盤: `ExperimentRecorder` (JSONL per step) + summary JSON  
2. wiring dispatch + `wire_with_gedig` (最初は単純閾値)  
3. geDIG 分解 (structural_gain vs consolidation_penalty) 出力のみ (まだ係数固定)  
4. dead-end ground truth 生成 (BFS で degree=1 内部ノード集合) + backtrack precision/recall ログ  
5. バッチ runner + parameter sweep (threshold × backtrack_threshold × strategy)  
6. 結果集計 (CSV + Pareto front 抽出)

### Phase 3 (最適化 / 理論化)

1. 動的閾値調整 (EWMA + z-score 管理図) → adaptive wiring  
2. k および係数 (α..ζ) の自動最適化 (grid / Bayesian sweep)  
3. pruning 戦略 (低 impact node, 再訪率高ノードの圧縮)  
4. backtrack classifier (単純閾値 → 2D decision: geDIG, densification_spike)  
5. 完全型近似 geDIG (C(20,3) 近似) の低コスト estimator を research ブランチで検証  
6. baseline 実装 (DFS, random-walk, curiosity-like intrinsic) + 有意差検定  
7. ドキュメント/論文化: 数式, 計算量解析, 比較表

## タスク詳細 (Phase 2)

| ID | Task | 出力 | 優先 | 所要 | Done条件 | 状態 |
|----|------|------|------|------|----------|------|
| P2-1 | ExperimentRecorder 実装 | recorder.py | High | 0.5d | JSONL/summary 生成 | [ ] |
| P2-2 | wiring dispatch + gedig 戦略 stub | graph_manager.py | High | 0.5d | strategy='gedig' 動作 | [ ] |
| P2-3 | geDIG 分解 出力拡張 | gedig_evaluator.py | High | 0.5d | 値ログ (gain/penalty) | [ ] |
| P2-4 | dead-end ground truth | maze_utils.py | High | 0.5d | 集合生成 + 指標算出 | [ ] |
| P2-5 | backtrack metrics (precision/recall) | recorder | High | 0.5d | summary JSON に反映 | [ ] |
| P2-6 | batch runner + sweep | scripts/run_maze_batch.py | High | 0.5d | CSV 出力 | [ ] |
| P2-7 | Pareto 抽出/推奨設定 | scripts/analyze_sweep.py | Med | 0.5d | best_config.json | [ ] |
| P2-8 | README への結果反映 | README.md | Med | 0.25d | セクション更新 | [ ] |

### Phase 2 Checklist

Legend: [x] 完了 / [ ] 未 / [~] 進行中

- [ ] P2-1 ExperimentRecorder
- [ ] P2-2 wiring dispatch + gedig 戦略 stub
- [ ] P2-3 geDIG 分解ログ
- [ ] P2-4 dead-end ground truth 生成
- [ ] P2-5 backtrack precision/recall 計測
- [ ] P2-6 batch runner + sweep
- [ ] P2-7 Pareto 抽出 / 推奨設定
- [ ] P2-8 README 結果反映

## 依存関係

- P2-3 は P2-1 前に変更可 (ログフォーマット確定前に core 単体テスト可)  
- P2-5 は P2-4 完了後  
- P2-7 は sweep 終了 (P2-6) 後

## 設定パラメータ (初期 sweep 候補)

| Param | 候補 | 備考 |
|-------|------|------|
| gedig_threshold | 0.1, 0.2, 0.3, 0.4 | wiring D 用 |
| backtrack_threshold | -0.05, -0.1, -0.15, -0.2 | 負スパイク感度 |
| temperature | 0.05, 0.1, 0.2 | 探索多様性 |
| k (Phase2固定) | 0.5 | 分解後さらに再調整予定 |

## ログ仕様 (JSONL per step)

```json
{
  "step": 12,
  "pos": [5,9],
  "action": "E",
  "gedig": 0.14,
  "gain": 0.18,
  "penalty": 0.04,
  "shortcut": false,
  "backtrack_trigger": false,
  "nodes": 120,
  "edges": 210,
  "unvisited": 43,
  "strategy": "gedig"
}
```

Summary JSON (抜粋): `steps, success, ratio_steps_shortest, backtrack_precision, backtrack_recall, edges_total, density_p95, negative_spike_rate`。

## リスク & 対策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| geDIG 計算コスト上昇 | スループット低下 | 差分計算 + キャッシュ (ノード/エッジ統計増分更新) |
| 閾値過学習 | 汎化性能低下 | 複数迷路生成パラメータで cross-set sweep |
| JSONL 増大 | ディスク肥大 | 圧縮 (gzip) オプション / 要約後削除 |
| shortcut 誤検知 | 無駄 backtrack | densification_spike + edge_ratio の複合条件化 |

## メトリクス算出式 (ドラフト)

```text
new_node_ratio = added_nodes / (prev_nodes + 1)
new_edge_ratio = added_edges / (prev_edges + 1)
redundant_edge_ratio = (added_edges - added_nodes) / max(1, added_edges)
densification_spike = density_now - ema_density
shortcut_intensity = max(0, (added_edges / (prev_edges+1)) - 0.5)
structural_gain = 0.6*new_node_ratio + 0.3*new_edge_ratio + 0.1*component_reduction_gain
consolidation_penalty = 0.5*densification_spike + 0.3*redundant_edge_ratio + 0.2*shortcut_intensity
geDIG = structural_gain - k * consolidation_penalty
```

## 承認要求事項

- 上記 Phase 2 スコープで進めてよいか  
- 係数初期値 (α..ζ) と sweep 範囲 (±50%) で妥当か  
- dead-end ground truth を BFS-based degree=1 内部セルと定義で合意できるか

## 次アクション (承認後 1 日目)

1. `ExperimentRecorder` 骨組み + JSONL 出力  
2. `wire_edges` dispatch + `gedig` stub 戦略追加  
3. geDIG 分解 (gain/penalty) ログのみ追加 (ロジック簡易版)

---

## GNN 統合ショートノート (2025-09-06 追加)

ローカル(macOS)は `INSIGHTSPIKE_DISABLE_GNN=1` により非GNNパス。Linux CI で PyG 拡張 (scatter/sparse/cluster/spline-conv) を取得し最小 GCN (2層) で multi-hop コンテキストを行動スコア融合する段階導入。

最小スコープ: G0-1..G0-4 (CI, skip marker, GraphBuilder, forward-only GCN)。

指標トラッキング: multi_hop_gain, redundant_branch_rate, backtrack_count, fusion_usage_ratio。

README の "GNN 統合計画メモ" セクション参照。安定後 Phase G1 で学習 / 自動重み調整へ拡張予定。

---
Archive Notice: Phase 2 タスク (P2-1..P2-8) は未着手のまま新統合計画 (TEST_FAILURE_REMEDIATION_PLAN_2025_09.md) の M2/M3 前後で再優先付け予定。本計画は参照専用としてアーカイブ移動。

---
Generated: 2025-08-25
