# 迷路実験（Maze PoC）— 統合実験設計・成功基準

最終更新: 2025-10-31

本仕様は `geDIG_onegauge_improved_v3.tex` の PoC「部分観測迷路」節（docs/paper/geDIG_onegauge_improved_v3.tex:1009 以降）の実験設計を一枚に統合し、目標と指標のブレを解消するための参照票です。

---

## 1. 目的（Primary / Secondary）

- Primary（未知環境の探索効率）
  - 未知迷路で「無駄な右往左往をしない」ことを定量化し、探索効率を最大化する。
  - 行き詰まりの即時検出（AG）と、最近傍未探索分岐への効率的復帰（DG）を示す。

- Secondary（参照上限への近接）
  - 既知地図の参照値（Dijkstra/A*）に対して、geDIG の「余分ステップ（Regret）」と SPL が小さいことを示す。
  - 目的関数は異なるため「完全一致」ではなく「整合度」で評価（順位相関・経路一致度）。

---

## 2. 環境・条件

- 迷路サイズ: 15×15, 25×25, 50×50
- シード: 各 100
- 観測: 部分観測（現在位置＋上下左右 1 マス）
- 行動選択: anti‑backtrack ON, hop>0 は DG 判定、AG は g0 分位（q=0.9）で動的
- リソース: equal‑resources（同じシード・同じ迷路・同じ予算で比較）

---

## 3. 比較手法（役割の明確化）

- 参照（上限）: Dijkstra（地図既知, 重み=1）, A*（Manhattan, 地図既知）
- ベースライン: Random Walk, DFS‑inspired（地図未知, 部分観測）
- 提案: geDIG（地図未知, 部分観測, F=ΔEPC−λ·ΔIG, AG/DG 二段ゲート）

---

## 4. 指標セット（定義と向き）

Primary（探索効率）
- 探索率 exploration_ratio = unique_visited / total_cells（↓良）
- 訪問重複 avg_visits_per_cell = total_steps / unique_visited（↓良）
- バックトラック効率 avg_backtrack_length（AG→DG 間の長さ, ↓良）
- デッドエンド検出遅延 deadend_detection_delay（deadend→AG, ↓良）
- 成功率 success_rate（↑良）

Secondary（参照近接）
- Regret = steps − L*（↓良）
- SPL = L* / max(L*, steps)（↑良）

運用・診断（論文付記）
- P50/P95 time_ms_eval（↓良）
- Gate 統計: AG/DG 発火回数・発火率・AG:DG 比（適正域の確認）
- Frontier 順位相関（Spearman ρ, geDIG の −F vs Dijkstra 優先度）（↑良）
- 経路一致度（Jaccard of nodes/edges, geDIG vs Dijkstra）（↑良）

---

## 5. 成功基準（控えめだが強い）

規模別に次を満たすこと（中央値または平均）:

- 15/25: success_rate ≥ 95%, Regret 中央 ≤ +3, SPL ≥ 0.90
- 25: exploration_ratio ≤ 0.40, avg_visits_per_cell ≤ 1.5,
      avg_backtrack_length ≤ 5, deadend_detection_delay ≤ 1
- 50: success_rate ≥ 95%, SPL ≥ 0.88（若干緩和）, exploration_ratio ≤ 0.45
- 既知地図解析: Frontier順位相関 ρ ≥ 0.80（時間とともに上昇する段階相関も報告）

注: 参照（Dijkstra/A*）は理想上限の比較枠。未知迷路に直接適用しない。

---

## 6. 解析計画（統計/可視化）

- 統計: Welch t‑test（Bonferroni 補正）, 効果量 Cohen’s d, 95% CI
- 可視化:
  - Regret 箱ひげ（15/25/50）
  - 訪問頻度ヒートマップ（Random/DFS/geDIG 並列）
  - バックトラック軌跡（AG→DG を赤強調, 復帰先を星）
  - Frontier 優先度散布（整合度 ρ を提示）

---

## 7. ログと集計（実装インタフェース）

- steps.json（1行1ステップ）に保持する主キー:
  - position, possible_moves, ag_fire, dg_fire, is_dead_end（補助）
- summary.json の追加セクション（Builder で補完可能）:
  - exploration_efficiency: { exploration_ratio, avg_visits_per_cell, unique_visited, total_cells }
  - backtrack_analysis: { n_backtracks, avg_backtrack_length, deadend_detection_delay, total_backtrack_steps }
  - gate_statistics: { ag_fire_count, dg_fire_count, ag_fire_rate, dg_fire_rate, ag_dg_ratio }

実装: `experiments/maze-query-hub-prototype/qhlib/metrics.py`（本リポジトリに追加済）
Builder 反映: `experiments/maze-query-hub-prototype/build_reports.py` が summary に自動付与。

---

## 8. 論文への反映（最小パッチ案）

- PoC の「実験設計」直下に「評価指標と成功基準（Maze）」の小節を追加し、上記 4–6 を簡潔に列挙。
- Dijkstra/A* は「参照上限」である旨を明記し、Primary は探索効率であると宣言。
- メイン本文は要点と代表値のみ、詳細統計は付録へ移管。

---

## 9. よくある誤解への備え（記述テンプレ）

- 「最短路を求めるのか？」→ いいえ。未知迷路での探索効率（無駄歩き最小化）が主目的。最短路は参照枠。
- 「地図なしで Dijkstra を超えるのか？」→ 比較対象ではなく上限の参照。整合度と Regret/SPL で近接を示す。

