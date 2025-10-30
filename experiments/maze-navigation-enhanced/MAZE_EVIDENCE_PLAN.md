# Maze Evidence 強化計画

作成日: 2025-08-29  
更新方針: 各タスク完了時にチェックを付与し、成果物 / コミット ID / 出力パスを追記。

---

## 進捗チェックリスト (整理済み)

Legend: [x] 完了 / [ ] 未 / [~] 進行中

1. 指標整備 / 正規化
   - [x] metrics_utils.py 作成 (冗長度派生値・カバレッジ・バックトラック率計算) 2025-08-29
   - [x] baseline_vs_simple_plot 改修: 指標拡張 (clipped, unique_coverage, backtrack_rate, mean_geDIG, geDIG_low_frac) 2025-08-29
   - [x] 既存他スクリプト (ablation, scaling) へ指標統合 2025-08-29
2. 統計強化 (n≥30 / 効果量 / CI)
   - [x] 共通関数 effect_size_and_ci (Cohen's d, bootstrap CI) 2025-08-29
   - [x] baseline 比較に統合 (stats.json, effect_sizes.md) 2025-08-29
   - [x] ablation (normal vs random vs zero) 統計付与 2025-08-29
3. AUC 統合 (phase ラベリング簡易版)
   - [x] 簡易 PhaseAnnotator (branch_entry / deepening / terminal / retreat) 2025-08-29
   - [x] geDIG / -ΔgeDIG / linear_combo AUC 算出 2025-08-29
   - [x] baseline_vs_simple_plot に AUC 出力 (auc_summary.md) 2025-08-29
4. サイズスケーリング真正化
   - [ ] size 対応 maze generator (generate_maze(width,height,branch_density,seed))
   - [ ] size_scaling_experiment: width×height×density グリッド実験
   - [ ] スケーリング図 (line + heatmap) 生成
5. geDIG Leave-One-Out アブレーション
   - [ ] LOO 重みセット生成 (特徴 i を 0)
   - [ ] 追加実行 & normal との差分効果量表 loo_stats.json
   - [ ] feature_contribution.md 出力
6. 外れ値診断
   - [ ] P95 算出 & 超過ケース抽出
   - [ ] outliers.json + outliers.md (要約 + 代表例)
7. 再現パイプライン化
   - [ ] Makefile ターゲット maze_evidence 追加 (依存順に実行)
   - [ ] scripts/collect_maze_evidence.py (最新結果統合 index.json)
8. 総合レポート生成
   - [ ] evidence_report.md (全図 / 統計 / AUC / アブレーション / スケーリング / 変動性 / 外れ値)
9. 品質ゲート / テスト
   - [ ] metrics_utils ユニットテスト (冗長度, effect size, AUC) tests/metrics_utils_test.py
   - [ ] smoke run (小規模 seed) 成功
   - [ ] フル run (n=30) 実行時間 & 成果物確認

---
\n## 指標仕様

| 指標 | 定義 | 目的 |
|------|------|------|
| loop_redundancy | path_len / loop_erased_len | 冗長探索評価 (高=低効率) |
| clipped_redundancy | path_len / max(loop_erased_len, k) (k=5) | 小さすぎる loop_erased_len による暴騰緩和 |
| winsorized_redundancy | loop_redundancy を [P1,P99] でクリップ | 外れ値耐性・比較安定化 |
| normalized_over_budget | (path_len - loop_erased_len)/path_len | 冗長部分割合 (0~1) |
| unique_coverage | unique_positions / path_len | 新規探索比率 |
| backtrack_rate | backtrack_steps / path_len | 戻り多寡 |
| mean_geDIG | 平均 geDIG 値 | 構造進展平均 |
| geDIG_low_frac | geDIG < threshold 比率 | 低進展状態滞留度 |

---
\n## 統計計算仕様

- Cohen's d: (μ1 - μ2) / spooled
- Bootstrap 10,000 サンプル (パーセンタイル 2.5/97.5)
- 効果量解釈目安: 0.2=small, 0.5=med, 0.8=large

---
\n## AUC フロー (簡易 Phase)

1. 各ステップでセル近傍分岐度 degree を算出
2. degree==1 かつ 末端: terminal
3. 直前座標への戻り: retreat
4. degree>=3 かつ 未訪問分岐≥2: branch_entry
5. 他: deepening

分類ターゲット: terminal (正例) vs 非 terminal  
スコア候補: geDIG, -ΔgeDIG, a*geDIG + b*(-ΔgeDIG) (a,b∈{0,0.25,0.5,0.75,1})

---
\n## スケーリング設計

- 生成: DFS spanning tree 基盤 → エッジ追加/除去で分岐密度制御
- サイズグリッド例: (15x15,25x25,35x35)
- branch_density: 0.05 / 0.1 / 0.2
- 出力: redundancy_lines.png, redundancy_heatmap.png

---
\n## Leave-One-Out 寄与評価

- Δ冗長度 (normal vs LOO_i)
- effect size d(normal, LOO_i)
- contribution_score = d(normal, zero) - d(LOO_i, zero)

---
\n## 外れ値診断

- 算出: clipped_redundancy の P95 (アルゴ別)
- 抽出: seed, algo, path_len, loop_erased_len, clipped_redundancy, backtrack_rate, mean_geDIG, geDIG_low_frac, first_backtrack_step
- 上位: clipped_redundancy 降順トップ5

---
\n## パイプライン (maze_evidence)

1. baseline_vs_simple_plot (n=30) → baseline/records.json, redundancy_boxplot.png, auc_summary.md
2. ablation_geDIG (normal/random/zero + LOO) → `ablation/*.json`
3. size_scaling_experiment (size×density) → `scaling/*.json`, scaling図
4. deadend_heuristic_probe → `deadend_probe/*.md`, `deadend_probe/*.png`
5. deadend_multiseed_variability → variability.json
6. outlier_detection → outliers.json, outliers.md
7. collect_maze_evidence.py → index.json, evidence_report.md

---
\n## アウトプット構造 (提案)

```text
results/maze_evidence/
  YYYYMMDD_HHMMSS/
    baseline/
    ablation/
    scaling/
    probe/
    variability/
    outliers/
    evidence_report.md
    index.json
```

---
\n## テスト計画

- metrics_utils: pytest 最小 3 ケース
  - 冗長度: path=[A,B,A] → loop_erased=[A,B]; redundancy=1.5
  - backtrack: 明示 2 回で率検証
  - AUC: 完全分離スコアで 1.0

---
\n## リソース/所要時間 (概算)

- 実装 (指標+統計+AUC): ~2h
- スケーリング & LOO: ~1.5h
- フル 30 seeds 実行: 数分〜 (サイズ依存)
- レポート生成: ~0.5h

---
\n## 変更履歴ログ

- [ ] 2025-08-29: 計画初版作成
- [x] 2025-08-29: 指標統合完了 + AUC/phase 実装反映

Archive Notice: 未完了タスク (サイズスケーリング, LOO, 外れ値診断, パイプライン化, レポート, テスト) は統合計画 TEST_FAILURE_REMEDIATION_PLAN_2025_09.md へ移管予定。本ドキュメントはアーカイブ対象。

---
\n## 備考

- smoke (n=5) → フル n=30 の順
- 乱数固定: numpy.random.seed + random.Random(seed)
- index.json にパラメータハッシュを保存し再現性担保
