# Minimal geDIG Experiment

目的: geDIG (Graph Exploration Differential Insight Gain) の Layer3 (局所探索価値: 未訪問セル + その未訪問近傍数) 単体効果を、Norm 距離フィルタ (Layer1) や Frontier 近似 (Layer2)、Adaptive (Layer4) を排した最小構成で定量化し Random に対する改善幅 (成功率 / 探索効率 / 経路効率) を測る。

## 方針

- 既存 Maze 環境・移動/観測ループを再利用
- ベクトル表現は 8 次元固定 (再定義): `(agent_x, agent_y, goal_x, goal_y, cand_x, cand_y, passable_flag, visited_flag)`
  - `agent_x, agent_y`: 現在位置 (幅/高さで 0-1 正規化)
  - `goal_x, goal_y`: 絶対ゴール座標を同様に正規化 (相対差分ではなく生座標を保持)
  - `cand_x, cand_y`: 候補移動先セル座標 (壁含む) を正規化
  - `passable_flag`: 候補セルが通路(1.0) / 壁(0.0) — 前回実験での「移動結果」要素を事前特徴として持つ
  - `visited_flag`: 候補セルが既訪問なら1.0, 未訪問0.0
- 旧仕様の `manhattan_to_goal` は最短距離ヒント=チートのため排除
- bias 項目は削除 (最小次元維持のため) — 必要になれば後で追加
- geDIG値 (初期 R1): "その候補を実際に進んだ場合に『既訪問でない到達セル + その到達セルの未訪問近傍数』がどれだけ増えるか" を近似 (探索価値)。距離短縮直接利用はしない。
- 選択戦略: 4 方向の geDIG を計算し最大を選択。全て <=0 の場合ランダムフォールバック。

## レイヤ構成と本実験スコープ

| Layer | 目的 | R1 での扱い |
|-------|------|-------------|
| Layer1: Norm Search / Embedding 距離フィルタ | 類似状態抑制・状態空間スパース化 | 未導入 (効果分離のため除外) |
| Layer2: Frontier Approx (2-step など) | 未訪問境界量の近似 | 未導入 |
| Layer3: geDIG Local Gain | 未訪問セル + 未訪問近傍増分 | 本実験主対象 |
| Layer4: Adaptive (stagnation / spike / collapse) | 動的モード切替 | 未導入 |

本 R1 では Layer3 単独寄与を計測後、R2 以降で Layer1/2 を追加し限界効用を評価予定。

## 比較条件 (ポリシー)

| ポリシー | 内容 |
|----------|------|
| random | 通行可能候補から一様ランダム (全 -1 の場合は全方向から) |
| gedig_simple | geDIG 最大 (同値はランダム) / 全部 -1 → random fallback |
| threshold_{1,2,3} | geDIG >= t (整数 1..3) の最大。同集合空なら random fallback |

## 測定指標

- success_rate: ゴール到達率 (N試行)
- mean_steps: 到達試行の平均ステップ数
- exploration_ratio: `unique_cells / steps`
- path_efficiency: `最短距離 / 実際距離`
- gedig_auc: geDIG時系列 (0開始積算をステップ正規化)

## 実験計画 (R1)

1. 迷路サイズ: 8x8 / 12x12 の 2 種 × シード 10 (0..9)
2. 各ポリシー 1 Episode (合計 100 run) ※ 後で episodes-per-seed を増やす場合あり
3. 閾値 sweep: t = 1,2,3 (geDIG ドメイン 0..5 の整数) — t=0 は simple と等価のため除外
4. 可視化: success_rate, mean_steps, path_efficiency, gedig_auc, threshold 曲線

## 判定基準

- gedig_simple が random より success_rate 向上 & mean_steps 短縮 (成功試行基準)
- threshold_t で t を上げると成功率が過小方向に U 字 or plateau を示す (過剰選別による探索範囲縮小)
- gedig_auc と path_efficiency に正の相関 (高い局所探索価値獲得が経路効率改善に寄与)

## 次段階 (R2+ 予定)

- Layer2: 簡易 frontier 近似 (2-step 到達未訪問セル数) を neighbor_gain へ加算
- Layer1: Norm 距離フィルタ (L2 距離閾値 or 近傍重複抑制) の軽量版導入
- R3: stagnation 単独 (停滞カウンタでランダム挿入 or ノルム緩和)
- R4: spike/decay の簡略適応 (高 geDIG 連鎖によるモード転換)

## フォルダ構成 (予定)

```text
experiments/gedig_minimal/
  README.md
  PLAN.md
  simple_navigator.py          # 最小 geDIG 選択ロジック
  run_batch.py                 # 一括実行 & JSON 出力
  analyze.ipynb (任意)         # 可視化
  results/                     # 出力(JSON, CSV, 図)
```

## 実装メモ

- 既存 MazeNavigator から: 経路保持 / ゴール判定 / step 増加を流用
- geDIG 計算は BFS 不使用: 近傍セル (上下左右) の未訪問数をその候補遷移後で推定
  - 推定式 (R1): `gedig = (1 - visited_flag) + new_unvisited_neighbor_count`
  - `new_unvisited_neighbor_count`: 候補セル近傍で未訪問 & 通行可能のセル数 (最大4)
- 壁セル (passable_flag=0) の場合は `gedig = -1` で除外 (フォールバック時のランダム対象からも除外)
- 拡張余地: Frontier 増分 / 情報ゲイン (R2), Stagnation 単独導入 (R3)

## ライセンス / 再現性

- 乱数種統一: `np.random.seed(seed); random.seed(seed)`
- 記録: 各試行 JSON に {seed, steps, success, gedig_series, path, unique_cells} を保存

---
短期ゴール: R1 (Layer3 単独) を 1〜2コミットで実走し結果サマリ (JSON + 図) を docs にリンクする。

## R1 結果サマリ (Layer3 単独)

実行条件: 迷路サイズ 8x8 / 12x12, seeds=0..9, policies={random, gedig_simple, threshold_1, threshold_2, threshold_3}, step_limit=500, 100 run.

主要指標 (全条件合算 by_policy):

| Policy | success_rate | mean_steps_success | path_eff_mean | exploration_ratio | gedig_auc_mean |
|--------|--------------|--------------------|---------------|-------------------|----------------|
| random | 0.50 | 258.00 | 0.119 | 0.125 | 0.080 |
| gedig_simple | 0.90 | 128.56 | 0.220 | 0.546 | 0.260 |
| threshold_1 | 0.90 | 128.56 | 0.220 | 0.546 | 0.260 |
| threshold_2 | 0.70 | 173.57 | 0.152 | 0.352 | 0.176 |
| threshold_3 | 0.45 | 220.89 | 0.098 | 0.134 | 0.080 |

所見:
- geDIG (threshold_1 / simple) は success_rate +40pt, 平均到達ステップ ~50% 短縮。
-
- threshold_1 は今回の分布で gedig_simple と同一行動 (geDIG>=1 が常に最良集合)。
- 閾値上昇 (2,3) で過選別 → 成功率・効率低下 (U 字傾向確認)。
- geDIG AUC 上昇が path_efficiency / exploration_ratio 改善と整合。

図: `results/figures/` 内
- success_rate.png
- mean_steps.png
- gedig_auc_vs_efficiency.png

再現 (R1 再走): `python run_batch.py --maze-sizes 8x8 12x12 --seeds 0 1 2 3 4 5 6 7 8 9 --policies random gedig_simple threshold_2 threshold_3 --include-threshold1 --step-limit 500 --out-dir results --progress`

デフォルトでは冗長性のため `threshold_1` を除外。検証時のみ `--include-threshold1` を付与。

次ステップ (R2 予定):
 
 - threshold_1 を除外 (冗長) し frontier 近似導入の差分測定。
 - geDIG 出力頻度/分布解析 (AUC ヒスト) で適切な adaptive 触媒指標を抽出。

---

## R2 追加 (Frontier 近似 / 2-step 拡張 概要)

目的: Layer2 (Frontier Approx) として「候補セルの 1-step 先」だけでなく「2-step 先で新規に開く未訪問セル集合」を探索価値へ加味し、局所閉塞回避と探索空間の広がりを早期に検知する。

定義:

```text
gedig_R2 = (1 - visited_flag) + n1 + α * n2_unique
  n1: 1-step 未訪問 & 通行可能セル数 (最大4)
  n2_unique: 2-step 到達セルのうち (候補セル自身 + 1-step 集合) を除いた未訪問ユニーク数 (概ね最大 ~8)
  α (frontier_weight): 0 < α ≤ 1 推奨 (初期 0.5)
```

正規化 (AUC 用):

```text
assumed_max_R1 = 5
assumed_max_R2 ≈ 1 + 4 + α * 8
gedig_auc = (Σ step gedig) / (steps * assumed_max_Rx)
```
注意: R1 と R2 の AUC は分母が異なるため直接比較不可。改善判定には (success_rate, mean_steps, exploration_ratio) を優先指標とする。

実装差分 (`simple_navigator.py`):

- CandidateFeature に `neighbor_unvisited_2`
- R2 ポリシー名: `gedig_r2`
- 2-step 計算は 1-step 集合を起点に 8 近傍をユニーク拡張 (O(方向数^2))

初期スモーク (8x8, seeds=0,1, step_limit=200, α=0.4):

| policy | success_rate | mean_steps_success | path_eff_mean | exploration_ratio | gedig_auc_mean |
|--------|--------------|--------------------|---------------|-------------------|----------------|
| gedig_simple | 1.0 | 71.0 | 0.212 | 0.677 | 0.326 |
| gedig_r2 (α=0.4) | 1.0 | 81.0 | 0.179 | 0.583 | 0.220 |

所見 (初期 α=0.4):

- 2-step 付加でやや遠回り (mean_steps 上昇) → 重み過大/未最適。
- exploration_ratio 低下は 2-step 見込みが局所枝選好を抑制し過剰フロンティア偏重になった可能性。
- AUC (R2 正規化) が低いのは足取り増加 + 分母拡張の双方要因。

チューニング計画:

1. α スイープ: 0.15, 0.25, 0.35, 0.45, 0.60
2. 指標最適化: `J = success_rate - λ*(mean_steps_success / median_random_steps)` (λ ≈ 0.3) の最大化を暫定目的関数に。
3. 追加ログ: R2 では `r2_contrib_1` (n1) と `r2_contrib_2` (n2_unique) の比率分布を箱ひげで可視化し α 調整へフィードバック。
4. 過剰探索検出: n2_unique が 0 の連続区間長を stagnation 簡易 proxy として R3 へ引き継ぐ。

実行例 (α スイープ):

```bash
for a in 0.15 0.25 0.35 0.45 0.60; do \
  python run_batch.py \
    --maze-sizes 8x8 12x12 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --policies gedig_simple gedig_r2 \
    --frontier-weight $a \
    --step-limit 500 \
    --out-dir results_r2_a${a/./_}; \
done
```

期待改善メカニズム (α 最適域想定 0.25±0.1):

- 初期層: R1 と同等 / 迷路開放度が高い局面のみ 2-step 利得で進路選別
- 後半層: 局所閉塞 (n1 低) でも n2_unique が差を提示し、遠回り前に“開ける方向”を早期提示

今後: α 最適化後 `gedig_r2` が mean_steps 低下 or success_rate 上昇 (特に 12x12) を示せば Layer2 有効性確定 → R3 へ (stagnation + minimal adaptive) 移行。

### α スイープ結果 (8x8 / 12x12 seeds=0..9, step_limit=500)

| α (frontier_weight) | gedig_r2 success_rate | gedig_r2 mean_steps_success | gedig_simple success_rate | 差分 (success) | 備考 |
|---------------------|-----------------------|-----------------------------|---------------------------|----------------|------|
| 0.15 | 0.85 | 118.24 | 0.90 | -0.05 | R2 が steps で優位 (短縮) / success わずかに低下 |
| 0.25 | 0.85 | 118.24 | 0.90 | -0.05 | 0.15 と同一挙動 (n2 影響まだ穏やか) |
| 0.35 | 0.85 | 118.24 | 0.90 | -0.05 | AUC 低下傾向開始 |
| 0.45 | 0.85 | 118.24 | 0.90 | -0.05 | 過重み移行境界 |
| 0.60 | 0.20 | 106.50 | 0.90 | -0.70 | 早期探索枯渇 / frontier 選好過剰 (探索壊れ) |

観察:
+
- α=0.15〜0.35 は成功率 0.85 で横ばい, R1 baseline (0.90) より 5pt 低い。
- しかし mean_steps_success は R1 (128.56) より ~10% 短縮 (≈118)。frontier 情報が経路長短縮に寄与する一方で僅かな失敗増。
- α=0.60 で成功率崩壊 (0.20) → 遠方 frontier 過剰バイアスで局所進行停滞 (n2=0 連鎖増大が推測)。
- gedig_auc_mean (R2 正規化) は α 上昇で緩やか低下 → ノイズ的 n2 探索分散が減り局所利得集中度が下がる兆候。

暫定結論:
+
-
- Layer2 (2-step frontier) は小さめ α=0.15〜0.25 に設定すると、成功率 -5pt とのトレードオフで平均到達ステップ ~10% 改善の可能性。
- 本実験目的 (成功率向上を最優先) の観点では R1 を default, R2 は `--frontier-weight 0.2` 付与時の高速化オプションとして位置付け。
- Adaptive (R3) で stagnation (n2_zero_fraction, longest_zero_streak) を検出し α を動的制御 (低→中) することで成功率低下を抑えつつステップ短縮を狙う設計が妥当。

次アクション (R3 準備):

1. zero-streak 指標に基づき (longest_zero_streak > κ) で一時的に α を減衰、成功安定後に徐々に回復するスケジューラ試作。
2. `summarize.py` へ n1_hist / n2_hist / zero-streak 統計の可視化追加。
3. 目的関数 J (成功率優先 λ=0.3) で α 自動選択ロジック試験。

## R3 (Adaptive α) サマリ

目的: n2_zero 連鎖 (frontier 枯渇) を検出し frontier 重み α を一時低下→回復させ成功率低下を抑えつつステップ短縮を狙う。

実装: `gedig_r2_adaptive_<αmin>_<αmax>` 形式ポリシー。初回は αmin=0.1, αmax=0.25。

最終チューニング:

- drop_threshold=3 (連続 n2=0 で半減)
- recover_threshold=8 (連続 n2>0 蓄積で +0.02 線形回復)
- メタ記録: adaptive_fw_avg / final, drop/raise イベント数

12x12 seeds=0..9 評価 (step_limit=500, frontier_weight base=0.2):

| Policy | success_rate | mean_steps_success | path_eff_mean | exploration_ratio | gedig_auc_mean |
|--------|--------------|--------------------|---------------|-------------------|----------------|
| gedig_simple | 0.80 | 189.50 | 0.215 | 0.475 | 0.222 |
| gedig_r2 (固定 α=0.2) | 0.70 | 194.29 | 0.204 | 0.453 | 0.245 |
| gedig_r2_adaptive_0.1_0.25 | 0.70 | 194.29 | 0.204 | 0.453 | 0.251 |

所見:

- 固定 R2 に対し adaptive は成功率/ステップ同一で AUC わずか上昇 (軽微な探索価値集中効果)。
- しかし `gedig_simple` が依然成功率で優位 ( +10pt )。R2(+adaptive) はこの設定では成功率改善に寄与せず。
- 適応発火 (drop / raise) は seeds 数拡張または αmax 拡大で差異が出る可能性。

ステータス: R3 Minimum Viable (実装 + ログ + 単一サイズ評価) 完了。成功率改善未達のため “探索高速化オプション” として限定採用。さらなる改善は以下に委譲：

1. αmax を 0.3~0.35 へ引き上げ適応でのみ高値許容
2. 成功未達 (ゴール距離残 / 残ステップ比) 条件で動的に recover_threshold 短縮
3. n2_zero_fraction が高い迷路パターンのみ adaptive 起動 (前処理判定)

現段階では R1 baseline を default, R2 を固定オプション, R3 adaptive は experimental フラグ扱いが妥当。


