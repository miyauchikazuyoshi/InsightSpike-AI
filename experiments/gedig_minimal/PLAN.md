# Minimal geDIG PLAN

## 目的
geDIG 理論を用いたエピソード記憶ベース探索エージェントの最小実装を構築し、適応層や高度なフロンティア評価を外した単純な geDIG (探索価値近似) がランダム方策に対してゴール到達率と探索効率をどの程度改善するかを検証する。

## スコープ境界
含む:
- 迷路生成/ロード (既存ユーティリティ流用可)
- 1 ステップ先 4 方向候補の単純評価
- geDIG (R1) = 未訪問セル到達 + そのセルの未訪問近傍カウント
- 閾値選択 + ランダムフォールバック
- 結果集計 JSON 出力 & 簡易統計
除外 (Phase A):
- BFS / A* / multi-step lookahead
- spike / stagnation / embedding health adaptive logic
- Frontier 構造解析 (R2 以降)
- 学習 (パラメータ適応)

## データモデル
EpisodeResult JSON (per run):
```jsonc
{
  "seed": 123,
  "maze_size": [8,8],
  "policy": "threshold_0.2", // or random, gedig_simple
  "success": true,
  "steps": 57,
  "path_efficiency": 0.78,
  "unique_cells": 45,
  "exploration_ratio": 0.789,
  "gedig_series": [2,1,3,...],
  "gedig_auc": 0.42
}
```
集計: 各条件 x maze_size で平均/分散を別 JSON へ。

## geDIG 定義 (R1)
```
if passable_flag == 0: gedig = -1
else:
  base = 1 - visited_flag            # 未訪問なら 1, 訪問済なら 0
  neighbor_gain = count(unvisited & passable neighbors of candidate)
  gedig = base + neighbor_gain        # 0〜5 の整数
```
正規化は行わない (閾値は整数ドメイン: 0..5)。

## 本ディレクトリ完結仕様 (Self-Contained Spec)

他の“実験ディレクトリ”に新規実装を複製せず、既存メインコード（コア/ナビゲーション系モジュール）がそのまま import 再利用できる形で、追加ファイル最小 & `python run_batch.py` 単独実行で完結する運用仕様を定義する（= 実験オーケストレーションは本ディレクトリ内で閉じるが、コア機能は積極利用する）。

#### 再利用ポリシー
- 迷路生成: 既存 `maze_navigation_enhanced` 系の `maze_generator` が import 可能ならそれを使用。不可時のみローカル簡易生成にフォールバック。
- ベクトル/距離/統計ユーティリティ: 過度な依存連鎖を避けるため “純粋関数” かつ副作用無しモジュールのみ直接 import。状態フルな複合コンポーネント (GraphManager 等) は使用しない。
- Episode / Navigator 大型クラス: 直接利用せず、最小ステップループを本ディレクトリで保持（将来差分評価しやすくするため）。
- 設定/定数: 共有 `config` が必要になった場合は read-only import のみ（ローカルにコピーしない）。
- 型定義: 既存 dataclass を参照するよりも、ここで定義した軽量 dataclass を使い外部には漏らさない（破壊的変更リスク低減）。

### 迷路表現

- 2D numpy 配列 (dtype=int8) で 0=通路, 1=壁。
- スタートは (0,0)、ゴールは (W-1,H-1) を基本とし、壁生成後に両セルを強制通路化。

### 迷路生成アルゴ (簡易版)

1. 全セル壁(1) 初期化
2. 深さ優先 (recursive backtracker) で偶数座標セルを掘る (クラシック迷路)
3. 掘る際は 2セル先を見る; 通路でなければ間の壁を 0 にして前進
4. サイズが偶数の場合は端1列/1行を通路で埋め合わせして到達性確保
5. 生成関数 `generate_maze(width, height, seed)` を本ディレクトリに実装

### 座標 & 訪問

- `visited: Set[Tuple[int,int]]` 初期 `{start}`
- 更新は通路へ成功移動時のみ追加
- ゴール判定: `pos == (W-1,H-1)`

### 特徴ベクトル (8次元)

順序: `(agent_x, agent_y, goal_x, goal_y, cand_x, cand_y, passable_flag, visited_flag)`

- 位置系は各々 `coord / (size-1)` で 0..1 正規化
- passable_flag: 通路=1.0 / 壁=0.0 (壁候補も -1 geDIG 用に特徴生成は可)
- visited_flag: 候補セルが既訪問なら1.0 それ以外0.0

### データ構造

```python
@dataclass
class CandidateFeature:
    pos: Tuple[int,int]
    passable: bool
    visited: bool
    vec: np.ndarray  # shape (8,)

@dataclass
class StepRecord:
    t: int
    agent_pos: Tuple[int,int]
    candidate_vecs: List[np.ndarray]      # 常に4要素 (方向順: U,D,L,R)
    gedig_values: List[int]               # len=4 (-1 含む)
    chosen_index: int                     # 0..3
    chosen_pos: Tuple[int,int]

@dataclass
class EpisodeResult:
    seed: int
    maze_size: Tuple[int,int]
    policy: str
    success: bool
    steps: int
    unique_cells: int
    path_efficiency: float
    exploration_ratio: float
    gedig_series: List[int]
    gedig_auc: float
    step_limit: int
```

### ステップ処理フロー

1. 現在位置 `p` 取得
2. 4方向 Δ 一覧 `[(0,-1),(0,1),(-1,0),(1,0)]` を順序固定で処理
3. 各方向で候補座標 c を算出 (境界外 -> passable=False, 壁扱い)
4. `passable_flag`, `visited_flag` 算出 → 8次元ベクトル構築
5. geDIG 計算 (`passable ? (1-visited) + neighbor_unvisited_count : -1`)
6. ポリシーで chosen_index を決定
7. 移動: passable なら位置更新、壁なら位置不変 (ただし R1 では壁選択を原則起こさないポリシー設計)
8. visited 更新 (通路移動成功時)
9. 終了判定 or 次ステップへ

### geDIG AUC

`gedig_auc = sum(gedig_series) / (steps * 5)` で 0..1 正規化 (最大 geDIG=5 仮定)。

### パス効率 (path_efficiency)

`manhattan(start, goal) / steps_if_success` 成功時のみ計算。失敗時は 0。
マンハッタン距離が "暗黙的距離ヒント" になる点は README に注意書きを追記予定 (純粋探索比較のため計測のみで利用しない)。

### 依存ライブラリ

- numpy のみ (標準ライブラリ + dataclasses)
- プロット時のみ matplotlib (オプション) → core 実行は不要

### CLI インタフェース (run_batch.py 想定)

```
python run_batch.py \
  --maze-sizes 8x8 12x12 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --policies random gedig_simple threshold_1 threshold_2 threshold_3 \
  --step-limit 500 \
  --out-dir results
```
オプション:

- `--episodes-per-seed N` (デフォルト1)
- `--progress` (進捗バー表示)
- `--no-save-steps` (StepRecord を raw JSON へ含めない; 省サイズ)

### 出力 JSON 最小例

```jsonc
{ "seed":0, "maze_size":[8,8], "policy":"gedig_simple", "success":true,
  "steps":57, "unique_cells":45, "path_efficiency":0.78, "exploration_ratio":0.789,
  "gedig_series":[2,1,3,...], "gedig_auc":0.42, "step_limit":500 }
```

### 既存 Maze Navigator 実装との差分サマリ

| 項目 | 本最小版 | 既存拡張版 |
|------|----------|-----------|
| Episode 単位 | 候補セル (方向は暗黙) | (位置,方向) ペア |
| ベクトル次元 | agent/goal/cand + flags | 位置/方向/壁/visit/log など |
| geDIG | ローカル未訪問近傍増分 | GED - k*IG + 構造改善 |
| グラフ/配線 | なし | GraphManager / wiring_strategy |
| インデックス/ANN | なし | InMemoryIndex / HNSW |
| 適応 (stagnation/spike/embedding) | なし | 有効 (段階的) |
| メモリ削減/flush | 不要 | flush / eviction / rehydrate |

### 将来拡張フック (R2+ 用の軽量な型変更余地)

- CandidateFeature に `neighbor_unvisited_count` を格納 (計算再利用)
- StepRecord へ `random_fallback` bool 追加 (行動選択理由追跡)
- geDIG バージョン: EpisodeResult.meta (簡易 dict) を追加し `{"gedig_version":"R1_local"}` 記録

### 実装ステップ順序再確認

1. `generate_maze` 実装
2. data classes 定義 & feature builder
3. geDIG 計算 & policy 選択関数
4. `run_episode(policy, size, seed)` 実装
5. バッチ実行 & 保存ユーティリティ
6. 集計 (summarize) & AUC/効率計算
7. 追加: プロットスクリプト (任意)

これで本ディレクトリ単独で再現可能な仕様は完結。

## ポリシー

- random: 候補通行可能セルから一様サンプル
- gedig_simple: gedig 最大 (同値はランダム) / 全部 -1 → random fallback
- threshold_t: gedig >= t の集合から最大 (なければ random)

## 実験設計

Maze サイズ: 8x8, 12x12
シード: 10 (0..9)
ポリシー: random, gedig_simple, threshold_{1,2,3} (t=1,2,3)  ※ t=0 は simple と同義のため省略
1 episodes/条件/シード → 合計: 2(サイズ)*10(シード)*5(ポリシー)=100 run
(必要なら 3 エピソード/シードに増量)

## 成功判定

- success_rate(gedig_simple) > success_rate(random)
- mean_steps(gedig_simple) < mean_steps(random) かつ path_efficiency 改善
- threshold_t の曲線で U 字または plateau (t 過大で探索狭窄 ⇒ success_rate 低下)

## 出力構成

```text
experiments/gedig_minimal/results/
  raw/
    <mazeWxH>/<policy>/seed_<n>.json
  summary/
    aggregate.json   // 全条件
    by_policy.json
  figures/
    success_rate.png
    mean_steps.png
    gedig_auc_vs_efficiency.png
```

## 実装タスク (R1)

1. simple_navigator.py  (環境と方策適用ループ)
2. policy モジュール: compute_gedig(candidate) 実装
3. run_batch.py: ループ (arg: --maze-size, --seeds, --policies, --output-dir)
4. summarize.py: raw 走査→集計 JSON
5. (任意) analyze.ipynb: プロット
6. docs 連携: README 更新 & 図リンク

## R2 以降ロードマップ (簡易)

R2: frontier 近似 (2-step reachable 未訪問数) を neighbor_gain へ加算
R3: stagnation 単独 (停滞カウンタで探索半径拡大/ランダム注入)
R4: spike/decay の差分導入 (adaptive 先行の簡略版)
R5: 現行本体 geDIG 実装との A/B 分解 (寄与分析)

### Layer 対応と本実験位置づけ

| Layer | 目的 | 本最小実装での扱い | 導入タイミング(予定) |
|-------|------|--------------------|----------------------|
| Layer1: Norm Search / Embedding 距離フィルタ | 広域類似状態の抑制・状態空間スパース化 | 省略 (純粋な geDIG 効果分離のため) | R2〜R3 で軽量版 (単純 L2/正規化閾値) を試験的導入 → 成功なら正式化 |
| Layer2: Frontier Approx (局所 2-step / 構造的拡張) | 未訪問境界の見積り | R1 では未導入 | R2 で近似版 (2-step reach 未訪問数) |
| Layer3: geDIG Local Gain (未訪問 + 未訪問近傍) | 探索価値 (局所成長余地) | 本 R1 の主対象 (効果検証対象) | 継続 (改良: frontier / weighting) |
| Layer4: Adaptive (stagnation / spike / collapse) | 動的モード切替 | 全て除外 | R3〜R4 (簡略化版) |

本ロードマップではまず Layer3 単独の寄与 (ランダムとの差分) を定量化し、その後 Layer1 (Norm Search) を導入した際の追加改善幅を段階的に測定する。これにより各 Layer の限界効用 (marginal gain) を分離推定できる設計とする。

## リスク & 緩和

- Maze 内部 util 依存差異 → 必要箇所のみコピー/軽量再実装
- 壁密度で成功率変動大 → 迷路生成アルゴのパラメータ固定
- 小サイズで ceiling 効果 → 12x12 を最低限含める
- gedig 負値のみケース → fallback random 安全確保

## 工数見積 (R1)

- コード: ~150 行以内
- 集計 & 図: +1h
- ドキュメント反映: +0.5h

## 完了条件 (Definition of Done)

- 100 run 実行完了 (raw JSON 生成)
- summary JSON & 3 図出力
- README に結果サマリ (数値 + 図リンク)
- コード & ドキュメント lint 無警告

### R1 達成状況 (実行後追記)

- 実行: 100 run 完了 (8x8/12x12 × seeds=0..9 × 5 policies)
- 生成物: raw JSON, aggregate / by_policy / by_size_policy, figures (3種) 作成済
- README: 結果サマリ & 指標表 & 図一覧 追記済
- geDIG 効果: success_rate +40pt (0.50→0.90), 平均ステップ ~50% 短縮, AUC 約3.25倍
- 閾値挙動: threshold_1 = gedig_simple (冗長) / 高閾値で性能劣化 → U 字傾向確認
- 次段階: R2 で frontier 近似 + 冗長ポリシー整理 (threshold_1 除外)

### R2 状況 (Frontier 2-step 近似)

実装: `gedig_r2 = (1 - visited) + n1 + α * n2_unique` / AUC 正規化分母: `1 + 4 + 8α`。

α スイープ (0.15,0.25,0.35,0.45,0.60) 結果要約 (8x8/12x12 seeds=0..9):
- 0.15〜0.35: success_rate = 0.85 (R1 0.90 から -5pt), mean_steps_success ≈ 118 (R1 128.56 から ~10% 短縮)
- 0.60: success_rate 崩壊 (0.20) → frontier 過剰バイアス臨界確認
- 探索高速化と成功率のトレードオフ帯を特定 (α≈0.2 推奨 / default では未適用)

所見:
- Frontier 情報は局所 geDIG に対し “成功率微減 / ステップ短縮” の二律背反寄与
- R1 の成功率を最優先する場合はオフ、長径迷路高速化が目的なら低 α でオン

### R3 状況 (Adaptive α 最小実装)

実装: `gedig_r2_adaptive_<αmin>_<αmax>` (初期 0.1〜0.25)。
- 停滞判定: 連続 n2=0 (zero-streak)
- アクション: drop_threshold=3 で α 半減 / recover_threshold=8 で +0.02 線形回復
- メタログ: drop / raise 回数, 平均 / 最終 α 記録

12x12 seeds=0..9:
| Policy | success_rate | mean_steps_success |
|--------|--------------|--------------------|
| gedig_simple | 0.80 | 189.50 |
| gedig_r2 (α=0.2 固定) | 0.70 | 194.29 |
| gedig_r2_adaptive | 0.70 | 194.29 |

所見:
- 現設定では成功率改善なし (R2 と同等) / AUC わずか上昇 → 発火効果は軽微
- “安全な実験的オプション” として最低限完成 (MV adaptive)

### 次アクション (スケール & Layer1 検証)

短期 (優先度 高):
1. 大サイズ 20x20 / 32x32 追加で R1 vs R2 vs R3 スケール曲線 (success, steps, AUC)
2. α 動的上限拡張実験: 停滞時のみ αmax=0.35 許容 (通常 0.25)
3. zero-streak 指標と成功率/失敗ケースの相関抽出 (閾値自動最適化)

中期:
4. Layer1 (簡易 Norm 近傍抑制: 直近 k 訪問セルとの L2 < τ をスキップ) 導入試作
5. geDIG AUC と “最短距離超過率 (steps / 最短距離)” の相関検証 (指標妥当性強化)
6. BFS / DFS baseline (コスト計測付き) との比較テーブル (効率 vs 品質)

後続 (論文化視点):
7. α 臨界点 (成功率急落) の安定再現と安全領域マップ化
8. Adaptive 改良: 失敗リスク上昇検知 (success 残余距離 / 残ステップ比) による α 上限自動低下

### R2/R3 完了判定メモ
- R2: 臨界帯特定 + トレードオフ記述 + 冗長閾値整理 済 → Done
- R3: 最小適応 (イベントログ, 発火条件) 実装 & 動作確認 → Minimum Done (性能改善は将来タスク)



## geDIG 理論指向拡張ロードマップ (Core → Full)

本節で “geDIG-core (minimal)” から “理論整合性の高い geDIG-full” へ段階的に拡張する計画を定義する。

### 用語整理

- geDIG-core: 現行 R1/R2/R3 実装 (未訪問 + 1-step 分岐 + 2-step frontier(単層) + 簡易 adaptive)。
- geDIG-full (目標像): multi-depth frontier 減衰合成 / 目標距離項 / 再訪・類似抑制 (Layer1) / 真正規化 AUC / 高度停滞適応 (比率指標) を含む。

### 拡張ラダー (Ablation Ladder)

順序と各段階の追加式要素・評価指標:

| 段階 | ラベル | 追加要素 | 目的 | 新規パラメータ | 主要評価指標(増分) |
|------|--------|----------|------|----------------|---------------------|
| 0 | Core | (既存) base + n1 (+ α*n2) | 最小核確認 | α (frontier) | success / steps / core_AUC |
| A | TrueNorm | True 正規化 AUC (候補毎 max) | カーブ妥当性 | なし | true_auc, core_auc 相関 |
| B | MultiDepth | n3 (depth=3) 減衰加算 | より広い frontier 近似 | α2 (既存), α3, γ | success 変化, AUC↑ |
| C | GoalBias | 距離項 -λ*(残距離/初期距離) | 収束加速 | λ | steps↓ / success 保持 |
| D | Redundancy | 再訪/近接ペナルティ -ρ | ループ抑制 | ρ, K (履歴長) | 再訪率↓ / steps↓ |
| E | Adaptive+ | 窓比率停滞検出 + 重み動的 (β, τ) | 停滞脱出安定 | τ, β, W | stagnation_ratio↓ |
| F | Full | 全統合 + 自動安全域調整 | 理論形完成 | (上記全部) | Ablation 曲線一貫性 |

### 予定式 (段階的拡張)

現行 (Core R2):

```text
geDIG_core = (1 - visited) + n1 + α2 * n2
```

MultiDepth (B):

```text
geDIG_md = (1 - visited) + n1 + α2 * n2 + α3 * γ * n3
```

- n3: 3-step 範囲で 1,2-step と重複しない新規未訪問ユニーク集合カウント
- γ: 深さ減衰 (0<γ<1, 初期 0.6)

GoalBias (C) 追加後:

```text
geDIG_goal = geDIG_md - λ * (ManhattanRemaining / ManhattanInitial)
```

Redundancy (D) 追加後:

```text
geDIG_red = geDIG_goal - ρ * I[candidate in recent_K or (近傍ベクトル類似度 > θ)]
```

- 初期は位置再訪のみ (θ 未使用)。

Adaptive+ (E): frontier 関連重み {α2, α3} を窓 W=12 の “frontier 有効率” f = (n2+n3>0 の割合) に基づき調整

```text
if f < τ_low: α2, α3 = 0.5 * α (下限保持)
if f > τ_high: α2 ← α2 + β*(α2_max-α2)  (α3 も同様)
```

### メトリクス拡張

- true_auc: Σ (geDIG_t / max_candidate_t) / steps
- stagnation_ratio: n2+n3==0 ステップ比
- redundancy_penalty_rate: 再訪ペナルティ発火回数 / steps
- frontier_depth_mix: (Σn2, Σn3) 比率
- distance_efficiency: 最短路長 / 実ステップ (成功時)
- adaptation_event_counts: drop / raise / stable 区間長分布

### 成功基準 (各段階)

1. TrueNorm: true_auc と core_auc のスピアマン相関 ≥0.9
2. MultiDepth: success_rate 非劣化 (Δ ≥ -2pt) かつ mean_steps_success 改善 (≤ -5%) または true_auc 有意上昇 (p<0.05)
3. GoalBias: success_rate 維持 (Δ ≥ -2pt) で mean_steps_success 追加 -5% 以上
4. Redundancy: 再訪率 (unique_cells/steps) 改善 or redundancy_penalty_rate >0 かつ steps↓
5. Adaptive+: stagnation_ratio 減少 (≥ -20% 相対)
6. Full: Ablation 曲線が単調または準単調改善 (失敗逆転なし)

### パラメータ初期レンジ

- α2: 0.15,0.20,0.25
- α3: 0.05,0.10
- γ: 0.5,0.6
- λ: 0.03,0.05,0.08
- ρ: 0.1,0.2  (K=40)
- τ_low / τ_high: 0.2 / 0.5
- β: 0.25

### リスク & 緩和 (追加分)

| リスク | 内容 | 緩和 |
|--------|------|------|
| Depth 計算負荷 | n3 集合構築でオーバーヘッド | 再利用 set と一時バッファ、プロファイルで >+25% なら pruning |
| 距離項過大 | ゴール一直線で探索幅喪失 | λ sweep 小刻み + AUC 低下検知でロールバック |
| 再訪ペナルティ過剰 | デターミニスティック化 | ρ 小さく開始 + ランダム tie-break 維持 |
| 適応振動 | α 上下反復 | hysteresis: τ_low < τ_high 明確化 + クールダウン導入 |
| 指標スパース | 失敗多で統計不安定 | seeds >=30 / Wilson 区間表示 |

### 実行優先度 (次スプリント)

1. (A) True 正規化 AUC 実装 & 既存結果再集計
2. (B) Multi-depth (n3) 追加 + 小スイープ
3. (C) 距離項 λ 導入 & 効果測定
→ レポート: Core vs A vs B vs C の 4 段階図表

次スプリント以降:
4. (D) 再訪ペナルティ
5. (E) Adaptive+ (比率指標版)
6. (F) Ablation & Full 定義まとめ (ドキュメント章 draft)

### ドキュメント更新指針

- README: “Core vs Full” 比較節を追加 (図: Ablation Stairs)
- PLAN: 本節維持し進捗毎に ✅ マーク付与
- 各段階コミットメッセージ: `geDIG:A TrueNorm`, `geDIG:B MultiDepth` 等プレフィックス統一

### 現状ステータス サマリ (2025-09-05)

| 段階 | 状態 | 備考 |
|------|------|------|
| Core (0) | ✅ | 8/12/20/32 スケール基礎取得 (32 で R1 崩壊 / R2 生存) |
| A TrueNorm | 未 | 実装予定 (次) |
| B MultiDepth | 未 | n3 設計のみ |
| C GoalBias | 未 | 距離正規化式案あり |
| D Redundancy | 未 | 位置再訪ペナルティ案策定 |
| E Adaptive+ | 未 | 窓比率指標・hysteresis 案済 |
| F Full | 未 | Ablation 完了後定義 |

---
（以上 追加）



