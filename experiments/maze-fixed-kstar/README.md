# 固定分母 geDIG 迷路実験（Fixed-K★ Maze Study）

本ディレクトリは、論文第5章「部分観測迷路におけるオンライン統合」の検証を、最新の理論更新（`F = ΔGED − λ·ΔIG`, 固定分母 `log K⋆`, 二段しきい値セレクタ）に沿って再構成した実験群です。`maze-online-phase1` 系のロジックを再利用せず、`src/insightspike` のメインコード（`SimpleMaze`, `GeDIGCore`, `TwoThresholdCandidateSelector` など）だけで完結するように実装しています。

## 実験目的
- 0-hop geDIG 指標（`g₀`）と multi-hop 集約値（`gmin`）が、固定分母 `log K⋆` かつ局所 Cmax (=1+K⋆) で安定的に比較可能かを確認する。
- 二段しきい値 (`θcand`, `θlink`) ＋ Top-M 固定により、候補数増大で ΔIG が希釈されない運転が可能かを評価する。
- 迷路規模（15×15〜25×25）、seed 多数（20〜50）での成功率・ステップ数・エッジ削減率を記録し、旧スケールのログ（`results/ig_norm_*`）と比較できるようにする。

## 主要パラメータ
| 区分 | 設定 | 備考 |
|------|------|------|
| 迷路生成 | `SimpleMaze` (DFS/rooms 等を選択可能) | `--maze-type` で切り替え |
| 探索上限 | `--max-steps`（既定 2,000） | goal 未到達でもログを残す |
| geDIG | `λ=0.5`, `use_multihop=True`, `max_hops=5`, `decay_factor=0.85`, `adaptive_hops=False`, `use_local_normalization=True` | 線形迷路でも multi-hop を深追い |
| 二段しきい値 | `θcand=0.45`, `θlink=0.35`, `candidate_cap=32`, `top_m=32` | CLI で変更可 |
| 半径制御 | `cand_radius=3.0`, `link_radius=1.5` | `--cand-radius`, `--link-radius` |
| 出力 | JSON（集計＋各ステップメトリクス）, 任意で CSV | `--output`, `--step-log` |

Scand(q) の半径 (`cand_radius`) は観測可能エピソードと未探索分岐のみを含めるために設定しています。Slink(q) (`link_radius`) は観測エピソードを中心によりタイトな半径でリンク集合を絞り込みます。これらはいずれも重み付き L2 距離（論文の diag(w) ノルム）で判定され、CLI から調整できます。

## 評価指標
- 成功率（goal 到達率）, 平均ステップ数, 平均追加エッジ数
- `g₀` 平均・分散, `gmin` 平均, multi-hop 使用率, `best_hop` 分布
- ΔGED, ΔIG, K⋆（候補サイズ）および `log K⋆`, `l1_candidates`
- 候補概要 `candidate_selection`（`k_star`, `theta_cand`, `theta_link`, `k_cap`, `top_m`, `log_k_star`）

## シーケンスフロー
```mermaid
flowchart TD
    A[Episode Start] --> B[Reset SimpleMaze & Graph]
    B --> C[観測取得: MazeObservation]
    C --> D[観測ノードをグラフへ登録]
    D --> E[観測＋記憶ノードを共通スコアリング]
    E --> FSEL[TwoThresholdCandidateSelector<br/>(θcand, θlink, k_cap, top_m)]
    FSEL --> GDEC{K⋆ ≥ 1 ?}
    GDEC -- yes --> HYES[ig_fixed_den = log K⋆<br/>l1_candidates = K⋆]
    GDEC -- no --> HNO[ig_fixed_den = None<br/>l1_candidates = None]
    HYES --> IRES
    HNO --> IRES[GeDIGCore.calculate<br/>(Cmax=1+K⋆, multihop)]
    IRES --> JMET[結果: g₀, gmin, ΔGED, ΔIG]
    JMET --> KACT[行動選択・迷路更新]
    KACT --> LEND{Goal 到達?}
    LEND -- no --> C
    LEND -- yes --> MFIN[エピソード確定]
    MFIN --> NLOG[統計集計 & ログ出力]
```

## 処理シーケンス詳細
```mermaid
flowchart LR
    subgraph Step[t 時点の1ステップ処理]
        O[観測取得<br/>MazeObservation] --> F[特徴ベクトル化]
        F --> REG[観測ノードを<br/>グラフへ反映]
        REG --> CAND[観測＋記憶ノードを統合<br/>共通スコアリング]
        CAND --> SEL[TwoThresholdCandidateSelector<br/>(θ_cand, θ_link, top_m, k_cap)]
        SEL -->|k⋆>=1| KPASS[log k⋆, l1_candidates<br/>を算出]
        SEL -->|k⋆=0| KZERO[フォールバック選択<br/>候補無し]
        KPASS --> ACT[観測候補から最終アクション選択]
        KZERO --> ACT
        ACT --> ENV[環境更新<br/>SimpleMaze.step]
        ENV --> GRAPH[グラフ更新<br/>ノード/エッジ追加]
        GRAPH --> GEDIG[GeDIGCore.calculate<br/>(enable_multihop, log k⋆, Cmax)]
        GEDIG --> METRIC[g₀, ΔGED, ΔIG, best_hop 取得]
        METRIC --> GATE{ゲート評価?}
        GATE -->|AG/Hop判定| HOPCTRL[多ホップ評価維持<br/>or 制限]
        GATE -->|DG発火| ACTIONCTRL[バックトラック/停止の判断材料]
        HOPCTRL --> LOG
        ACTIONCTRL --> LOG
        GATE -->|ログのみ| LOG[StepRecordへ記録]
    end
    LOG --> NEXT{Goal判定}
    NEXT -- 未達 --> Step
    NEXT -- 到達 --> END[エピソード完了・集計出力]
```

## ベクトル表現（論文仕様）
- エピソードは 8 次元ベクトル $\mathbf{v}=[x/W, y/H, dx, dy, \text{wall}, \log(1+\text{visits}), \text{success}, \text{goal}]$ を用います。ノード側では絶対座標ベース（$x/W, y/H$ はノード自身の座標）でメタ情報を保持し、クエリ処理時に「現在位置を基準とした相対表現」に変換してから候補スコアリングに供します。
- クエリは $\mathbf{q}=[x/W, y/H, 0, 0, 1, 0, 0, 0]$ とし、方向バイアスを持たず未探索通路を優先します。観測候補・記憶候補いずれもこのクエリ座標と一致するように再表現されます。
- 類似度は重みベクトル $\mathbf{w}=[1,1,0,0,3,2,0,0]$ に基づく加重 L2 距離と $\exp(-d/T)$（$T=0.1$）で計算し、TwoThresholdCandidateSelector のスコアとして用いています。IG 計算時も同じ重みで候補・クエリをマスキングし、行き止まり・T 字路で情報量が適切に変動するようにしています。
- 観測候補には直近の行動 $(dx,dy)$ と訪問回数を含む仮想エピソードベクトルを生成し、記憶ノードは訪問履歴（座標・入射方向・訪問回数）を絶対表現として保持します。候補計算時に現在位置を基準とした相対ベクトルへ変換することで、クエリ／候補が同じ `x/W, y/H` から始まる表現になります。

## 使い方
```bash
# 15×15 迷路で 20 seeds を評価し、集計を JSON に保存
python run_experiment.py \
  --maze-size 15 \
  --seeds 20 \
  --lambda-weight 0.5 \
  --theta-cand 0.45 --theta-link 0.35 \
  --candidate-cap 32 --top-m 32 \
  --max-hops 5 --decay-factor 0.85 \
  --cand-radius 3.0 --link-radius 1.5 \
  --output results/maze_fixed_kstar_15.json \
  --step-log results/maze_fixed_kstar_15_steps.json

# 25×25 迷路で DFS 生成、最大ステップ 4000、ログのみ収集
python run_experiment.py \
  --maze-size 25 \
  --maze-type dfs \
  --max-steps 4000 \
  --seeds 30 \
  --max-hops 5 --decay-factor 0.85 \
  --cand-radius 3.0 --link-radius 1.5 \
  --output results/maze_fixed_kstar_25.json

# hop を自動で途中終了させたい場合（既定では無効）
python run_experiment.py \
  --maze-size 15 \
  --seeds 10 \
  --cand-radius 3.0 --link-radius 1.5 \
  --adaptive-hops \
  --output results/maze_fixed_kstar_adaptive.json
```

## 出力形式
### 集計 JSON (`--output`)
```json
{
  "config": {"maze_size": 15, "seeds": 20, ...},
  "summary": {
    "success_rate": 0.85,
    "avg_steps": 412.3,
    "avg_edges": 368.1,
    "g0_mean": -0.012,
    "gmin_mean": -0.037,
    "multihop_usage": 0.42,
    "k_star_mean": 3.6
  },
  "runs": [
    {
      "seed": 0,
      "success": true,
      "steps": 398,
      "edges": 352,
      "k_star_series": [3,4,2,...],
      "g0_series": [-0.01,-0.02,...],
      "gmin_series": [...],
      "multihop_best_hop": 1
    },
    ...
  ]
}
```

### ステップログ (`--step-log`)
```json
[
  {
    "seed": 0,
    "step": 0,
    "position": [1,1],
    "candidate_selection": {"k_star": 3, ...},
    "delta_ged": 0.125,
    "delta_ig": -0.108,
    "g0": -0.012,
    "gmin": -0.028,
    "best_hop": 1,
    "action": "right"
  },
  ...
]
```

## 関連資料
- 論文 第3章（構造–情報ポテンシャル F）: 固定分母と Cmax 正規化の前提。
- 論文 第5章（迷路実験）: 成功率/スパイク率/ステップ数の指標。
- `docs/engineering/refactor_plan_geDIG_v2.md`: 変更履歴と追加テストログ。

## 今後の拡張メモ
- 旧ログ (`results/ig_norm_*`) との diff スクリプトを追加し、固定分母モードでの改善点を自動可視化する。
- Maze Navigator 既存コードとの比較ランを自動化（同一 seed で before/after を並列表に出力）。
- multi-hop シナリオで `k⋆` や `best_hop` の分布を可視化する Notebook を準備する。
