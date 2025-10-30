# 迷路ナビゲーション実験 統一仕様書 v3

## 観測→判断→行動 決定フロー（Mermaid）

```mermaid
flowchart TD
  A[1. 観測<br/>現在位置から4方向の観測エピソード生成] --> B[2. クエリ生成<br/>DecisionEngine.create_query]
  B --> C[3. 必要な結線のみ実施<br/>・直前軌跡(trajectory)<br/>・L1閾値内の観測のみ]
  C --> D[4. L1ノルム検索 (観測+記憶)<br/>Top‑K/τ内を候補化]
  D --> E[4'. Virtual query‑geDIG(0‑hop)<br/>構造は汚さずスコアのみ]
  E -->|value ≤ θ_NA| F[通常行動: ソフトマックス選択]
  E -->|value > θ_NA| G[多段評価: geDIG multi‑hop]
  G --> H{ループ検出 ΔSP≥τ?}
  H -->|Yes| I[BT計画: ループ経路を採用]
  H -->|No| J[ge_full=min hop]
  J --> K{ge_full ≤ θ_BT?}
  K -->|Yes| L[BTターゲット選定 (policy=gedig_l1)<br/>F= w1·travel − kT·IG]
  K -->|No| F
  L --> M[BTプラン生成 (記憶グラフBFS)]
  I --> N[行動実行]
  M --> N
  F --> N
  N --> O[メモリ更新/スナップショット]
  O --> A
```

補足:
- Step3で「必要最小限の結線（直前軌跡＋観測のL1閾値内）」までを確定。
- Step4は「構造を変えず」virtual query‑geDIG（0‑hop）で判定。NA/BTゲートに併用可。
- BTターゲットは L1再検索の候補を geDIG風コスト F= w1·travel − kT·IG で最小化（policy=gedig_l1）。

# Maze Navigation Experiment: Autonomous Exploration with Episodic Memory

> NOTE (2025-09-01): PoC 方針により **形状付け(shaping)報酬/ペナルティはデフォルト無効** です。
> `MazeNavigatorConfig` の `wall_penalty`, `unknown_bonus`, `node_creation_cost` は全て 0.0 初期値。
> 行動選択は基本的に自前メモリグラフと geDIG `structural_improvement` による差分評価のみで駆動。
> 旧 shaping を有効化したい場合は構成ファイル/コードで該当値を明示設定してください。
This directory contains the implementation and results for the maze navigation experiment, a core component of the geDIG framework. The goal is to demonstrate that an agent can autonomously explore and solve complex mazes using only its episodic memory, driven by the geDIG metric as an intrinsic reward.

**Key Principle**: This experiment operates under a "pure intrinsic motivation" principle. All external rewards, such as distance-to-goal shaping or wall penalties (`wall_penalty`, `unknown_bonus`), are intentionally disabled. The agent's behavior is driven solely by its internal drive to organize its knowledge graph, as quantified by the geDIG score.

## 🎯 実験の目的
## 🎯 Objective

geDIG（Graph Edit Distance - Information Gain）理論を用いた迷路ナビゲーションの実装。
エピソード記憶による探索学習を実現する。
To validate the geDIG theory in a spatial navigation task by demonstrating:
1.  **Autonomous Exploration**: Solving complex mazes (up to 50x50) with only local 3x3 observability and no prior map knowledge.
2.  **Efficient Memory Formation**: Building a compact and efficient graph representation of the maze, proven by a >99% reduction in memory (edges) compared to naive strategies.
3.  **The "NA/DA Two-Stage Gate" Model**: Showing how the agent uses a combination of ambiguity detection (NA-like) and structural value assessment (DA-like) to make intelligent decisions like backtracking.

## 🧭 実験ステータス（2025-09-21）
## 🚀 How to Reproduce Paper Results

- トラック中のスイート
  - 15×15（複数 seed）/ 25×25（複数 seed）/ 50×50（スモーク）
- 直近の集計スナップショット
  - CSV: `results/fast_results_all.csv`, `results/pure_ge_bt_l1tau_summary.csv`
  - JSON: `results/final_gedig_test/summary_*.json`
  - ギャラリー: `scripts/build_gallery.py`（`run_summary.json` を含むディレクトリから index.html と表を生成）
- 推奨デフォルト（現行）
  - NA/BT: `MAZE_NA_GE_THRESH≈-0.005..-0.006`、`MAZE_BT_AGG=na_min`、`MAZE_BACKTRACK_THRESHOLD≈-0.012..-0.018`
  - マルチホップ: `MAZE_USE_HOP_DECISION=1`, `MAZE_HOP_DECISION_LEVEL=min`, `H≈3`
  - L1 検索: `MAZE_L1_WEIGHTS=[1,1,0,0,6,4,0,0]`, `MAZE_L1_UNIT_NORM=1`, `MAZE_L1_CAND_TOPK=8..10`
  - 強制 L1: `MAZE_WIRING_FORCE_L1=1`, `MAZE_WIRING_FORCE_L1_TAU≈0.66..0.70`
  - 配線: `MAZE_WIRING_TOPK=3`, `MAZE_WIRING_MIN_ACCEPT=1`, `MAZE_GEDIG_THRESHOLD≈0.0`
- 既知の傾向（抜粋）
  - geDIG が 0 近傍で停滞 → 強制 L1(τ≈0.66–0.70) で前進を再開、ノイズは抑制可能
  - 編み込み迷路で ΔSP<0 がショートカット形成と整合、サンプル/予算を抑えた測定が有効
  - 25×25 は NA をやや厳しめ、BT をやや緩めにすると BT 往復が減少
All results presented in the paper can be reproduced using the scripts in the `scripts/` directory. The primary configurations are managed via environment variables, with presets available in the runner scripts.

再集計の実行例:
### Reproducing the Main Comparison (Table 4 & 5 in Paper)
To reproduce the main results for the 25x25 and 50x50 mazes, use the batch runner script. This will run the experiment across multiple seeds and generate a summary CSV.

```bash
# 25×25 複数ディレクトリを CSV に集計
# Run the full experiment suite for 25x25 mazes
./experiments/maze-navigation-enhanced/scripts/run_25x25_seeds.sh 33 57 88 101

# Run the full experiment suite for 50x50 mazes
./experiments/maze-navigation-enhanced/scripts/run_50x50_batch.sh

# After running, aggregate the results into a CSV
python experiments/maze-navigation-enhanced/scripts/summarize_runs.py \
  experiments/maze-navigation-enhanced/results/25x25_maze \
  experiments/maze-navigation-enhanced/results/final_gedig_test \
  > experiments/maze-navigation-enhanced/results/summary_25.csv

# ビューワ付きギャラリーを生成（run_summary.json を含むフォルダ群から）
python experiments/maze-navigation-enhanced/scripts/build_gallery.py \
  --roots experiments/maze-navigation-enhanced/results \
  --out  experiments/maze-navigation-enhanced/results/index.html
  docs/images/gedegkaisetsu/50x50_run_* > results/summary_50x50.csv
```

## ⚙️ Presets and Calibration (New)

- Presets live in `experiments/maze-navigation-enhanced/configs/`:
  - `default.yaml` / `15x15.yaml` / `25x25.yaml` / `50x50.yaml`
- Loader utility exports ENV < preset < CLI overrides precedence for legacy scripts:

```python
from experiments.maze-navigation-enhanced.src.utils.preset_loader import load_preset, apply_env

cfg = load_preset(preset_name='25x25')  # or '15x15', '50x50'
apply_env(cfg)  # sets MAZE_* env variables and helper values
```

### k・τ Calibration (maze)

Grid search for `gedig_ig_weight (k)` and thresholds `gedig_threshold (τ)`, `backtrack_threshold (τ_bt)`:

```bash
PYTHONPATH=experiments/maze-navigation-enhanced/src \
python experiments/maze-navigation-enhanced/src/analysis/calibrate_ktau.py \
  --size 25 --seeds 16 \
  --k-grid 0.08 0.10 0.12 0.15 \
  --tau-grid -0.22 -0.18 -0.15 -0.12 \
  --tau-bt-grid -0.30 -0.25 -0.22 -0.18

# Results -> experiments/maze-navigation-enhanced/results/calibration/{grid_results,calibration}.json
```

### Statistical Summary

Quick success/steps/edges summary with 95% CI and win-rate vs simple:

```bash
PYTHONPATH=experiments/maze-navigation-enhanced/src \
python experiments/maze-navigation-enhanced/src/analysis/stats_summary.py --size 25 --seeds 16
```

## 📐 核心技術：geDIG理論

### アルゴリズム

```python
geDIG = GED - k*IG  # k=0.5

where:
- GED: Graph Edit Distance（グラフ編集距離）
- IG: Information Gain（情報利得）
- 正の値: 新しい情報の追加
- 負の値: グラフの短絡や統合
```

## 🏗️ リファクタリング後のアーキテクチャ（2025-08-22実装）

### ディレクトリ構造

```text
experiments/maze-navigation-enhanced/
├── src/
│   ├── core/                      # コアコンポーネント（再利用可能）
│   │   ├── __init__.py
│   │   ├── episode_manager.py     # エピソード管理
│   │   ├── vector_processor.py    # ベクトル処理
│   │   ├── graph_manager.py       # グラフ管理
│   │   └── gedig_evaluator.py     # geDIG計算
│   │
│   ├── navigation/                # ナビゲーション関連
│   │   ├── __init__.py
│   │   ├── decision_engine.py     # 行動選択
│   │   ├── branch_detector.py     # 分岐検出
│   │   └── maze_navigator.py      # メインナビゲーター
│   │
│   ├── experiments/                # 実験用スクリプト
│   │   ├── test_t_junction.py     # T字迷路テスト
│   │   ├── test_25x25_maze.py     # 大規模迷路テスト  
│   │   ├── test_gedig_threshold.py # geDIG閾値実験
│   │   └── benchmark.py           # パフォーマンス測定
│   │
│   ├── visualization/              # 可視化ツール
│   │   ├── maze_visualizer.py     # 迷路可視化
│   │   └── graph_visualizer.py    # グラフ可視化
│   │
│   ├── utils/                      # ユーティリティ
│   │   ├── maze_generator.py      # 迷路生成
│   │   └── config.py              # 設定管理
│   │
│   └── legacy/                     # 旧実装（参照用）
│
├── tests/                          # ユニットテスト
│   ├── core/
│   └── navigation/
│
├── configs/                        # 設定ファイル
│   ├── default.yaml               # デフォルト設定
│   └── weights.yaml               # 重みベクトル設定
│
└── results/                        # 実験結果
    ├── t_junction/
    ├── 25x25_maze/
    └── gedig_threshold/
```

## 📦 コアクラス設計

### 1. Core Components

#### EpisodeManager

```python
class EpisodeManager:
    """エピソードのライフサイクル管理"""
    
    責務:
    - エピソード作成（観測時）
    - 訪問回数の初期値設定と更新
    - エピソード検索
    - 統計情報の収集
    
    主要メソッド:
    - observe(position, maze) -> Dict[str, Episode]
    - move(position, direction) -> bool
    - get_episode(position, direction) -> Episode
    - get_statistics() -> Dict
```

#### VectorProcessor

```python
class VectorProcessor:
    """8次元ベクトルの生成と処理"""
    
    責務:
    - ベクトル生成
    - 重み適用
    - 距離計算
    
    主要メソッド:
    - create_vector(pos, direction, is_wall, visits) -> np.ndarray
    - apply_weights(vector, weights) -> np.ndarray
    - calculate_distance(vec1, vec2, weights) -> float
```

#### GraphManager

```python
class GraphManager:
    """エピソードグラフの構築と管理"""
    
    責務:
    - エッジ配線戦略の実装
    - グラフ構造の維持
    - グラフ統計の収集
    
    主要メソッド:
    - add_episode_node(episode)
    - wire_edges(episodes, strategy='simple')
    - wire_with_gedig(episodes, threshold=0.3)  # Approach D
    - get_graph_statistics() -> Dict
```

#### GeDIGEvaluator

```python
class GeDIGEvaluator:
    """geDIG値の計算と評価"""
    
    責務:
    - geDIG値計算
    - グラフ短絡検出
    - 閾値判定
    
    主要メソッド:
    - calculate(g1: nx.Graph, g2: nx.Graph) -> float
    - detect_shortcut(g1, g2) -> bool
    - should_backtrack(gedig_value, threshold) -> bool
```

### 2. Navigation Components

#### DecisionEngine

```python
class DecisionEngine:
    """行動選択ロジック"""
    
    責務:
    - クエリ生成
    - ノルム検索
    - 確率計算と選択
    
    主要メソッド:
    - create_query(position) -> np.ndarray
    - norm_search(query, episodes, weights) -> List[Tuple[float, Episode]]
    - select_action(episodes, temperature=0.1) -> str

    標準仕様（壁エピソードの扱い 固定ポリシー）:
    - 壁方向 Episode も常に距離計算候補に含める（除外しない方針に統一）
    - wall_flag 次元 (通路=+1, 壁=-1, weight=3.0) により通路との差が十分大きく選択確率はほぼ0
    - 負例（壁）ベクトルを保持することで密度/閉塞/ショートカット兆候など二次特徴に利用可能
    - include_walls フラグによる除外運用は廃止（後方互換のため引数は残るが内部的に True 固定）
```

#### BranchDetector

```python
class BranchDetector:
    """分岐点の検出と管理"""
    
    責務:
    - 分岐進入検出
    - 分岐完了検出
    - バックトラック判定
    
    主要メソッド:
    - detect_branch_entry(position, maze) -> bool
    - detect_branch_completion(position) -> bool
    - should_backtrack(gedig_value, threshold) -> bool
```

#### MazeNavigator

```python
class MazeNavigator:
    """メインナビゲーションシステム"""
    
    責務:
    - コンポーネント統合
    - メインループ実行
    - イベント管理
    
    構成:
    - episode_manager: EpisodeManager
    - graph_manager: GraphManager
    - vector_processor: VectorProcessor
    - decision_engine: DecisionEngine
    - gedig_evaluator: GeDIGEvaluator
    - branch_detector: BranchDetector
    
    主要メソッド:
    - run(maze, start, goal, max_steps=1000) -> bool
    - step() -> bool
    - get_statistics() -> Dict
```

## 🔧 エピソード管理仕様

### エピソードデータ構造

```python
@dataclass
class Episode:
    position: Tuple[int, int]      # 位置
    direction: str                  # 方向 ('N', 'S', 'E', 'W')
    vector: np.ndarray             # 8次元ベクトル
    is_wall: bool                  # 壁かどうか
    visit_count: int = 0           # 訪問回数
    episode_id: int = 0            # エピソードID
    timestamp: int = 0             # 作成時刻（ステップ数）
```

### 8次元ベクトル表現

```python
vector = [
    x/width,        # 0: X座標（正規化）
    y/height,       # 1: Y座標（正規化）
    dx,             # 2: 方向X（-1, 0, 1）
    dy,             # 3: 方向Y（-1, 0, 1）
    wall_flag,      # 4: 壁フラグ（1: 通路, -1: 壁）
    log_visits,     # 5: 訪問回数（log1p正規化）
    0.0,            # 6: 予約
    0.0             # 7: 予約
]
```

### 訪問回数管理仕様（重要）

#### 基本原則

- **エピソードは (位置, 方向) のペア**を表す
- 訪問回数は「その位置からその方向へ進んだ回数」をカウント
- 初期値は既訪問位置への方向なら1、未訪問なら0

#### 実装詳細

```python
# 観測フェーズ（4方向を観測）
def observe(current_pos, maze):
    for direction in ['N', 'S', 'E', 'W']:
        next_pos = current_pos + direction_vector[direction]
        
        if (current_pos, direction) not in episodes:
            # 初期訪問回数の決定
            initial_visits = 1 if next_pos in visited_positions else 0
            
            episode = Episode(
                position=current_pos,
                direction=direction,
                visit_count=initial_visits,
                ...
            )
            episodes[(current_pos, direction)] = episode

# 移動フェーズ（選択した方向のみ更新）
def move(current_pos, selected_direction):
    key = (current_pos, selected_direction)
    episodes[key].visit_count += 1
    # ベクトルも更新
    episodes[key].vector[5] = np.log1p(episodes[key].visit_count)
```

## 📊 重み設計

### デフォルト重みベクトル

```python
weights = np.array([
    1.0,  # x座標
    1.0,  # y座標
    0.0,  # dx（方向性は使用しない）
    0.0,  # dy（方向性は使用しない）
    3.0,  # 壁フラグ（重要：壁回避）
    2.0,  # 訪問回数（重要：未探索優先）
    0.1,  # 予約
    0.0   # 予約
])
```

## 🔄 処理フロー

### メインループ（現行実装 / バックトラック拡張 & クエリ配線反映）

### ベクトルインデックス統合 (Phase 4)

本実装ではクエリベース配線最適化のための軽量 VectorIndex 抽象を導入しています。

提供コンポーネント:

- `InMemoryIndex`: 線形走査 (L2) によるシンプルな実装。既存 heap ベース top-k と結果パリティ。
- `DataStoreIndex` (スタブ): DataStore が存在する環境でベクトルを永続化するためのラッパ (現段階では読み込み/保存ベストエフォート + 線形検索)。

Navigator への注入:

```python
from indexes.vector_index import InMemoryIndex
nav = MazeNavigator(maze, start, goal, wiring_strategy='query', vector_index=InMemoryIndex())
```

統計出力 (`get_statistics()`):

- `vector_index_size`: インデックスに登録されたエピソードベクトル数 (index 未使用時は 0)。
- タイミング計測: `wiring_ms`, `gedig_ms`, `snapshot_ms`, `recall_ms` 各 count / mean / p95 / max。

CLI 予定フラグ / 設計意図:

- `--use_vector_index`: 明示指定で InMemoryIndex を構築し query 配線を有効化 (デフォルト: オフ → 従来 heap フォールバック)。
- (挙動) 各ステップで新規エピソードの weighted ベクトル自動追加 (`index_auto_add=True`)
- (挙動) インデックス検索 oversample (k+5) → 壁/自己除外フィルタ → 上位 k エッジ生成 (距離上限 `query_wiring_max_dist`)
- (挙動) 失敗/例外時はヒープ線形スキャンへフォールバック (堅牢性)

今後 (Phase 5/6) 拡張予定:

- インデックスのメモリ制御 (flush / lazy load) と ANN backend (Faiss / hnswlib) 置換。
- インデックス成長を基にした適応的 top-k / dynamic 戦略。

### ⚡ Phase 6: ANN Backend & Dynamic Upgrade

実装済み機能:

- オプション `--ann_backend hnsw` 指定で `hnswlib` ベース ANN インデックスを利用 (未インストール時は安全に失敗し線形へフォールバック)。
- 線形 `InMemoryIndex` 利用時、ベクトル数が `ann_upgrade_threshold` を超えると自動で HNSW にアップグレード (再投入ベクトルは内部 `_vectors` 保持時のみ)。
- `get_statistics()` 追加フィールド:
- `ann_backend`: 現在のバックエンド (`None` or `hnsw`)
- `ann_init_error`: 初期化失敗時のエラーメッセージ
- `ann_index_elements`: ANN 有効時の登録件数
- `ann_upgrade_threshold`: 実際に使用された閾値
- `evicted_catalog_bytes`: 直近カタログサイズ (バイト)

#### 新規パラメータ (Navigator / CLI)

| Parameter | 説明 |
|-----------|------|
| `--ann_backend` | 最初から ANN を利用 (`hnsw`) |
| `--ann_m` / `--ann_ef_construction` / `--ann_ef_search` | HNSW チューニングパラメータ |
| `--ann_upgrade_threshold` | 自動アップグレード判定に用いる登録数閾値 (デフォルト 600) |
| `--catalog_compaction_on_close` | 実行終了時にエビクションカタログ JSONL を現在の LRU 内容で再書き出し (断片化解消) |

#### Eviction Catalog Compaction

Flush 有効化 + `persistence_dir` 指定時に生成される `evicted_catalog.jsonl` は追記型。`--catalog_compaction_on_close` を付与すると終了時に:

1. 現在 LRU マップ内容のみを書き直し
2. `catalog_compact` イベントを `event_log` に記録 (before/after バイトサイズ)
3. 統計 `evicted_catalog_bytes` 更新

#### ベンチマークスクリプト

`experiments/ann_benchmark.py` を追加。線形 vs HNSW の:

- 平均 / p95 レイテンシ
- Top-K 近傍 Jaccard / Hit 率
- スピードアップ比

実行例:

```bash
python -m experiments.ann_benchmark --n 8000 --queries 500 --top_k 8 --ann_ef_search 128
```

`hnswlib` が無い場合は警告のみ表示し線形結果のみ出力。

#### 推奨利用フロー (大規模迷路)

1. まず `--use_vector_index` (線形) で挙動/品質確認
2. 規模が拡大し `vector_index_size` > 閾値になり配線時間が増加 → 自動 `ann_upgrade` イベント発火
3. あるいは直接 `--ann_backend hnsw` で起動しパラメータチューニング

イベントログで `ann_init`, `ann_upgrade`, `ann_upgrade_failed` を確認可能。

---

### 🔍 Query Wiring Mode 仕様 (新)

従来の `simple` 配線は「現在位置で観測した 4 方向 Episode をローカルにスコア→即選択」でした。`wiring_strategy='query'` を指定すると以下の追加処理を行い、エピソード集合全体から類似検索 (nearest-neighbor) を利用した配線/候補選別を行います。

主眼:

1. 既存 Episode のベクトル空間構造を利用し「似た状況で高評価だった進行方向」を早期に再利用 (リコール的探索)
2. 距離分布が“飽和/退化” したタイミングで探索方針をリセット (バックトラック) し局所ループを避ける

処理フロー (query wiring 追加部分):

1. 新規観測 Episode を作成しベクトル (重み適用後) をインデックスへ自動追加 (`--use_vector_index` 指定時)
2. クエリベクトル (現在位置中心) を生成し top-k (既定4) 近傍検索 (線形 `InMemoryIndex` → 将来 ANN)
3. 取得候補をフィルタ (壁/自己位置など) → 距離上限 `query_wiring_max_dist` 内で採択
4. 採択 Episode 群からエッジ配線 (探索グラフ拡張) ＆ 次アクション候補生成
5. NN 距離統計を収集し退化判定 (後述) → 退化時バックトラック計画イベントを発火

関連統計 (`get_statistics`):

- `vector_index_size`: インデックス登録件数
- `simple_mode.query_generated`, `queries_per_step`: クエリ生成頻度
- `simple_mode.nn_degeneracy_triggers`: 退化トリガ発火回数

CLI 主要フラグ (例: `examples/maze50_experiment.py`):

```bash
--wiring-strategy query \
--use-vector-index \
--nn-degeneracy-trigger \
--nn-deg-var-thresh 1e-4 \
--nn-deg-range-thresh 5e-4 \
--nn-deg-min-unvisited 0.2 \
--nn-deg-no-growth-window 5
```

#### NN距離分布退化バックトラックトリガ

局所探索が“既知状態の再訪 + 未訪問進展なし” に陥る初期兆候をベクトル距離統計で検出し、早期バックトラックする新トリガです。

判定指標 (Top-K 近傍集合の距離配列 d):

- 分散: `var(d) < nn_deg_var_thresh`
- 距離レンジ: `(max(d)-min(d)) < nn_deg_range_thresh`
- 未訪問割合: `unvisited_ratio < nn_deg_min_unvisited_ratio` (= 新規開拓余地が小さい)
- グラフ成長無し: 直近 `nn_deg_min_window_no_growth` ステップでノード増加 0

全条件成立で `BACKTRACK_TRIGGER (reason=nn_degeneracy)` を発火し、続くステップで計画により後退。発火時点の統計スナップショット (var, range, unvisited_ratio, k, no_growth_recent) をイベント payload に格納。（デバッグ用: `self._nn_last_ranked_snapshot`）

推奨初期値 (迷路 50x50):

| パラメータ | 目的 | 目安 |
|------------|------|------|
| var_thresh | 距離分布の均質化検出 | 1e-4 |
| range_thresh | 近傍集合の“判別力消失” | 5e-4 |
| min_unvisited_ratio | 新規方向欠乏閾値 | 0.2 |
| no_growth_window | グラフ停滞許容長 | 5 |

チューニング指針:

- 退化判定が早すぎる → var/range を 10x 下げる、min_unvisited_ratio を 0.1 に下げる
- 遅すぎる/ループ残存 → var/range を 2〜5x 上げる、no_growth_window を 3 に短縮

ログ観察:
バックトラック理由別頻度 (将来拡張予定) で nn_degeneracy が支配的になり過ぎる場合は閾値緩和。

メリット:

- 設計上 geDIG のしきい値 (負のスパイク) を待たずに “探索的な停滞” を検出
- NN 距離分布を副次的な「内部状態エントロピー指標」として利用

制約 / 今後:

- 現状 k 固定 (動的 k 調整は未実装)
- ANN backend 利用時には距離推定誤差を考慮した適応閾値が必要 (TODO)
- マルチモーダル化 (方向クラスタ毎の局所分散) で誤検知低減予定

動作確認スニペット:

```bash
python examples/maze50_experiment.py \
    --size 50 --max-steps 800 --simple-mode 1 \
    --wiring-strategy query --use-vector-index \
    --nn-degeneracy-trigger --verbosity 1
```

終了時統計例 (抜粋):

```json
"simple_mode": {
    "query_generated": 780,
    "queries_per_step": 0.975,
    "backtrack_trigger_rate": 0.012,
    "nn_degeneracy_triggers": 5
}
```

調整後は再度同条件で比較し `nn_degeneracy_triggers` の増減や `unique_coverage` の改善を評価してください。

---

#### CLI 実行例

```bash
# クエリ配線 + インメモリインデックス有効化 (Top-K=4)
poetry run python scripts/run_maze_experiments.py \
  --variant simple \
  --wiring-strategy query \
  --use_vector_index \
  --wiring-top-k 4 \
  --max-steps 600 --summary

# ベースライン (従来 heap) との比較
poetry run python scripts/run_maze_experiments.py \
  --variant simple \
  --wiring-strategy query \
  --wiring-top-k 4 \
  --max-steps 600 --summary
```

差分確認ポイント:

- statistics.vector_index_size (>0 ならインデックス利用成功)
- timing.wiring_ms.mean_ms (インデックス / 非インデックス比較)
- path_length, unique_positions のパリティ（大きな乖離が無いか）

---

## 🧱 Phase 5: Episode Flush / Lazy Load (実装済み)

大規模迷路でのメモリ使用量を制御するための二段階エビクション + 遅延リハイドレート層。

### 機能概要

| 機能 | 説明 |
|------|------|
| 位置キャップ | `max_in_memory_positions` 指定時、位置単位で LRU/スコアリング選抜し丸ごとエビクト |
| Episode キャップ | `max_in_memory` 超過分をスコア(Recency Rank / inverse visit / 距離)で剪定 |
| 永続カタログ | エビクト時に `evicted_catalog.jsonl` へ JSONL 追記 (LRU メタ: id, position, direction, visit_count など) |
| 遅延リハイドレート | 現在位置訪問時に同位置のメタをスキャンし欠落方向 Episode を復元 (ベクトル再生成 + インデックス再登録) |
| カタログ圧縮 | `--catalog_compaction_on_close` で終了時に LRU 現在値のみでファイル再生成 (断片化/肥大化抑制) |
| 統計/イベント | `flush_eviction`, `rehydration`, `catalog_load`, `catalog_compact` などを `event_log` に記録 |

### 主要パラメータ (Navigator / CLI)

| パラメータ | 役割 | 典型値 |
|------------|------|--------|
| `--enable_flush` | フラッシュ機構オン/オフ | 大規模時にオン |
| `--flush_interval` | エビクション評価間隔 (step) | 50〜300 |
| `--max_in_memory` | Episode 総数上限 | 5k〜20k |
| `--max_in_memory_positions` | 位置数上限 (省メモリ & 広域抑制) | 2k〜8k |
| `--persistence_dir` | カタログ永続ディレクトリ | `./data/cache/...` |
| `--evicted_catalog_max` (内部) | LRU メタ保持上限 | 5k など |
| `--catalog_compaction_on_close` | 終了時再書き出し | 長時間ランで推奨 |

### スコアリング詳細

Episode レベル: `score = 0.6*recency_rank + 1.2*(1/(1+visit)) + 0.2*manhattan_distance(current)`

位置レベル (位置内 Episode 集約): oldest Episode の recency + 位置内平均 visit + 現在地点距離。

### リハイドレート戦略

- ランタイムに必要な位置へ到達したタイミングでのみ復元 (遅延)
- 復元済み方向の重複再生成を避けるため既存方向集合をチェック
- 復元直後に vector index (壁以外) に再追加

### 統計フィールド (一部)

`episodes_evicted_total`, `episodes_rehydrated_total`, `rehydration_events`, `rehydrated_unique_positions`, `flush_events`, `episode_eviction_events`, `position_eviction_events`, `evicted_catalog_size`, `evicted_catalog_bytes`

追加メトリクス説明 (2025-09-01 追加):

- `rehydration_events`: 1回の遅延リハイドレート試行で >=1 Episode が再構築された回数 (成功イベント数)
- `rehydrated_unique_positions`: 少なくとも1方向が再構築された一意の位置数 (空間的カバレッジ指標)

### 使い方例

```bash
python experiments/maze-navigation-enhanced/src/experiments/baseline_vs_simple_plot.py \
    --variant ultra50 \
    --wiring_strategy query --use_vector_index \
    --enable_flush --flush_interval 60 \
    --max_in_memory 8000 --max_in_memory_positions 3200 \
    --persistence_dir ./data/maze_mem --catalog_compaction_on_close \
    --seeds 101 202 --max_steps 600 --bootstrap_iterations 0 --verbosity 0
```

終了時 `catalog_compact` イベントが出力され、`get_statistics()` に `evicted_catalog_bytes` が反映されます。
3. EpisodeStore 永続化ラッパ + シリアライズ (JSON / DataStore)
4. Lazy rehydrate パス + テスト (evict→アクセス→復元)
5. 最適化 (バッチ書き込み / 圧縮 / ANN 連携)

これにより長時間実行 / 大規模迷路でのメモリフットプリント制御を実現予定。

### Eviction Policy パラメータ概要 (補足)

Navigator 生成時 `eviction_policy` を指定することで内部 Episode 削除戦略を切替可能:

| 値 | 戦略 | スコア式/特徴 |
|----|------|---------------|
| `heuristic` (デフォルト) | 複合スコア | `0.6*recency_rank + 1.2*(1/(1+visit)) + 0.2*manhattan_distance` |
| `lru` / `lru_visit` | LRU + visit バイアス | `timestamp + 0.05*visit_count` (低 visit 優先) |

選抜フロー:

1. `max_in_memory` 超過量を算出 (`over_by`)
2. 指定 Policy `select(episodes, over_by, context={current_pos})` で evict 対象 Episode ID 群取得
3. カタログへ追記 & index/unload を実行

拡張候補 (未実装):

| 候補 | 目的 |
|-------|------|
| `distance_weighted` | 現在地遠距離ノード優先エビクトで局所性強化 |
| `importance_based` | geDIG / structural_improvement 貢献度低ノード削除 |


リハイドレート (再構築) プロトタイプは `insightspike.algorithms.rehydration` に導入済 (統計: attempted / restored / skipped)。

---

## 📈 実験結果（2025-08-22）

### T字迷路テスト

- 迷路サイズ: 11×11
- ゴール到達: **27ステップ**
- ユニーク位置: 17

---

## 🧪 Day2 GeDIGリファクタ統合フラグ (2025-08-23 追加)

迷路実験で新旧GeDIG挙動を安全に切替・評価するための設定フラグ:

| フィールド | 型 | デフォルト | 説明 |
|------------|----|-----------|------|
| `use_refactored_gedig` | bool | True | 新しい正規化 & 報酬経路を使用 (Falseでlegacy product formula) |
| `enable_dual_evaluate` | bool | False | legacy+ref 並列計算し divergence Δ を計測 (性能コスト小増) |
| `dual_delta_threshold` | float | 0.3 | Δ が閾値超過で警告ログ |
| `structural_improvement_weight` | float | 0.5 | 構造改善 (>=0) をエネルギーから減算する係数 |

### 最小使用例

```python
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig
from insightspike.maze_experimental.navigators.gediq_navigator import GeDIGNavigator

cfg = MazeNavigatorConfig(
    use_refactored_gedig=True,
    enable_dual_evaluate=True,
    dual_delta_threshold=0.25,
    structural_improvement_weight=0.6,
)
navigator = GeDIGNavigator(cfg)
```

### 推奨ワークフロー

1. `enable_dual_evaluate=True` で 50〜100 ステップのウォームアップを走らせ Δ 分布を把握
2. Δ の p95 が閾値未満であれば `enable_dual_evaluate=False` に戻して本番ラン
3. 構造改善係数は 0.3〜0.7 の範囲で感度分析 (高すぎると探索が構造バイアスに偏る)

### 既知の制約 / TODO

- Δ 記録の集計CSV未出力 (今後: divergence ログチャネル追加)
- Spike FP 自動調整 (GeDIGMonitor) まだ Maze ループ未接続
- 分岐完了: 1回

### EventType 一覧 (自動生成セクション)

| 名前 | 値 | 説明(暫定) |
|------|----|-----------|
| START | start | ラン開始 |
| GOAL | goal | ゴール到達 |
| TIMEOUT | timeout | ステップ上限到達 |
| BRANCH_ENTRY | branch_entry | 新しい分岐に入った |
| BRANCH_COMPLETION | branch_completion | 分岐完了 (全方向探索) |
| DEAD_END | dead_end | 行き止まり検出 |
| BACKTRACK_TRIGGER | backtrack_trigger | バックトラック開始条件発火 |
| BACKTRACK_STEP | backtrack_step | バックトラック移動1ステップ |
| BACKTRACK_PLAN | backtrack_plan | バックトラック計画生成 |
| BRANCH_REMINDER | branch_reminder | 未探索分岐のリマインド |
| WALL_SELECTED | wall_selected | 壁方向が選択された (衝突) |
| FALLBACK_MOVE | fallback_move | デフォルト安全遷移 |
| FALLBACK_FAILED | fallback_failed | 安全遷移無し |
| STUCK | stuck | 行動不能状態 |
| SHORTCUT_CANDIDATE | shortcut_candidate | ショートカット候補検出 |
| REVERSE_TRACE | reverse_trace | 逆探索トレース完了 / 途中状態 |
| REVERSE_TRACE_ERROR | reverse_trace_error | 逆探索失敗 |
| FLUSH_SCORE | flush_score_probe | スコア/エナジー集計フラッシュ |
| FLUSH_EVICT | flush_eviction | エピソード/位置エビクション実行 |
| FLUSH_ERROR | flush_error | フラッシュ中エラー |
| CATALOG_COMPACT | catalog_compact | エビクションカタログ圧縮/読込 |
| REHYDRATION | rehydration | エピソード再構築試行 |
| ANN_INIT | ann_init | 近傍探索初期化成功 |
| ANN_INIT_FAILED | ann_init_failed | 近傍探索初期化失敗 |
| ANN_UPGRADE | ann_upgrade | ANN 構造アップグレード |
| ANN_UPGRADE_FAILED | ann_upgrade_failed | ANN アップグレード失敗 |
| ANALYSIS | analysis | ラン最終分析出力 |

> NOTE: 値と説明は現行コードベース走査より抽出 (要精査)。将来的に自動スクリプト生成へ移行予定。

### geDIG閾値実験

- 推奨閾値: **-0.1**
- 検出された負のgeDIG平均: -0.1000
- 効果: バックトラック機能による効率的な探索

## 🔍 今後の課題

### Phase 1（完了）

- ✅ コアクラス実装
- ✅ ナビゲーションコンポーネント実装
- ✅ 基本動作テスト

### Phase 2（進行中）

- [ ] 25×25大規模迷路テスト
- [ ] Approach D（geDIG閾値ベースエッジ配線）の実装
- [ ] パフォーマンス最適化

### Phase 3（計画）

- [ ] 完全版geDIG（C(20,3)評価）の実装
- [ ] エピソード削除戦略
- [ ] 探索と活用のバランス（ε-greedy）

## ⚠️ 重要な設計原則

### 責務の分離

- 各クラスが単一責務を持つ
- 実験スクリプトと本実装が明確に分離
- テストが容易な構造

### 拡張性

- 新しいエッジ配線戦略の追加が容易
- パラメータ調整が動的に可能
- 異なる意思決定アルゴリズムの実装が容易

### パフォーマンス

- エピソード数の制限
- グラフエッジ数の管理
- 効率的な検索アルゴリズム

---

### Version 3.0 - リファクタリング完了版（2025-08-22）

```

## 🧠 GNN (PyG) 統合計画メモ / Maze 実験拡張 (2025-09-06)

本セクションはローカル(macOS)環境で PyG ネイティブ拡張 (torch-scatter / torch-sparse / torch-cluster / torch-spline-conv) が未導入でも開発継続できるようにした現状と、Linux CI 上でのフル GNN 有効化ロードマップを整理したメモ。軽量ベクトル + geDIG 再ランク方式に「多段構造要約 / 先読みスコア拡張」を追加する段階的アプローチ。

### 1. 目的 (Why GNN?)
単純なベクトル類似度 + geDIG 局所差分では「離れた未探索枝の連動性」「潜在ショートカット候補の多 hop 文脈」「バックトラック後の再探索優先順位」などマクロ構造要約が弱い。GNN を導入し以下を狙う:

- Multi-hop 伝播: 未訪問領域に近い“ハブ”節点の価値増幅
- 構造圧縮: 分岐群を潜在空間でクラスタ化し冗長探索を減衰
- Risk/Reward 予測: 負 geDIG スパイク (密度過剰) 兆候を事前信号化
- Backtrack 回数削減: 行き止まり系列パターンの特徴抽出

### 2. 現状ステータス
- `INSIGHTSPIKE_DISABLE_GNN` 追加済: macOS では既定で 1 (無効) に設定想定
- layer3 graph reasoner: GNN 初期化は完全 lazy / 失敗時スタブへフォールバック
- torch / torch_geometric のみ導入 (ネイティブ拡張なし) でも例外を起こさない
- 迷路ナビゲーション本体は GNN 無しパスで安定稼働

### 3. アーキ挿入ポイント
```
[Episode Memory] -> [Vector Retrieval (top-K)] -> [geDIG Re-Rank]
    -> (optional) [Graph Builder -> PyG Data -> GNN Propagation]
    -> [Score Fusion] -> [Action Selection]
```

### 4. 最小実装 (Phase G0: Linux CI 有効化)
| ID | タスク | 出力/変更 | 目的 |
|----|--------|-----------|------|
| G0-1 | CI ワークフロー追加 (Linux) | `.github/workflows/maze_gnn.yml` | 拡張モジュール wheel 取得 & import 確認 |
| G0-2 | 条件付き pytest マーカ | `tests/*` | GNN 無効時は skip/xfail で安定 |
| G0-3 | GraphBuilder コンポーネント | `graph_builder.py` | 迷路状態→PyG `Data` 変換 |
| G0-4 | GNN Baseline モデル (2層 GCN) | `models/maze_gnn.py` | node -> pooled context + move logits |
| G0-5 | Feature Extractor | `features/maze_gnn_features.py` | ノード / エッジ特徴生成 |
| G0-6 | Score Fusion 層 | `decision/fusion.py` | geDIG / GNN 重み付き合成 (α,β) |
| G0-7 | メトリクス拡張 | recorder | multi-hop gain / fusion 比率ログ |
| G0-8 | ドキュメント更新 | 本README/PLAN | 手順/指標明文化 |

### 5. データスキーマ (初期案)
Node features (概念順):
1. normalized_visit_count
2. is_current (0/1)
3. is_goal (0/1)
4. recent_gedig_delta ([-])
5. branch_degree (0..4)
6. dead_end_flag (0/1 推定)
7. manhattan_distance_norm (0..1)
8. exploration_progress_local (近傍未訪問率)

Edge features (初期は省略 or 1-hot):
1. is_shortcut
2. is_backtrack_edge
3. local_density_delta (オプション)

### 6. モデル (Baseline v0)
```
X -> GCNConv(h=64) -> ReLU -> GCNConv(h=64) -> ReLU
   -> (a) Node embedding (current position行抽出)
   -> (b) Global mean pool (全体構造圧縮)
Concat(a,b) -> MLP(64->32->#moves) -> move logits
```
損失 (将来案): 選択行動と事後最短路比から導く ranking / margin 目的 (Phase G1)。最初は推論のみ (ランダム初期重み / 手動調整) で効果計測。

### 7. スコア融合
`final_score(move) = w_g * gedig_norm + w_v * vector_sim + w_n * gnn_logit_norm`

初期重み案: w_g=0.5, w_v=0.3, w_n=0.2 (後に自動 sweep)。GNN 出力信頼度 (variance / softmax entropy) 高いほど w_n を動的上乗せする適応式も Phase G2 で検討。

### 8. 計測指標 (GNN 導入差分)
| 指標 | 定義 | 期待方向 |
|------|------|----------|
| multi_hop_gain | (GNN あり steps / なし steps) | < 1.0 |
| redundant_branch_rate | 重複探索率 | 低下 |
| backtrack_count | 1ランの backtrack 回数 | 低下 |
| negative_spike_rate | 負 geDIG スパイク頻度 | 低下 |
| fusion_usage_ratio | GNN が最終選択に寄与 (w_n>0.25) 割合 | 適度 (~30-60%) |

### 9. フォールバック戦略
- Import エラー / flag / 拡張未導入 いずれかで即座に `GNN_DISABLED` ログ
- スタブはゼロベクトル出力 → fusion 層で w_n=0
- テストは `requires_gnn` マーカで条件 skip

### 10. リスク & 緩和
| リスク | 内容 | 緩和 |
|--------|------|------|
| 起動遅延 | PyG import 重 | lazy import + flag | 
| 過学習 (局所 maze seed) | 特定生成パターン偏り | 異種レイアウトバッチ検証 |
| 追加計算コスト | 1 step ms 増 | ノード数制限 + キャッシュ | 
| 融合不安定 | 重み調整難 | entropy-based gating |

### 11. 次アクション (優先順)
1. G0-1 CI ワークフロー
2. G0-2 pytest マーカ / skip
3. G0-3 GraphBuilder + 最小 Data (node only)
4. G0-4 GNN baseline (forward のみ)
5. G0-6 Fusion 実装 / 設定パラメータ追加
6. G0-7 メトリクス + ログ可視化

### 12. 設定パラメータ (追加予定)
| 名称 | 型 | 既定 | 説明 |
|------|----|------|------|
| enable_gnn_reasoner | bool | False | GNN モデル有効化 (Linux CI true) |
| gnn_hidden_dim | int | 64 | 中間層次元 |
| gnn_score_weight | float | 0.2 | w_n 初期値 |
| gnn_min_entropy | float | 0.6 | entropy 以下で w_n 増幅 |
| gnn_builder_max_nodes | int | 1200 | グラフ構築ノード上限 |

### 13. 参考ログ例 (想定 JSONL フィールド)
```json
{
  "step": 140,
  "fusion": {"w_g":0.5, "w_v":0.3, "w_n":0.2, "entropy":0.73},
  "gnn": {"nodes":412, "edges":768, "build_ms":4.2, "fwd_ms":2.9},
  "multi_hop_gain_est": 0.87
}
```

---

（この GNN セクションは安定後に専用 PLAN / Docs へ分離予定）
