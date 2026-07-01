# Graph-Persistent DG 仕様書

> **注記（2026-07-02）**: 本文書は設計時仕様。報酬値は §7 の調整指針に従い実装で調整済み —
> 実装値は novel **+0.2** / revisit **−0.4**（本文の +0.3/−0.3 は初期案のまま残す）。
> `move_success (+0.1)` は実装には存在しない。§3.5 の prune（閾値剪定）は実装済みだが
> 既定 OFF（孤立ノード除去のみ実行）。現行の実装値は README.md と
> `run_experiment_query.py` の報酬記録部を正とする。
> §6 Phase 2 の 4 条件アブレーション（伝播/辞書β/両方/なし）のうち、
> 「両方 vs 辞書βのみ」ペアが `docs/prereg/maze_sleep_ablation.md` として事前登録済み。

## 1. 背景と問題

### 1.1 現状

```
Wake1:
  graph = nx.Graph()          # 白紙
  探索 → graph構築
  # graph は捨てられる

Sleep:
  steps ログから辞書を抽出:
    sleep_plan:  Dict[(r,c)] → action_id
    sleep_q:     Dict[(r,c)] → Dict[action → Q値]
    sleep_edge:  Dict[(r,c)] → Dict[action → weight]

Wake2:
  graph = nx.Graph()          # また白紙
  辞書を softmax に β で加算
  # グラフを辿っているのではなく、辞書を引いている
```

### 1.2 欲しいもの

```
Wake1 → graph構築（経験を記録）→ Sleep（graph最適化）→ Wake2（最適化graph引き継ぎ）
```

---

## 2. Wake1: 何を、どこに記録するか

### 2.1 ヒューリスティックな報酬定義

各ステップの移動結果に対してスカラー報酬を定義する:

| イベント | 報酬 r | 根拠 |
|---|---|---|
| goal到達 | **+1.0** | 大正例: 目的達成 |
| 新セル通過 | **+0.3** | 小正例: 探索の進展 |
| 既訪問セル再訪 | **-0.3** | 小負例: 進展なし |
| 行き止まり到達 | **-1.0** | 大負例: 構造的な罠 |
| 壁衝突 | **-1.0** | 大負例: 無効な行動 |

これらは既存のevent検出ロジック（export_event_weights.py L188-224）で
判定可能なイベントと完全に一致する。新しい検出コードは不要。

### 2.2 記録場所: ノード属性

```python
# Wake1: 各ステップの移動後
dir_node = make_dir_node(current_position, action)
graph.nodes[dir_node]["reward"] = reward_value
```

**なぜノードか:**
- dirノード = 移動エピソードそのもの。報酬はエピソードの直接属性
- nx.Graph は無向 → エッジに報酬を置くと参照方向の曖昧さが生じる
- ノード属性ならどちらから参照しても一意

**エッジの役割:**
- 構造的接続関係のみを表す
- 報酬の伝播経路として機能する（伝播値自体はノードに格納）

> **設計動機（アナロジー）**: 人間の記憶では、評価は対象に直接結びつく（「蛇が怖い」）のであって
> 関係に結びつくのではない（「藪→蛇の関係が怖い」わけではない）。
> 藪が怖いのは蛇に接続しているから（伝播）。これと同じ構造を採用する。

### 2.3 DGエッジの役割

DGエッジは構造接続のみを表す:

```python
# commit.py: DGエッジ追加時
graph.add_edge(eu, ev, dg_committed=True)
```

DGエッジは報酬を持たない。到達先ノードの報酬は既にノード属性に記録されている。
伝播時に、DGエッジを通じて遠方ノードの報酬が手前ノードに影響する。

### 2.4 Wake1終了時のグラフの状態

```
例: 5x5迷路、Wake1でゴール未到達

ノード属性:
  dir(0,0,east).reward  = +0.3    新セル通過
  dir(0,1,east).reward  = +0.3    新セル通過
  dir(0,2,east).reward  = -1.0    壁衝突
  dir(0,2,south).reward = -1.0    行き止まり
  dir(0,2,west).reward  = -0.3    再訪問（戻る）
  dir(2,2,east).reward  = -1.0    行き止まり

エッジ（構造接続のみ）:
  query(0,0) --- dir(0,0,east)
  query(0,1) --- dir(0,1,east)
  query(0,2) --- dir(0,2,east)
  query(0,2) --- dir(0,2,south)
  query(0,2) --- dir(0,2,west)
  dir(0,2,south) ---[dg]--- dir(2,2,east)    DGエッジ
```

---

## 3. Sleep: 報酬の伝播

### 3.1 目的

Wake1のグラフには各ノードの直接的な報酬が記録されている。
Sleepはこれをエッジに沿って伝播させ、各ノードに**到達可能な累積報酬**を付与する。

```
Wake1 記録: node(B).reward = +0.3
            node(C).reward = +0.3
            node(goal).reward = +1.0

Sleep 伝播後: node(B).propagated = +0.3 + γ·max(C.propagated, ...)
                                 = +0.3 + 0.95·(+0.3 + 0.95·1.0)
                                 ≈ +1.49

            → 「Bから先に進むと、最終的にゴールに到達できる」
```

### 3.2 伝播アルゴリズム（ノードベース、Q-learning的）

```python
def propagate_rewards(graph: nx.Graph, gamma: float = 0.95, n_iters: int = 50):
    """
    各ノードの reward をエッジを通じて伝播する。

    propagated(n) = reward(n) + γ · max(propagated(neighbor))
    """
    # 初期化: propagated = 自身の reward
    for node, data in graph.nodes(data=True):
        data["propagated"] = data.get("reward", 0.0)

    # 反復伝播
    for _ in range(n_iters):
        updated = False
        for node, data in graph.nodes(data=True):
            neighbors_prop = [
                graph.nodes[nb].get("propagated", 0.0)
                for nb in graph.neighbors(node)
            ]
            best_neighbor = max(neighbors_prop, default=0.0)
            new_val = data.get("reward", 0.0) + gamma * best_neighbor

            if abs(new_val - data["propagated"]) > 1e-6:
                data["propagated"] = new_val
                updated = True

        if not updated:
            break  # 収束
```

### 3.3 伝播結果の具体例

```
Wake1で2つの経路を経験:
  経路1: start → A → B → C → dead_end (失敗)
  経路2: start → A → D → E → goal     (成功)

各ノードの reward（感情）:
  node(A).reward = +0.3   (新セル通過)
  node(B).reward = +0.3   (新セル通過)
  node(C).reward = +0.3   (新セル通過)
  node(dead_end).reward = -1.0  (行き止まり = 蛇)
  node(D).reward = +0.3   (新セル通過)
  node(E).reward = +0.3   (新セル通過)
  node(goal).reward = +1.0 (ゴール)

Sleep ノードベース伝播後 (γ=0.95):
  node(goal).propagated     = +1.0                         ゴール自体
  node(E).propagated        = +0.3 + 0.95 × 1.0  = +1.25  ゴールの隣
  node(D).propagated        = +0.3 + 0.95 × 1.25 = +1.49  ゴールに至る道

  node(dead_end).propagated = -1.0                         行き止まり自体（蛇）
  node(C).propagated        = +0.3 + 0.95 × (-1.0) = -0.65  蛇の隣（藪）
  node(B).propagated        = +0.3 + 0.95 × (-0.65) = -0.32 藪の手前

  node(A): neighborsは B(=-0.32) と D(=+1.49)
  node(A).propagated        = +0.3 + 0.95 × max(-0.32, +1.49)
                            = +0.3 + 0.95 × 1.49 = +1.72  (D方向の影響)
```

### 3.4 Sleep後のグラフの状態

```
分岐点Aでの選択:
  Aの隣人の propagated:
    B.propagated = -0.32  ← dead_end に接続（負の伝播）
    D.propagated = +1.49  ← goal に接続（正の伝播）

  → Wake2 でDを選ぶべきことが、隣人の伝播値から明らか
```

dead_end の負報酬が C → B へγ減衰で伝播し、goal の正報酬が E → D へ伝播する。

### 3.5 Sleep追加処理（オプション）

```python
def sleep_optimize(graph: nx.Graph, gamma=0.95, n_iters=50, prune=False):
    """Sleepの全体フロー"""
    optimized = graph.copy()

    # 1. 報酬伝播
    propagate_rewards(optimized, gamma=gamma, n_iters=n_iters)

    # 2. 孤立ノード削除
    optimized.remove_nodes_from(list(nx.isolates(optimized)))

    # 3. (オプション) 極端に低い伝播値のエッジを剪定
    if prune:
        to_remove = [
            (u, v) for u, v, d in optimized.edges(data=True)
            if d.get("propagated", 0) < -0.8
        ]
        optimized.remove_edges_from(to_remove)
        # 剪定後に孤立したノードも削除
        optimized.remove_nodes_from(list(nx.isolates(optimized)))

    return optimized
```

---

## 4. Wake2: 伝播値を使った行動選択

### 4.1 Softmaxへの介入

```python
# 現状 (辞書β方式):
score = similarity + β₁·sleep_q + β₂·sleep_edge + β₃·event + β₄·affordance

# 提案 (伝播値方式):
# 候補 dir_node の propagated を読む（ノード属性）
propagated = graph.nodes[dir_node].get("propagated", 0.0)
score = similarity + alpha * propagated
```

**4つのβ → 1つのalpha。** propagated は Sleep が計算済みなので、
Wake2の_score()は1行追加するだけ。

### 4.2 具体的な行動選択の流れ

```
Wake2 step N: 分岐点 A に到着。候補の dir_node を評価:

  dir(A,B方向): similarity=0.8, propagated=-0.32（dead_endへの伝播）
  dir(A,D方向): similarity=0.7, propagated=+1.49（goalへの伝播）
  dir(A,西):    similarity=0.9, propagated=-0.30（再訪問の負報酬）

  α=1.0 の場合:
    score(B方向) = 0.8 + 1.0×(-0.32) = 0.48
    score(D方向) = 0.7 + 1.0×(+1.49) = 2.19  ← 選ばれる
    score(西)    = 0.9 + 1.0×(-0.30) = 0.60

  → D方向が選ばれる（ゴールへの経路）
```

### 4.3 Wake2でグラフに存在しないノードの扱い

Wake1で未探索だった方向はグラフにノードがない → propagated = 0.0

```
score(未探索方向) = similarity + α · 0.0 = similarity のみ
```

→ 未探索方向は similarity だけで評価される（従来と同じ）。
propagated が正の方向が無ければ、未探索が自然に選ばれる。

### 4.4 失敗エピソード（ゴール未到達）の場合

Wake1でゴールに到達できなかった場合:
- 全ノードの reward は +0.3（通過）か -1.0（dead-end/壁）か -0.3（再訪問）
- **正の大報酬（+1.0）がないので、伝播値は全体的に低い/負になる**

```
Sleep 伝播後:
  全 dir_node の propagated が 0 以下 or 小さい正
  → 未探索方向（グラフに存在しない = propagated=0）が相対的に有利
  → Wake2 は未探索方向を優先する（これは正しい行動）
```

**ゴール未到達でも合理的な行動が出る。**
経験済みノードの伝播値が全て低い/負 → 未経験方向（propagated=0）が相対的に有利。

---

## 5. 実装仕様

### 5.1 Wake1: 報酬記録（run_experiment_query.py）

```python
# 各ステップの移動後、ノードに reward を記録
def _record_reward(graph, dir_node, event):
    """移動結果をノード属性に記録（感情はエピソードに結びつく）"""
    reward_table = {
        "goal_reached": +1.0,
        "novel_cell":   +0.3,
        "move_success": +0.1,   # 既知セルだが壁ではない
        "revisit":      -0.3,
        "deadend":      -1.0,
        "blocked":      -1.0,
    }
    r = reward_table.get(event, 0.0)

    if graph.has_node(dir_node):
        graph.nodes[dir_node]["reward"] = r
```

### 5.2 Wake1: DGエッジ（commit.py）

```python
# DGエッジは構造接続のみ。報酬はノードに既に記録されている。
def apply_commit_policy(...):
    if do_commit:
        for eu, ev in to_commit:
            graph_commit.add_edge(eu, ev, dg_committed=True)
            # reward は eu, ev の各ノードに既にある → エッジには不要
```

### 5.3 Sleep: 報酬伝播（新規ファイル graph_persistent_dg/sleep_propagate.py）

```python
def propagate_rewards(graph, gamma=0.95, n_iters=50):
    """セクション3.2のアルゴリズム"""
    # （上述の通り）

def sleep_optimize(graph, gamma=0.95, n_iters=50, prune=False):
    """セクション3.5のアルゴリズム"""
    # （上述の通り）
```

### 5.4 Wake2: グラフ引き継ぎ + propagated スコアリング

```python
def run_episode_query(
    seed, config, *,
    # 既存（後方互換）
    sleep_plan=None, sleep_q=None, sleep_edge=None, sleep_guide="off",
    # 新規
    inherited_graph=None,
):
    if inherited_graph is not None:
        graph = inherited_graph
    else:
        graph = nx.Graph()
```

### 5.5 Wake2: _score() の変更

```python
# 既存スコアに propagated を加算
def _score(...):
    base = similarity_score  # 既存計算

    # dir_node の伝播値を読む（Sleepが計算済み、ノード属性）
    propagated = 0.0
    if graph.has_node(direction_node):
        propagated = graph.nodes[direction_node].get("propagated", 0.0)

    return base + config.propagated_alpha * propagated
```

### 5.6 main() フロー

```python
# Wake1
warm_artifacts = run_episode_query(seed=seed, config=warm_cfg)

# Sleep
optimized = sleep_optimize(warm_artifacts.graph, gamma=0.95, n_iters=50)

# Wake2
eval_artifacts = run_episode_query(
    seed=seed, config=config,
    inherited_graph=optimized,
)
```

---

## 6. 段階的実装計画

### Phase 0: 報酬記録 + グラフ引き継ぎ（Sleepなし）

**変更**:
1. Wake1: 各ステップで `graph.edges[q, d]["reward"] = r` を記録
2. `inherited_graph` パラメータ追加
3. main(): Wake1のgraphをそのままWake2に渡す（伝播なし）
4. Wake2の_score(): `propagated` の代わりに `reward` を直接読む

**検証**: 直接的な reward だけでも行動改善するか
**変更量**: ~30行

### Phase 1: Sleep伝播

**変更**:
1. `sleep_propagate.py` 新規作成
2. `propagate_rewards()` 実装
3. main(): Wake1 → sleep_optimize() → Wake2

**検証**:
- propagated 値の分布を可視化（goal方向が正、dead-end方向が負になるか）
- γの感度分析
- 25x25, 51x51での成功率

### Phase 2: Softmax統合 + β廃止

**変更**:
1. _score() で propagated_alpha を使ったスコアリング
2. 辞書モード（sleep_plan, sleep_q, sleep_edge）と比較実験
3. β廃止の判断

**検証**:
- ablation: 伝播あり vs 辞書β vs 両方 vs なし
- alpha 1本 vs β 4本の成功率比較

---

## 7. 報酬テーブルの調整指針

初期値は以下だが、実験で調整する:

| イベント | 初期値 | 調整の方向 |
|---|---|---|
| goal_reached | +1.0 | 固定 |
| novel_cell | +0.3 | 大きくすると探索優先、小さくすると搾取優先 |
| move_success | +0.1 | 0にしても良い（既知は中立） |
| revisit | -0.3 | 大きくするとループ回避が強まる |
| deadend | -1.0 | 固定 |
| blocked | -1.0 | 固定 |

γ（割引率）: 0.95 → 伝播の到達距離を制御
- 大きい(→1.0): 遠くの goal/dead-end の影響が強い
- 小さい(→0.5): 直近の経験のみ反映

alpha（softmax介入強度）: 1.0 → propagated と similarity のバランス
- 大きい: 過去の経験を重視
- 小さい: 類似度（現在の構造評価）を重視

---

## 8. 既存コードとの対応

| 既存の仕組み | 新設計での対応 |
|---|---|
| sleep_q (Q-learning辞書) | propagated (ノード属性) |
| sleep_q_beta | propagated_alpha |
| sleep_edge (edge weight辞書) | reward (ノード属性) → propagatedに吸収 |
| sleep_edge_beta | 不要 |
| event_beta | 不要（rewardに統合） |
| affordance_beta | 不要（propagatedに統合） |
| sleep_plan (BFS辞書) | 不要（propagated が最大の隣人方向がplan相当） |

**辞書5種 + β4本 → ノード属性2つ(reward, propagated) + α1本**
**エッジは純粋に構造（接続関係）のみを表す。**

---

## 9. 検証シナリオ

### 9.1 最小テスト: 5x5迷路

```
S . . . .
. # # # .
. . . # .
. # . . .
. . . # G

Wake1: S→右→右→下→行き止まり→戻る→...→G到達 or 未到達
Sleep: 伝播
Wake2: 分岐点で正しい方向を選べるか？
```

期待: dead-end方向の propagated が負、goal方向が正

### 9.2 本番テスト: 25x25

| 条件 | 成功率の期待値 |
|---|---|
| 辞書なし（現状ベースライン） | 72% |
| 辞書あり（現状最良） | 85% |
| graph引き継ぎ + reward直読み（Phase 0） | > 75% |
| graph引き継ぎ + propagated（Phase 1） | > 85% |
| propagated + α最適化（Phase 2） | > 88% |
