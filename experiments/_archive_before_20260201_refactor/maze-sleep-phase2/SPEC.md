# 仕様書: Maze Sleep Phase 2

**Version**: 0.1 (Draft)
**Date**: 2026-01-31

---

## 1. 目的

迷路実験において、**インデックス + グラフ構造** を用いた Sleep 相を実装し、
以下を実現する：

1. 過去の経験から「嫌な予感」を生成して行き止まりを回避
2. ゴール方向への経路を優先的に選択
3. 2回目以降の走破で学習効果を確認

---

## 2. データ構造

### 2.1 エピソードノード

```python
@dataclass
class EpisodeNode:
    # 識別子（離散）
    node_id: str                    # hash または (x, y, dir) のタプル

    # 構造特徴（類似度計算用）
    position: Tuple[int, int]       # (row, col)
    direction: int                  # 0-3 (上右下左)
    target_position: Tuple[int, int]  # 遷移先

    # 状態属性
    is_passable: bool
    visit_count: int

    # ラベル（評価用）
    outcome: float                  # +1 (正例), 0 (中立), -1 (負例)
    purpose: float                  # 目的への寄与度

    # メタデータ
    created_at: int                 # step番号
    episode_id: int                 # エピソードID
```

### 2.2 グラフ構造

```python
class KnowledgeGraph:
    nodes: Dict[str, EpisodeNode]
    edges: Dict[Tuple[str, str], EdgeData]

@dataclass
class EdgeData:
    confidence: float       # DG確定時の g_min
    outcome: float          # 接続先の評価
    created_at: int
```

### 2.3 インデックス

```python
class EpisodeIndex:
    # 識別用（exact match）
    id_index: Dict[str, str]        # node_id → node
    position_index: Dict[Tuple[int, int], List[str]]  # position → nodes

    # 類似検索用（オプション、大規模時のみ）
    vector_index: Optional[faiss.Index]
```

---

## 3. 処理フロー

### 3.1 Wake 相（オンライン）

```
1. 現在位置を取得
2. 再認/想起で関連エピソードを検索
3. エピソードのラベル（outcome）を参照
4. 行動を選択（AG-DG的判断）
5. 結果を観測
6. エピソードを記録（ノード追加）
7. 関連があればエッジを追加（DG確定）
```

### 3.2 Sleep 相（オフライン）

```
1. エピソードのラベルを逆伝播で更新
   - ゴール到達 → 経路上のノードに +α × γ^d
   - 行き止まり → 手前のノードに -β

2. エッジの整理
   - 使われなかったエッジの重み減衰
   - 頻出パターンの強化

3. インデックスの再構築（必要に応じて）
```

---

## 4. 再認/想起アルゴリズム

### 4.1 二段階検索

```python
def recognize_or_recall(current_state: State, graph: KnowledgeGraph) -> SearchResult:
    # Step 1: 再認（exact match）
    exact_nodes = graph.find_by_position(current_state.position)

    if exact_nodes:
        # 同じ場所の経験がある
        return SearchResult(
            type="recognition",
            confidence=1.0,
            nodes=exact_nodes
        )

    # Step 2: 想起（類似検索）
    query_vector = encode(current_state)
    similar_nodes = graph.search_similar(query_vector, k=10)

    return SearchResult(
        type="recall",
        confidence=max_similarity(similar_nodes),
        nodes=similar_nodes
    )
```

### 4.2 先読み（Foresight）

```python
def foresight(node: EpisodeNode, graph: KnowledgeGraph) -> float:
    """1-hopでの先読み。接続先のoutcomeを集約。"""

    score = 0.0
    for neighbor_id in graph.neighbors(node.node_id):
        neighbor = graph.nodes[neighbor_id]
        edge = graph.edges[(node.node_id, neighbor_id)]

        # 接続先のoutcomeを重み付き集約
        score += edge.confidence * neighbor.outcome

    return score
```

### 4.3 行動選択

```python
def select_action(current_state: State, graph: KnowledgeGraph) -> Action:
    # 再認または想起
    result = recognize_or_recall(current_state, graph)

    action_scores = defaultdict(float)

    for node in result.nodes:
        action = node.direction

        # 直接の評価
        direct = node.outcome

        # 先読みの評価
        foresight_score = foresight(node, graph)

        # 統合
        confidence = sigmoid(direct + γ * foresight_score)
        score = node.similarity * confidence

        action_scores[action] = max(action_scores[action], score)

    # 最高スコアの行動を選択（またはsoftmax）
    return max(action_scores, key=action_scores.get)
```

---

## 5. ラベル更新ルール

### 5.1 outcome の更新

```python
def update_outcome_on_goal(path: List[EpisodeNode], α: float, γ: float):
    """ゴール到達時、経路上のノードを正例として強化"""
    for depth, node in enumerate(reversed(path)):
        node.outcome += α * (γ ** depth)
        node.outcome = clip(node.outcome, -1.0, +1.0)

def update_outcome_on_deadend(node: EpisodeNode, β: float):
    """行き止まり到達時、そのノードを負例として更新"""
    node.outcome -= β
    node.outcome = clip(node.outcome, -1.0, +1.0)

def update_outcome_on_revisit(node: EpisodeNode, β_revisit: float):
    """再訪問時、軽い負例として更新"""
    node.outcome -= β_revisit
    node.outcome = clip(node.outcome, -1.0, +1.0)
```

### 5.2 エッジ確定（DG）

```python
def should_add_edge(node_a: EpisodeNode, node_b: EpisodeNode,
                    similarity: float, f_value: float) -> bool:
    """AG-DG的なエッジ追加判断"""

    # AG: 候補として検討するか
    if similarity < θ_ag:
        return False

    # DG: 確定するか
    if f_value < θ_dg:
        return True

    return False
```

---

## 6. 評価指標

### 6.1 主要KPI

| 指標 | 説明 |
|------|------|
| success_rate | ゴール到達率 |
| avg_steps | 平均ステップ数 |
| deadend_avoidance | 行き止まり回避率（2回目以降） |
| learning_curve | 試行ごとのステップ数減少 |

### 6.2 内部指標

| 指標 | 説明 |
|------|------|
| recognition_rate | 再認が発生した割合 |
| foresight_accuracy | 先読みが正しかった割合 |
| edge_reuse_rate | 確定エッジが再利用された割合 |

---

## 7. 実装計画

### Phase 2a: 最小実装
- [ ] データ構造の定義
- [ ] 再認/想起の実装
- [ ] outcome ラベルの更新
- [ ] 基本的な行動選択

### Phase 2b: 先読み
- [ ] グラフ構造によるエッジ管理
- [ ] foresight の実装
- [ ] DGによるエッジ確定

### Phase 2c: 評価
- [ ] 評価ハーネスの整備
- [ ] Phase 1 との比較実験
- [ ] 学習曲線の可視化

---

## 参照

- `ARCHITECTURE.md` - アーキテクチャ図
- `README.md` - 概要
- `../maze-query-hub-prototype/` - Phase 1 実装
