# メインコード改修計画

**Version**: 0.1 (Draft)
**Date**: 2026-01-31

---

## 1. 現状分析

### 1.1 問題点

```
run_experiment_query.py:
  - 281KB+ の巨大ファイル
  - 全機能が1ファイルに集約
  - 改修・テストが困難
```

### 1.2 改修が必要な箇所

| 現状 | Phase 2 |
|------|---------|
| `compute_episode_vector()` 8次元固定 | 構造特徴 + ラベル分離 |
| `weighted_distance()` 全次元で距離計算 | 構造特徴のみで類似度 |
| 類似検索のみ | 再認 → 想起の二段階 |
| `sleep_edge_weight` 外付け | outcome ラベルをノードに内包 |
| 先読みなし | グラフ1-hopで先読み |

---

## 2. 改修アプローチ

### 推奨: モジュール分離 + 段階的移行

```
experiments/maze-sleep-phase2/
├── src/
│   ├── __init__.py
│   ├── memory.py          # KnowledgeGraph, EpisodeNode
│   ├── recognition.py     # 再認/想起アルゴリズム
│   ├── foresight.py       # 先読み
│   ├── action_selection.py # 行動選択
│   ├── label_update.py    # outcome更新
│   └── sleep.py           # Sleep相処理
├── run_experiment.py      # 新しいエントリポイント
└── tests/
    └── ...
```

### なぜ分離か

1. **テスト容易性**: 各モジュールを独立してテスト可能
2. **段階的移行**: 既存コードと共存しながら徐々に置き換え
3. **可読性**: 責務が明確になる
4. **再利用性**: ARC等への展開が容易

---

## 3. モジュール設計

### 3.1 memory.py

```python
"""記憶システムの中核"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import networkx as nx

@dataclass
class EpisodeNode:
    node_id: str
    position: Tuple[int, int]
    direction: int
    target_position: Tuple[int, int]

    # 構造特徴（類似度計算用）
    vector: np.ndarray = field(default_factory=lambda: np.zeros(6))

    # ラベル（評価用）
    outcome: float = 0.0
    purpose: float = 0.0

    # メタ
    visit_count: int = 0
    created_at: int = 0


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.position_index: Dict[Tuple[int, int], List[str]] = {}

    def add_node(self, node: EpisodeNode) -> str:
        """ノード追加 + インデックス更新"""
        self.graph.add_node(node.node_id, data=node)

        # position index
        pos = node.position
        if pos not in self.position_index:
            self.position_index[pos] = []
        self.position_index[pos].append(node.node_id)

        return node.node_id

    def add_edge(self, src: str, dst: str, confidence: float, outcome: float):
        """エッジ追加（DG確定）"""
        self.graph.add_edge(src, dst, confidence=confidence, outcome=outcome)

    def find_by_position(self, pos: Tuple[int, int]) -> List[EpisodeNode]:
        """位置でexact match（再認）"""
        node_ids = self.position_index.get(pos, [])
        return [self.graph.nodes[nid]['data'] for nid in node_ids]

    def neighbors(self, node_id: str) -> List[str]:
        """1-hopの接続先"""
        return list(self.graph.neighbors(node_id))
```

### 3.2 recognition.py

```python
"""再認/想起アルゴリズム"""

from dataclasses import dataclass
from typing import List, Literal
import numpy as np

@dataclass
class SearchResult:
    type: Literal["recognition", "recall", "novel"]
    confidence: float
    nodes: List[EpisodeNode]


def recognize_or_recall(
    current_position: Tuple[int, int],
    current_vector: np.ndarray,
    graph: KnowledgeGraph,
    similarity_threshold: float = 0.8
) -> SearchResult:
    """二段階検索"""

    # Step 1: 再認（exact match by position）
    exact_nodes = graph.find_by_position(current_position)
    if exact_nodes:
        return SearchResult(
            type="recognition",
            confidence=1.0,
            nodes=exact_nodes
        )

    # Step 2: 想起（類似検索）
    similar_nodes = search_similar(current_vector, graph, k=10)
    if similar_nodes and similar_nodes[0].similarity > similarity_threshold:
        return SearchResult(
            type="recall",
            confidence=similar_nodes[0].similarity,
            nodes=[s.node for s in similar_nodes]
        )

    # 新規状況
    return SearchResult(
        type="novel",
        confidence=0.0,
        nodes=[]
    )


def search_similar(
    query: np.ndarray,
    graph: KnowledgeGraph,
    k: int = 10
) -> List[SimilarityResult]:
    """構造特徴のみで類似検索"""
    results = []

    for node_id in graph.graph.nodes:
        node = graph.graph.nodes[node_id]['data']

        # 構造特徴のみで距離計算（ラベル除く）
        dist = np.linalg.norm(query[:6] - node.vector[:6])
        sim = 1.0 / (1.0 + dist)

        results.append(SimilarityResult(node=node, similarity=sim))

    return sorted(results, key=lambda x: -x.similarity)[:k]
```

### 3.3 foresight.py

```python
"""先読み（1-hop探索）"""

def compute_foresight(
    node: EpisodeNode,
    graph: KnowledgeGraph,
    gamma: float = 0.5
) -> float:
    """接続先のoutcomeを集約"""

    score = 0.0
    neighbors = graph.neighbors(node.node_id)

    for neighbor_id in neighbors:
        neighbor = graph.graph.nodes[neighbor_id]['data']
        edge_data = graph.graph.edges[node.node_id, neighbor_id]

        confidence = edge_data.get('confidence', 1.0)
        outcome = neighbor.outcome

        score += confidence * outcome

    return gamma * score
```

### 3.4 action_selection.py

```python
"""行動選択"""

import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def select_action(
    current_position: Tuple[int, int],
    current_vector: np.ndarray,
    graph: KnowledgeGraph,
    possible_actions: List[int],
    gamma: float = 0.5
) -> int:
    """再認/想起 + 先読みに基づく行動選択"""

    result = recognize_or_recall(current_position, current_vector, graph)

    action_scores = {a: 0.0 for a in possible_actions}

    for node in result.nodes:
        action = node.direction
        if action not in possible_actions:
            continue

        # 直接の評価
        direct = node.outcome

        # 先読み
        foresight = compute_foresight(node, graph, gamma)

        # 統合
        confidence = sigmoid(direct + foresight)

        # 類似度（再認なら1.0）
        if result.type == "recognition":
            similarity = 1.0
        else:
            similarity = result.confidence

        score = similarity * confidence
        action_scores[action] = max(action_scores[action], score)

    # 最高スコアの行動（未経験はデフォルトスコア）
    for a in possible_actions:
        if action_scores[a] == 0.0:
            action_scores[a] = 0.5  # 未経験のデフォルト

    return max(action_scores, key=action_scores.get)
```

### 3.5 label_update.py

```python
"""ラベル更新"""

def update_on_goal(
    path: List[EpisodeNode],
    alpha: float = 0.4,
    gamma: float = 0.95
):
    """ゴール到達時の逆伝播"""
    for depth, node in enumerate(reversed(path)):
        node.outcome += alpha * (gamma ** depth)
        node.outcome = max(-1.0, min(1.0, node.outcome))


def update_on_deadend(node: EpisodeNode, beta: float = 0.3):
    """行き止まり"""
    node.outcome -= beta
    node.outcome = max(-1.0, min(1.0, node.outcome))


def update_on_revisit(node: EpisodeNode, beta: float = 0.1):
    """再訪問"""
    node.outcome -= beta
    node.outcome = max(-1.0, min(1.0, node.outcome))
```

---

## 4. 移行計画

### Phase 2a: 最小実装（1週目）

```
[ ] memory.py: EpisodeNode, KnowledgeGraph
[ ] recognition.py: 再認/想起の基本実装
[ ] 既存の迷路環境と接続するアダプタ
[ ] 単体テスト
```

### Phase 2b: 先読み + ラベル（2週目）

```
[ ] foresight.py
[ ] label_update.py
[ ] action_selection.py
[ ] 統合テスト
```

### Phase 2c: 比較実験（3週目）

```
[ ] run_experiment.py: 新エントリポイント
[ ] Phase 1 との比較ベンチマーク
[ ] 結果可視化
```

---

## 5. 既存コードとの共存

### アダプタパターン

```python
# adapter.py
"""既存コードと新コードの橋渡し"""

from maze_query_hub_prototype.run_experiment_query import (
    SimpleMaze,
    compute_episode_vector as old_compute_vector,
)
from .memory import EpisodeNode, KnowledgeGraph


def convert_old_to_new(old_node_data: dict) -> EpisodeNode:
    """既存ノードデータを新形式に変換"""
    return EpisodeNode(
        node_id=str(old_node_data.get('node_id')),
        position=tuple(old_node_data.get('anchor_positions', [[0,0]])[0]),
        direction=old_node_data.get('direction', 0),
        target_position=tuple(old_node_data.get('target_position', [0,0])),
        vector=old_node_data.get('vector', np.zeros(6)),
        outcome=old_node_data.get('success', 0.0),  # 既存のsuccessを流用
        purpose=1.0 if old_node_data.get('is_goal') else 0.0,
    )
```

### 段階的置き換え

```python
# run_experiment.py

# 既存の迷路環境はそのまま使用
from maze_query_hub_prototype.run_experiment_query import SimpleMaze

# 記憶システムは新実装
from src.memory import KnowledgeGraph
from src.action_selection import select_action

def run_episode(maze: SimpleMaze, graph: KnowledgeGraph):
    """新しい記憶システムで迷路を解く"""
    ...
```

---

## 6. テスト戦略

### 単体テスト

```python
# tests/test_recognition.py

def test_recognition_exact_match():
    """同じ位置は再認される"""
    graph = KnowledgeGraph()
    node = EpisodeNode(node_id="1", position=(3, 5), ...)
    graph.add_node(node)

    result = recognize_or_recall((3, 5), ..., graph)

    assert result.type == "recognition"
    assert result.confidence == 1.0


def test_recall_similar():
    """似た位置は想起される"""
    ...


def test_novel_unknown():
    """未知の位置は新規"""
    ...
```

### 統合テスト

```python
# tests/test_integration.py

def test_deadend_avoidance():
    """2回目で行き止まりを回避できる"""
    maze = SimpleMaze(size=5, seed=42)
    graph = KnowledgeGraph()

    # 1回目: 行き止まりに入る
    run_episode(maze, graph, allow_deadend=True)

    # 2回目: 行き止まりを回避
    result = run_episode(maze, graph)

    assert result.deadend_visits == 0
```

---

## 参照

- `SPEC.md` - 詳細仕様
- `ARCHITECTURE.md` - アーキテクチャ図
- `../maze-query-hub-prototype/run_experiment_query.py` - 既存実装
