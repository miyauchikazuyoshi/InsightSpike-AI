# 記憶検索の三層アーキテクチャ：attention重み付き実装メモ

## 2026-02-08

---

## 1. 核心アイデア

**計算量は不確実性に比例する。知っていることに計算を使わない。**

| Layer | 判定 | 計算量 | β₁ | 人間の対応 |
|---|---|---|---|---|
| Layer 0 | 完全一致（再訪判定） | O(1) | β₁ = 0 | 「あ、ここ知ってる」 |
| Layer 1 | グラフ走査（attention参照） | O(degree) | β₁ = 小 | 「前は右だった」 |
| Layer 2 | 記憶全ソート（類似度検索） | O(N log N) | β₁ = 大 | 「似た場所あったかな…」 |

**再訪ならLayer 2をスキップ。** グラフが既に候補を持っている。

---

## 2. 二種類の類似度

| 類似度 | 重み | 目的 | 使う場面 |
|---|---|---|---|
| 生類似度 | なし（全次元等重） | **再訪認識**（ここに来たか） | Layer 0 発火条件 |
| 重み付き類似度 | WEIGHT_VECTOR | **候補選択**（どこに進むか） | Layer 2 ソート基準 |

生類似度は迷路の座標一致を一般化したもの。座標がない環境（RAG等）でも使える。

**重み付き類似度を再訪判定に使ってはいけない。**
方向やタスク依存情報をマスクするため、同一地点でも方向が違えば「遠い」と判定してしまう。

---

## 3. フロー全体

```
クエリ到着: query_vector 生成
      │
      ▼
[Layer 0] ハッシュ完全一致検索 O(1)
      │
      ├── ヒット → 再訪確定
      │     │
      │     ▼
      │   [Layer 1] グラフエッジ走査 O(degree)
      │     │  attention > θ のエッジだけ返す
      │     │  → 候補確定。Layer 2 スキップ
      │     │
      │     └── 候補不足 → Layer 2 へフォールスルー
      │
      └── ミス → 新規地点
            │
            ▼
          [Layer 2] 重み付き類似度ソート O(N log N)
            │  通常の候補選択
            │
            ▼
          evaluate_multihop → commit
          ハッシュインデックスに追加 O(1)
          グラフに新エッジ追加（attention = 1.0）
```

---

## 4. 実装コード

### 4.1 Layer 0: 量子化ハッシュインデックス

```python
"""
Layer 0: 生ベクトルの近似完全一致検索。
量子化によりO(1)ハッシュルックアップで再訪判定する。

迷路: resolution = 1/maze_size（セル1個分）
RAG:  resolution = 埋め込み粒度（例: 0.05）
"""

from typing import Dict, List, Tuple
import numpy as np

Node = Tuple[int, int, int]


class VectorHashIndex:
    def __init__(self, resolution: float = 0.05):
        self.res = float(max(1e-9, resolution))
        self._buckets: Dict[tuple, List[Tuple[Node, np.ndarray]]] = {}
        self._count = 0

    def _quantize(self, vec: np.ndarray) -> tuple:
        return tuple((vec / self.res).astype(int))

    def add(self, node_id: Node, raw_vector: np.ndarray) -> None:
        key = self._quantize(raw_vector)
        self._buckets.setdefault(key, []).append((node_id, raw_vector.copy()))
        self._count += 1

    def lookup(self, query_vector: np.ndarray,
               theta_revisit: float = 0.95) -> List[Tuple[Node, float]]:
        """O(1) ハッシュ + 生コサイン類似度確認。"""
        key = self._quantize(query_vector)
        candidates = self._buckets.get(key, [])
        if not candidates:
            return []

        qn = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        results = []
        for node_id, stored_vec in candidates:
            vn = stored_vec / (np.linalg.norm(stored_vec) + 1e-9)
            sim = float(np.dot(qn, vn))
            if sim >= theta_revisit:
                results.append((node_id, sim))

        results.sort(key=lambda x: -x[1])
        return results

    def lookup_with_neighbors(self, query_vector: np.ndarray,
                               theta_revisit: float = 0.95) -> List[Tuple[Node, float]]:
        """隣接ビンも検索。量子化境界付近のミス防止。O(3^d)。"""
        key = self._quantize(query_vector)
        dim = len(key)
        from itertools import product
        all_candidates = []
        for offset in product([-1, 0, 1], repeat=dim):
            nkey = tuple(k + o for k, o in zip(key, offset))
            all_candidates.extend(self._buckets.get(nkey, []))

        if not all_candidates:
            return []

        qn = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        seen = set()
        results = []
        for node_id, stored_vec in all_candidates:
            if node_id in seen:
                continue
            seen.add(node_id)
            vn = stored_vec / (np.linalg.norm(stored_vec) + 1e-9)
            sim = float(np.dot(qn, vn))
            if sim >= theta_revisit:
                results.append((node_id, sim))

        results.sort(key=lambda x: -x[1])
        return results

    @property
    def size(self) -> int:
        return self._count
```

### 4.2 Layer 1: Attentionグラフ走査

```python
"""
Layer 1: 再訪ノードからattention > θ のエッジを走査。
計算量: O(degree) — 記憶全体に依存しない。
"""

import networkx as nx


class AttentionGraphWalker:
    def __init__(self, theta: float = 0.3, alpha: float = 0.5):
        self.theta = theta     # attention閾値
        self.alpha = alpha     # attention指数

    def get_candidates(
        self,
        graph: nx.Graph,
        revisit_nodes: List[Tuple[Node, float]],
        query_vector: np.ndarray,
        weight_vector: np.ndarray,
    ) -> List[Dict]:
        """再訪ノード接続先から候補抽出。effective_score降順。"""
        candidates = []
        seen = set()

        for revisit_node, raw_sim in revisit_nodes:
            if revisit_node not in graph:
                continue
            for neighbor in graph.neighbors(revisit_node):
                if neighbor in seen:
                    continue
                seen.add(neighbor)

                edge_data = graph[revisit_node][neighbor]
                attention = float(edge_data.get('attention', 0.0))
                if attention < self.theta:
                    continue

                neighbor_vec = (graph.nodes[neighbor].get('abs_vector')
                                or graph.nodes[neighbor].get('vector'))
                if neighbor_vec is None:
                    continue
                neighbor_arr = np.asarray(neighbor_vec, dtype=float)

                w_sim = self._weighted_similarity(
                    query_vector, neighbor_arr, weight_vector
                )
                effective_score = (attention ** self.alpha) * w_sim

                candidates.append({
                    'node_id': neighbor,
                    'attention': attention,
                    'weighted_similarity': w_sim,
                    'effective_score': effective_score,
                    'source_revisit_node': revisit_node,
                    'edge_type': edge_data.get('edge_type', 'unknown'),
                })

        candidates.sort(key=lambda x: -x['effective_score'])
        return candidates

    def _weighted_similarity(self, q, v, w) -> float:
        n = min(len(q), len(v), len(w))
        qw = q[:n] * w[:n]
        vw = v[:n] * w[:n]
        dot = float(np.dot(qw, vw))
        nq = float(np.linalg.norm(qw))
        nv = float(np.linalg.norm(vw))
        if nq < 1e-9 or nv < 1e-9:
            return 0.0
        return max(0.0, dot / (nq * nv))
```

### 4.3 Layer 2: 全記憶ソート（現行ロジック相当）

```python
"""Layer 2: 新規地点のみ実行。現行 weighted_distance + build_ecand と等価。"""


class FullMemorySearch:
    def __init__(self, weight_vector: np.ndarray, top_k: int = 32):
        self.weight_vector = weight_vector
        self.top_k = top_k

    def search(self, query_vector: np.ndarray,
               memory_pool: List[Dict]) -> List[Dict]:
        scored = []
        for item in memory_pool:
            vec = item.get('abs_vector') or item.get('vector')
            if vec is None:
                continue
            dist = self._weighted_distance(query_vector, np.asarray(vec))
            c = dict(item)
            c['weighted_distance'] = dist
            c['effective_score'] = 1.0 / (1.0 + dist)
            scored.append(c)
        scored.sort(key=lambda x: x['weighted_distance'])
        return scored[:self.top_k]

    def _weighted_distance(self, q, v) -> float:
        w = self.weight_vector
        n = min(len(q), len(v), len(w))
        diff = w[:n] * (q[:n] - v[:n])
        return float(np.linalg.norm(diff))
```

### 4.4 三層統合エンジン

```python
"""三層統合コントローラ。Layer 0→1→2、ヒットした層で停止。"""

from dataclasses import dataclass
from typing import Any, Optional
import time


@dataclass
class SearchResult:
    candidates: List[Dict]
    layer_used: int            # 0, 1, 2
    is_revisit: bool
    revisit_similarity: float = 0.0
    search_time_ms: float = 0.0


class ThreeLayerSearchEngine:
    def __init__(
        self, *,
        hash_resolution: float = 0.05,
        theta_revisit: float = 0.95,
        theta_attention: float = 0.3,
        attention_alpha: float = 0.5,
        weight_vector: np.ndarray,
        top_k: int = 32,
        min_layer1_candidates: int = 2,
    ):
        self.hash_index = VectorHashIndex(resolution=hash_resolution)
        self.graph_walker = AttentionGraphWalker(
            theta=theta_attention, alpha=attention_alpha
        )
        self.full_search = FullMemorySearch(
            weight_vector=weight_vector, top_k=top_k
        )
        self.theta_revisit = theta_revisit
        self.min_layer1 = min_layer1_candidates
        self.weight_vector = weight_vector
        self._stats = {'L0': 0, 'L1': 0, 'L2': 0, 'total': 0}

    def search(
        self, query_vector: np.ndarray,
        graph: nx.Graph,
        memory_pool: List[Dict],
    ) -> SearchResult:
        t0 = time.monotonic()
        self._stats['total'] += 1

        # --- Layer 0 ---
        revisit = self.hash_index.lookup(query_vector, self.theta_revisit)

        if revisit:
            # --- Layer 1 ---
            cands = self.graph_walker.get_candidates(
                graph, revisit, query_vector, self.weight_vector
            )
            if len(cands) >= self.min_layer1:
                self._stats['L1'] += 1
                return SearchResult(
                    candidates=cands, layer_used=1, is_revisit=True,
                    revisit_similarity=revisit[0][1],
                    search_time_ms=(time.monotonic() - t0) * 1000,
                )

        # --- Layer 2 ---
        cands = self.full_search.search(query_vector, memory_pool)
        self._stats['L2'] += 1
        return SearchResult(
            candidates=cands, layer_used=2,
            is_revisit=bool(revisit),
            revisit_similarity=revisit[0][1] if revisit else 0.0,
            search_time_ms=(time.monotonic() - t0) * 1000,
        )

    def register(self, node_id: Node, raw_vector: np.ndarray) -> None:
        self.hash_index.add(node_id, raw_vector)

    def get_stats(self) -> Dict[str, Any]:
        t = max(1, self._stats['total'])
        return {
            **self._stats,
            'L1_skip_rate': self._stats['L1'] / t,
        }
```

### 4.5 attention管理

```python
"""エッジのattention管理。生成・減衰・強化・再活性化。"""


class AttentionManager:
    def __init__(self, decay_rate=0.95, use_boost=0.1, theta=0.3):
        self.decay_rate = decay_rate
        self.use_boost = use_boost
        self.theta = theta

    def on_new_edge(self, G, u, v, edge_type="explore"):
        if not G.has_edge(u, v):
            G.add_edge(u, v, attention=1.0, edge_type=edge_type, use_count=0)

    def on_step(self, G):
        """毎ステップ減衰"""
        for u, v, d in G.edges(data=True):
            d['attention'] = d.get('attention', 0.0) * self.decay_rate

    def on_traverse(self, G, u, v):
        """通過時の強化"""
        if G.has_edge(u, v):
            d = G[u][v]
            d['attention'] = min(1.0, d.get('attention', 0.0) + self.use_boost)
            d['use_count'] = d.get('use_count', 0) + 1

    def on_ag_fire(self, G, node) -> List[Tuple[Node, float]]:
        """AG発火: 無意識エッジを再活性化"""
        reactivated = []
        if node not in G:
            return reactivated
        for nb in G.neighbors(node):
            att = G[node][nb].get('attention', 0.0)
            if att < self.theta:
                new_att = self.theta + 0.1
                G[node][nb]['attention'] = new_att
                reactivated.append((nb, new_att))
        return reactivated

    def beta1(self, G, theta=None) -> int:
        theta = theta or self.theta
        edges = [(u,v) for u,v,d in G.edges(data=True) if d.get('attention',0) > theta]
        sub = nx.Graph()
        sub.add_nodes_from(G.nodes())
        sub.add_edges_from(edges)
        return sub.number_of_edges() - sub.number_of_nodes() + nx.number_connected_components(sub)
```

---

## 5. 既存コードへの統合ポイント

| ファイル | 変更内容 |
|---|---|
| `commit.py` | `add_edge()` 時に `attention=1.0` をエッジ属性に追加 |
| `edges.py` | `build_ecand` をラップして三層検索に委譲 |
| `sleep.py` | `build_sleep_beta1()` をQ-learningの代替として追加 |
| `evaluator.py` | **変更なし**（上流で候補が決まるだけ） |
| `models.py` | StepRecord に `layer_used`, `is_revisit` フィールド追加 |

---

## 6. 迷路固有 vs 一般化

| 機能 | 迷路 | 一般RAG |
|---|---|---|
| Layer 0 | 座標ハッシュ | ベクトル量子化 / LSH |
| Layer 1 | nx.Graph | 同じ |
| Layer 2 | SpatialGrid + WeightedL2 | ANN (HNSW等) |
| 再訪判定 | raw_sim > 0.95 | 同じ（閾値はドメイン依存） |

---

## 7. 検証指標

- **Layer 1 ヒット率**: 再訪時にLayer 2をスキップできた割合
- **β₁推移**: 探索↑ Sleep↓ のサイクルが出るか
- **検索時間実測**: Layer 1 vs Layer 2 の速度差
- **候補品質**: effective_score上位の候補が実際に良い選択か
