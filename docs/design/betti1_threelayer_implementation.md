# Betti 数 + 三層記憶検索 実装設計書

**日付:** 2026-02-09（β₀ 拡張: 2026-02-10）
**目的:** Betti 数（β₀, β₁）の導入と三層記憶検索アーキテクチャの実装仕様
**方針:** 既存SP(ASP)を除去せず、β₁を並行記録 → 相関確認後に切替判断
**β₁ステータス:** ✅ 実装済み（`--sp-mode both` で並行記録可能）。ASP vs β₁ 比較検証中。
**β₀ステータス:** 未実装（設計のみ）→ §11 参照
**三層検索ステータス:** ✅ 実装済み（`--search-mode threelayer`）

---

## 1. 変更対象の全体像

| 機能 | 対象コードベース | 変更規模 | 優先度 |
|---|---|---|---|
| SP定義切替 | 迷路/RAG (`cli.py`, `models.py`, `gedig_core.py`) | ~20行 | **P0** |
| β₁計算・記録 | 迷路/RAG (`graph_utils.py`, `gedig_core.py`, `types.py`) | ~40行 | **P0** |
| V/Eステップログ | 迷路 (`run_experiment_query.py`, `models.py`) | ~15行 | **P0** |
| β₁計算・記録 | Transformer (`metrics.py`, `run_inference_gedig_v2.py`) | ~60行 | P1 |
| 三層記憶検索 | 迷路 (`run_experiment_query.py`, `edges.py`, `models.py`) | ~300行(新規) + ~50行(既存修正) | **P0** |
| 相関検証 | 分析スクリプト | ~60行(新規) | P1 |

---

## 2. SP定義の切替config

### 2.1 問題

現状、SP項は ASP（平均最短路長）固定。β₁ への切替・並行記録を制御するconfigが存在しない。

### 2.2 CLI フラグ

```python
parser.add_argument("--sp-mode",
                    choices=["asp", "betti1", "both"],
                    default="asp",
                    help="SP definition: asp (default), betti1, or both (parallel recording)")
```

| モード | 動作 | F分解での使用 | ログ出力 |
|---|---|---|---|
| `asp` | 現行動作（デフォルト） | SP = ASP | `delta_sp_series` のみ |
| `betti1` | β₁ のみ計算 | SP = Δβ₁ | `betti1_series` のみ |
| `both` | 両方計算 | SP = ASP（互換） | `delta_sp_series` + `betti1_series` 両方 |

### 2.3 QueryHubConfig

```python
sp_mode: str = "asp"  # "asp" | "betti1" | "both"
```

### 2.4 gedig_core.py calculate() 内の分岐

```python
if sp_mode in ("asp", "both"):
    delta_sp_rel = compute_sp_gain_norm(g1, g2)  # 既存ロジック
if sp_mode in ("betti1", "both"):
    _b1_before = compute_betti_1(g1)
    _b1_after = compute_betti_1(g2)
    _delta_b1 = _b1_after - _b1_before
```

`sp_mode="asp"` では β₁ 計算をスキップ → 既存実験と完全同一動作。

### 2.5 既存ログからのβ₁事後計算

**現状: 不可能。** 既存ログには以下の制約がある:

- `edges` はrun単位の最終値のみ（ステップごとのV/Eがない）
- HopResult 内部の `node_count`, `edge_count` はJSON出力に含まれていない
- unified10d 実験では `--persist-graph-sqlite` 未使用（SQLiteファイルなし）

**対策:** 次節のV/Eステップログを追加し、今後の実験ではβ₁を事後計算可能にする。

---

## 3. V/E ステップログの追加

### 3.1 StepRecord に追加

```python
@dataclass
class StepRecord:
    # ... 既存フィールド ...
    graph_node_count: int = 0     # V at this step
    graph_edge_count: int = 0     # E at this step
    betti_1: int = 0              # β₁ = E - V + 1 (connected graph)
```

### 3.2 JSON出力に追加

run 単位の series に追加:

```python
"node_count_series": [2, 4, 6, 8, ...],
"edge_count_series": [1, 3, 5, 7, ...],
"betti1_series": [0, 0, 0, 0, ...],
```

### 3.3 記録タイミング

`run_experiment_query.py` の毎ステップ、commit後に:

```python
step_record.graph_node_count = inherited_graph.number_of_nodes()
step_record.graph_edge_count = inherited_graph.number_of_edges()
step_record.betti_1 = step_record.graph_edge_count - step_record.graph_node_count + 1
```

迷路グラフは常に連結（C=1）なので `+ 1` 固定。

---

## 4. β₁ 実装

### 4.1 定義

```
β₁ = E - V + C
```

- E: 辺数、V: 頂点数、C: 連結成分数
- 連結グラフ（迷路等）では C=1 固定 → β₁ = E - V + 1
- 計算量: O(1)（連結グラフ）、O(V+E)（一般グラフ）

### 4.2 迷路/RAG: graph_utils.py

**挿入箇所:** `compute_ged_min_proxy` の直後

```python
def compute_betti_1(g: nx.Graph) -> int:
    """First Betti number: β₁ = E - V + C."""
    V = g.number_of_nodes()
    if V == 0:
        return 0
    E = g.number_of_edges()
    C = nx.number_connected_components(g)
    return E - V + C
```

`__all__` に `"compute_betti_1"` を追加。

### 4.3 迷路/RAG: types.py

```python
@dataclass
class HopResult:
    # ... 既存フィールド ...
    betti_1: int = 0               # β₁ of g_after at this hop

@dataclass
class GeDIGResult:
    # ... 既存フィールド ...
    betti_1_before: int = 0
    betti_1_after: int = 0
    delta_betti_1: int = 0
```

### 4.4 迷路/RAG: gedig_core.py

**import追加:**

```python
from .gedig.graph_utils import (
    # ... 既存 ...
    compute_betti_1,
)
```

**calculate() 内、`g2 = ensure_networkx(g_now)` の直後:**

```python
_b1_before = compute_betti_1(g1)
_b1_after = compute_betti_1(g2)
_delta_b1 = _b1_after - _b1_before
```

HopResult 生成時に `betti_1=_b1_after`、GeDIGResult 生成時に `betti_1_before`, `betti_1_after`, `delta_betti_1` を設定。

multihop path では `calculate_multihop` の戻り値に事後設定:

```python
result = calculate_multihop(...)
result.betti_1_before = _b1_before
result.betti_1_after = _b1_after
result.delta_betti_1 = _delta_b1
```

### 4.5 Transformer: metrics.py

距離行列から k-NN グラフを構築し β₁ を計算。

```python
def _betti_1_from_distance_matrix(
    dist_mat: torch.Tensor,
    k_neighbors: int = 5,
) -> int:
    """Compute β₁ from pairwise distance matrix via k-NN graph."""
    n = dist_mat.shape[0]
    if n < 2:
        return 0
    _, indices = torch.topk(dist_mat, k=min(k_neighbors + 1, n), largest=False, dim=1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in indices[i]:
            j = int(j.item())
            if j != i:
                G.add_edge(i, j)
    E = G.number_of_edges()
    V = G.number_of_nodes()
    C = nx.number_connected_components(G)
    return E - V + C
```

LayerCurves に `B1: List[Optional[int]]` と `delta_B1: List[Optional[int]]` を追加。
`compute_layer_curves` 内で各層の `dm` を再利用して `_betti_1_from_distance_matrix(dm)` を呼ぶ。
`run_inference_gedig_v2.py` の `curves_by_metric` に `"B1"`, `"delta_B1"` を追加。

### 4.6 テスト

```python
import networkx as nx
from gedig.graph_utils import compute_betti_1

def test_tree():
    assert compute_betti_1(nx.path_graph(5)) == 0

def test_cycle():
    assert compute_betti_1(nx.cycle_graph(3)) == 1

def test_complete_k4():
    # K4: E=6, V=4, C=1 → β₁=3
    assert compute_betti_1(nx.complete_graph(4)) == 3

def test_grid_10x10():
    # V=100, E=180, C=1 → β₁=81
    assert compute_betti_1(nx.grid_2d_graph(10, 10)) == 81

def test_disconnected():
    g = nx.Graph()
    g.add_edges_from([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)])
    # E=6, V=6, C=2 → β₁=2
    assert compute_betti_1(g) == 2

def test_empty():
    assert compute_betti_1(nx.Graph()) == 0
```

---

## 5. 三層記憶検索

### 5.1 設計原理

```
計算量 ∝ 不確実性
```

| Layer | 判定 | 計算量 | 条件 |
|---|---|---|---|
| Layer 0 | ハッシュ完全一致（再訪判定） | O(1) | 既知地点 |
| Layer 1 | グラフ走査（attention参照） | O(degree) | 再訪 + 隣接に候補あり |
| Layer 2 | 全記憶ソート（重み付き類似度） | O(N log N) | 新規地点 or Layer 1 候補不足 |

**二種類の類似度を使い分ける:**

| 類似度 | 用途 | 理由 |
|---|---|---|
| 生類似度（等重み） | Layer 0 再訪判定 | 方向やタスク情報をマスクしないため同一地点を正確に認識 |
| 重み付き類似度 | Layer 1-2 候補選択 | 探索方向を考慮した候補ランキング |

### 5.2 Layer 0: 量子化ハッシュインデックス

```python
class VectorHashIndex:
    """生ベクトルの量子化によるO(1)再訪判定。"""

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

    @property
    def size(self) -> int:
        return self._count
```

**迷路での resolution:** `1 / maze_size`（セル1個分）。
**ハッシュ対象次元:** dim0-1（座標）のみで量子化し、高次元の 3^d 問題を回避。

### 5.3 Layer 1: Attention グラフ走査

```python
class AttentionGraphWalker:
    """再訪ノードの隣接エッジから attention > θ の候補を抽出。"""

    def __init__(self, theta: float = 0.3, alpha: float = 0.5):
        self.theta = theta
        self.alpha = alpha

    def get_candidates(
        self,
        graph: nx.Graph,
        revisit_nodes: List[Tuple[Node, float]],
        query_vector: np.ndarray,
        weight_vector: np.ndarray,
    ) -> List[Dict]:
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
                    query_vector, neighbor_arr, weight_vector)
                effective_score = (attention ** self.alpha) * w_sim
                candidates.append({
                    'node_id': neighbor,
                    'attention': attention,
                    'weighted_similarity': w_sim,
                    'effective_score': effective_score,
                    'source_revisit_node': revisit_node,
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

### 5.4 Layer 2: 全記憶ソート（現行ロジック相当）

```python
class FullMemorySearch:
    """新規地点のみ実行。現行 weighted_distance + build_ecand と等価。"""

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

### 5.5 三層統合エンジン

```python
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
            theta=theta_attention, alpha=attention_alpha)
        self.full_search = FullMemorySearch(
            weight_vector=weight_vector, top_k=top_k)
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

        # --- Layer 0: ハッシュ再訪判定 ---
        revisit = self.hash_index.lookup(query_vector, self.theta_revisit)

        if revisit:
            # --- Layer 1: グラフ走査 ---
            cands = self.graph_walker.get_candidates(
                graph, revisit, query_vector, self.weight_vector)
            if len(cands) >= self.min_layer1:
                self._stats['L1'] += 1
                return SearchResult(
                    candidates=cands, layer_used=1, is_revisit=True,
                    revisit_similarity=revisit[0][1],
                    search_time_ms=(time.monotonic() - t0) * 1000)

        # --- Layer 2: 全記憶ソート ---
        cands = self.full_search.search(query_vector, memory_pool)
        self._stats['L2'] += 1
        return SearchResult(
            candidates=cands, layer_used=2,
            is_revisit=bool(revisit),
            revisit_similarity=revisit[0][1] if revisit else 0.0,
            search_time_ms=(time.monotonic() - t0) * 1000)

    def register(self, node_id: Node, raw_vector: np.ndarray) -> None:
        self.hash_index.add(node_id, raw_vector)

    def get_stats(self) -> Dict[str, Any]:
        t = max(1, self._stats['total'])
        return {**self._stats, 'L1_skip_rate': self._stats['L1'] / t}
```

### 5.6 Attention 管理

```python
class AttentionManager:
    def __init__(self, decay_rate=0.95, use_boost=0.1, theta=0.3):
        self.decay_rate = decay_rate
        self.use_boost = use_boost
        self.theta = theta

    def on_new_edge(self, G, u, v, edge_type="explore"):
        if not G.has_edge(u, v):
            G.add_edge(u, v, attention=1.0, edge_type=edge_type, use_count=0)

    def on_step(self, G):
        """毎ステップ全エッジ減衰"""
        for u, v, d in G.edges(data=True):
            d['attention'] = d.get('attention', 0.0) * self.decay_rate

    def on_traverse(self, G, u, v):
        """通過時の強化"""
        if G.has_edge(u, v):
            d = G[u][v]
            d['attention'] = min(1.0, d.get('attention', 0.0) + self.use_boost)
            d['use_count'] = d.get('use_count', 0) + 1

    def on_ag_fire(self, G, node) -> List[Tuple]:
        """AG発火時: θ未満のエッジを再活性化"""
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
```

---

## 6. 既存コードへの統合

### 6.1 迷路実験 (`run_experiment_query.py`)

| 統合箇所 | 変更内容 |
|---|---|
| 初期化 | `ThreeLayerSearchEngine` を生成、`hash_resolution = 1/maze_size` |
| commit 時 | `search_engine.register(node_id, raw_vector)` でハッシュ登録 |
| `add_edge()` 時 | エッジ属性に `attention=1.0` を追加 |
| obs候補構築 | `search_engine.search()` で Layer 判定、Layer 1 ヒット時は `build_ecand` をスキップ |
| memory候補構築 | 同上 |
| 毎ステップ | `attention_manager.on_step(inherited_graph)` で減衰 |
| 移動時 | `attention_manager.on_traverse(G, prev, curr)` で強化 |
| AG発火時 | `attention_manager.on_ag_fire(G, node)` で再活性化 |
| StepRecord | `layer_used`, `is_revisit`, `search_time_ms` フィールド追加 |

### 6.2 新規ファイル構成

```
experiments/maze/
  qhlib/
    three_layer_search.py    # VectorHashIndex, AttentionGraphWalker,
                              # FullMemorySearch, ThreeLayerSearchEngine
    attention_manager.py      # AttentionManager
  tests/
    test_betti_1.py
    test_three_layer_search.py
```

---

## 7. パラメータ一覧

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `sp_mode` | `"asp"` | SP定義: `asp`(現行), `betti1`, `both`(並行記録) |
| `hash_resolution` | `1 / maze_size` | Layer 0 量子化解像度 |
| `theta_revisit` | 0.95 | Layer 0 再訪判定の生類似度閾値 |
| `theta_attention` | 0.3 | Layer 1 エッジの attention 閾値 |
| `attention_alpha` | 0.5 | Layer 1 effective_score の attention 指数 |
| `min_layer1_candidates` | 2 | Layer 1 → Layer 2 フォールスルー条件 |
| `decay_rate` | 0.95 | 毎ステップの attention 減衰率 |
| `use_boost` | 0.1 | 通過時の attention 増加量 |
| `betti_k_neighbors` | 5 | Transformer 版: k-NN グラフの k |

---

## 8. 検証指標

| 指標 | 目的 |
|---|---|
| Layer 1 ヒット率 | 再訪時に Layer 2 をスキップできた割合 |
| β₁ vs ASP 相関 | Spearman \|r\| > 0.7 で置換可能と判断 |
| 検索時間 | Layer 1 vs Layer 2 の実測速度差 |
| 成功率変化 | 三層導入前後の 60-seed eval 成功率比較 |

---

## 9. 後方互換性

### 9.1 互換性の原則

**三層検索はCLIフラグによるopt-in。フラグなし = 現行動作（Layer 2 のみ）。**

```
--search-mode legacy     → 現行ロジックそのまま（デフォルト）
--search-mode three-layer → 三層検索有効
```

β₁ は並行記録のみ。既存の SP(ASP) 計算はそのまま残す。

### 9.2 変更箇所ごとの互換性評価

| 変更箇所 | 互換性 | 理由 |
|---|---|---|
| `types.py` HopResult | **安全** | `betti_1: int = 0` デフォルト付き。既存コードは無影響 |
| `types.py` GeDIGResult | **安全** | 3フィールド全てデフォルト=0。既存の `gedig_value` 等は不変 |
| `graph_utils.py` | **安全** | 関数追加 + `__all__` 追記のみ。既存関数は不変 |
| `gedig_core.py` | **安全** | β₁計算を追加するが、`calculate()` の戻り値型は同一（GeDIGResult） |
| `models.py` StepRecord | **安全** | 新フィールドはデフォルト付き。既存182フィールドは不変 |
| `models.py` QueryHubConfig | **安全** | 新フィールドはデフォルト付き。74/78フィールドが既にデフォルト持ち |
| `cli.py` | **安全** | 新CLIフラグは全てデフォルト値を持つ。既存100+フラグは不変 |
| SQLite スキーマ | **安全** | `attributes` カラムがJSON TEXT → 任意キー追加にスキーマ変更不要 |
| `inherited_graph` | **安全** | ノード属性は `dict.get()` で動的参照。新キー追加は既存参照に無影響 |

### 9.3 新規CLIフラグ

`qhlib/cli.py` に追加。全てデフォルト値付き。

```python
# --- SP定義 ---
parser.add_argument("--sp-mode",
                    choices=["asp", "betti1", "both"],
                    default="asp",
                    help="SP definition mode (default: asp = current behavior)")

# --- 三層検索 ---
parser.add_argument("--search-mode",
                    choices=["legacy", "three-layer"],
                    default="legacy",
                    help="Memory search mode (default: legacy)")
parser.add_argument("--theta-revisit", type=float, default=0.95,
                    help="Layer 0 revisit similarity threshold")
parser.add_argument("--theta-attention", type=float, default=0.3,
                    help="Layer 1 attention edge threshold")
parser.add_argument("--attention-alpha", type=float, default=0.5,
                    help="Layer 1 effective_score attention exponent")
parser.add_argument("--attention-decay", type=float, default=0.95,
                    help="Per-step attention decay rate")
parser.add_argument("--attention-boost", type=float, default=0.1,
                    help="On-traverse attention boost")
parser.add_argument("--min-layer1-candidates", type=int, default=2,
                    help="Min candidates for Layer 1 to avoid fallthrough")

# --- β₁ ---
parser.add_argument("--record-betti1", action="store_true", default=False,
                    help="Record β₁ alongside SP (parallel recording)")
```

### 9.4 run_experiment_query.py での分岐

```python
if config.search_mode == "three-layer":
    search_engine = ThreeLayerSearchEngine(
        hash_resolution=1.0 / config.maze_size,
        theta_revisit=config.theta_revisit,
        theta_attention=config.theta_attention,
        attention_alpha=config.attention_alpha,
        weight_vector=weight_vec,
        top_k=config.candidate_cap,
        min_layer1_candidates=config.min_layer1_candidates,
    )
    attention_mgr = AttentionManager(
        decay_rate=config.attention_decay,
        use_boost=config.attention_boost,
        theta=config.theta_attention,
    )
else:
    search_engine = None
    attention_mgr = None
```

候補構築時:

```python
if search_engine is not None:
    result = search_engine.search(query_vec, inherited_graph, memory_pool)
    if result.layer_used <= 1:
        # Layer 0/1 ヒット → build_ecand スキップ
        candidates = result.candidates
    else:
        # Layer 2 フォールスルー → 現行ロジック
        candidates = build_ecand(...)  # 既存コードそのまま
else:
    # legacy mode → 現行ロジックそのまま
    candidates = build_ecand(...)
```

### 9.5 SQLite エッジ属性

既存の `attributes` カラム（JSON TEXT）にそのまま追加。スキーマ変更不要。

```python
# 既存: {"stage": "timeline", "nodes": [[r,c,d], [r',c',d']]}
# 追加: {"stage": "timeline", "nodes": [...], "attention": 1.0}
```

読み込み時は `dict.get("attention", 1.0)` でデフォルト付き参照。
過去のSQLiteデータに `attention` キーがなくても安全。

### 9.6 gedig_core.py の互換保証

`calculate()` は既に後方互換のための引数マッピングを持っている（positional → keyword）。
β₁フィールドは GeDIGResult にデフォルト=0 で追加するため:

- 既存の `result.gedig_value`, `result.hop_results` は不変
- `result.betti_1_before` 等は参照しなければ 0 のまま
- archive 実験（`maze-query-hub-prototype`, `hotpotqa-benchmark` 等）も影響なし

### 9.7 テスト互換性

β₁フィールドのデフォルト=0 により、既存テストの GeDIGResult/HopResult アサーションは全てパスする。
新テスト (`test_betti_1.py`, `test_three_layer_search.py`) は独立ファイルで追加。

---

## 10. 実施順序

```
Phase 1: β₁ 並行記録（即時）
  ├─ graph_utils.py に compute_betti_1 追加
  ├─ types.py にフィールド追加
  ├─ gedig_core.py に記録追加
  └─ test_betti_1.py 作成・実行

Phase 2: 三層検索の実装（β₁と並行）
  ├─ three_layer_search.py 新規作成
  ├─ attention_manager.py 新規作成
  ├─ run_experiment_query.py への統合
  └─ test_three_layer_search.py 作成・実行

Phase 3: 60-seed 実験
  ├─ 三層検索 + β₁記録付きで走行
  ├─ Layer 1 ヒット率・成功率・β₁推移を分析
  └─ β₁ vs ASP 相関検証 → 移行判断

Phase 4: Transformer 版（Phase 3 完了後）
  ├─ metrics.py に β₁ 計算追加
  ├─ LayerCurves 拡張
  └─ k パラメータ感度分析（k=3,5,7,10）
```

---

---

## 11. β₀（連結成分数）の昇格

**追記日:** 2026-02-10
**動機:** HotPotQA v2 実験（`experiments/hotpotqa_v2/SPEC.md`）

### 11.1 動機

β₁（独立閉路数）は連結グラフ内のループを検出するが、**非連結グラフにおける島（connected components）の統合を検出できない**。

RAG のナレッジグラフでは、検索結果が複数の孤立クラスタを形成することが一般的：

```
検索前:  [Q]                           β₀ = 1, β₁ = 0

検索後:  [Q]---[Doc A]   [Doc B]---[Doc C]   β₀ = 2, β₁ = 0
         ↑ 接続済み        ↑ 孤立した島

橋渡し:  [Q]---[Doc A]---[Doc B]---[Doc C]   β₀ = 1, β₁ = 0
                      ↑ 島が統合（Δβ₀ = -1）
```

HotPotQA の **bridge 型質問**は、まさに 2 つの独立文書の橋渡しを要求する。β₀ の減少はこの構造変化を直接捉える。

**迷路実験との互換:** 迷路グラフは常に連結（β₀ = 1）のため、Δβ₀ = 0 → β₀ 項は自動的に消滅し、既存結果に影響しない。

### 11.2 拡張ゲージ v5

```
F = ΔEPC_norm − λ ( ΔH_norm + γ₁·Δβ₁ − γ₀·Δβ₀ )
```

| イベント | Δβ₀ | Δβ₁ | 寄与 (−γ₀Δβ₀ + γ₁Δβ₁) | F への効果 |
|---------|---:|---:|:----------------------:|:---------:|
| 島の統合 | -1 | 0 | +γ₀ | F 減少（良い） |
| ループ生成 | 0 | +1 | +γ₁ | F 減少（良い） |
| 孤立島の追加 | +1 | 0 | -γ₀ | F 増加（悪い） |
| ループ破壊 | 0 | -1 | -γ₁ | F 増加（悪い） |

`structural_mode` パラメータで切替：

| モード | ゲージ式 | 用途 |
|--------|---------|------|
| `"sp"` | F = ΔEPC − λ(ΔH + γ·ΔSP) | v4 互換（デフォルト） |
| `"betti"` | F = ΔEPC − λ(ΔH + γ₁·Δβ₁) | β₁ のみ |
| `"betti_full"` | F = ΔEPC − λ(ΔH + γ₁·Δβ₁ − γ₀·Δβ₀) | フル Betti |

### 11.3 実装変更

#### graph_utils.py: `compute_betti_numbers()`

```python
def compute_betti_numbers(g: nx.Graph) -> tuple[int, int]:
    """Betti numbers β₀ (connected components) and β₁ (independent cycles).

    β₀ = number of connected components
    β₁ = E - V + β₀  (Euler relation for graphs)

    Returns: (β₀, β₁)
    Cost: O(V+E) for β₀ via union-find, O(1) for β₁ given β₀.
    """
    V = g.number_of_nodes()
    if V == 0:
        return (0, 0)
    E = g.number_of_edges()
    beta_0 = nx.number_connected_components(g)
    beta_1 = E - V + beta_0
    return (beta_0, beta_1)
```

既存の `compute_betti_1()` は互換のため残し、内部で `compute_betti_numbers()` を呼ぶ。

#### types.py: GeDIGResult 拡張

```python
@dataclass
class GeDIGResult:
    # ... 既存フィールド ...
    betti_0_before: int = 0    # NEW
    betti_0_after: int = 0     # NEW
    delta_betti_0: int = 0     # NEW
    betti_1_before: int = 0    # 既存
    betti_1_after: int = 0     # 既存
    delta_betti_1: int = 0     # 既存
```

#### config: 新規パラメータ

```python
structural_mode: str = "sp"    # "sp" | "betti" | "betti_full"
gamma_0: float = 0.0           # β₀ weight (default 0 = backward compat)
gamma_1: float = 0.0           # β₁ weight (default 0 = use sp_beta with ΔSP)
```

- `structural_mode="sp"`: 現行動作。`sp_beta` と `ΔSP_rel` を使用
- `structural_mode="betti"`: `gamma_1` と `Δβ₁` を使用
- `structural_mode="betti_full"`: `gamma_0`, `gamma_1` と `Δβ₀`, `Δβ₁` を使用

#### gedig_core.py: F 計算の分岐

```python
if structural_mode == "sp":
    # v4 互換
    f_value = delta_ged_norm - lambda_weight * (delta_h_norm + sp_beta * delta_sp_rel)
elif structural_mode == "betti":
    f_value = delta_ged_norm - lambda_weight * (delta_h_norm + gamma_1 * delta_beta_1)
elif structural_mode == "betti_full":
    topo_term = gamma_1 * delta_beta_1 - gamma_0 * delta_beta_0
    f_value = delta_ged_norm - lambda_weight * (delta_h_norm + topo_term)
```

### 11.4 テスト

```python
def test_betti_numbers_empty():
    assert compute_betti_numbers(nx.Graph()) == (0, 0)

def test_betti_numbers_tree():
    # Path: V=5, E=4, C=1 → β₀=1, β₁=0
    assert compute_betti_numbers(nx.path_graph(5)) == (1, 0)

def test_betti_numbers_cycle():
    # Cycle: V=4, E=4, C=1 → β₀=1, β₁=1
    assert compute_betti_numbers(nx.cycle_graph(4)) == (1, 1)

def test_betti_numbers_two_islands():
    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2)])       # Island 1: path
    g.add_edges_from([(3,4), (4,5), (5,3)]) # Island 2: triangle
    # V=6, E=5, C=2 → β₀=2, β₁=1
    assert compute_betti_numbers(g) == (2, 1)

def test_island_merge_decreases_f():
    """β₀ 減少（島統合）→ F 減少を確認"""
    # g_before: 2 islands → g_after: 1 island (bridge added)
    # Δβ₀ = -1 → topo_term に +γ₀ → F 減少
    ...

def test_backward_compat_sp_mode():
    """structural_mode='sp' で既存結果と一致"""
    ...

def test_connected_graph_beta0_neutral():
    """連結グラフでは Δβ₀ = 0 → γ₀ 項消滅"""
    ...
```

### 11.5 後方互換性

| 変更箇所 | 互換性 | 理由 |
|---------|--------|------|
| `compute_betti_numbers()` | **安全** | 新規関数追加のみ |
| `compute_betti_1()` | **安全** | 内部実装変更のみ、シグネチャ不変 |
| `GeDIGResult` β₀ フィールド | **安全** | デフォルト=0、既存コード無影響 |
| `structural_mode` | **安全** | デフォルト=`"sp"`で現行動作 |
| `gamma_0`, `gamma_1` | **安全** | デフォルト=0.0 |

### 11.6 gedig_spec.md v5 更新

```markdown
## ゲージ v5（Betti 拡張）

F = ΔEPC_norm − λ ( ΔH_norm + γ₁·Δβ₁ − γ₀·Δβ₀ )

β₀: 連結成分数（島の数）
β₁: 独立閉路数（ループの数）
β₁ = E − V + β₀（グラフのオイラー関係）

structural_mode:
- "sp":         F = ΔEPC − λ(ΔH + γ·ΔSP)           ← v4 互換
- "betti":      F = ΔEPC − λ(ΔH + γ₁·Δβ₁)          ← β₁ のみ
- "betti_full": F = ΔEPC − λ(ΔH + γ₁·Δβ₁ − γ₀·Δβ₀) ← フル Betti
```

---

## 12. 実施順序（更新）

```
Phase 1: β₁ 並行記録 ✅
  ├─ graph_utils.py に compute_betti_1 追加
  ├─ types.py にフィールド追加
  ├─ gedig_core.py に記録追加
  └─ test_betti_1.py 作成・実行

Phase 2: 三層検索の実装 ✅
  ├─ hash_index.py, graph_walker.py, attention.py, search_engine.py
  ├─ run_experiment_query.py への統合
  └─ test_threelayer.py 作成・実行（23テスト）

Phase 3: β₀ 昇格（NEW）
  ├─ graph_utils.py に compute_betti_numbers 追加
  ├─ types.py に β₀ フィールド追加
  ├─ config に structural_mode, gamma_0, gamma_1 追加
  ├─ gedig_core.py の F 計算を structural_mode で分岐
  ├─ テスト追加（島統合・後方互換・連結グラフ中立性）
  └─ gedig_spec.md を v5 に更新

Phase 4: HotPotQA v2 実験
  ├─ experiments/hotpotqa_v2/ にメインコード GeDIGCore を使う adapter
  ├─ 5条件比較（SP / β₁ / β₀ / β₀+β₁ / tuned）
  └─ bridge 型質問での β₀ 効果検証

Phase 5: Transformer 版（Phase 3 完了後）
  ├─ metrics.py に β₁ 計算追加
  ├─ LayerCurves 拡張
  └─ k パラメータ感度分析（k=3,5,7,10）
```

---

*Generated: 2026-02-09, Updated: 2026-02-10 (§11-12 β₀ extension)*
