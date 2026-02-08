# β₁ 実装仕様書（エンジニアリング）

**日付:** 2026-02-07
**目的:** SP項のβ₁代替を段階的に実装するための具体的コード変更仕様
**方針:** SP(ASP)を除去せず、β₁を並行記録 → 相関確認後に切替判断

---

## 0. 前提：二つの独立したSP定義

調査の結果、SP計算が二箇所に存在し、定義が異なる。

| コードベース | ファイル | 現在のSP定義 | 計算量 |
|---|---|---|---|
| 迷路/RAG | `graph_utils.py` → `gedig_core.py` | 平均最短路長(ASP)の変化 | O(V²⁺) |
| Transformer | `metrics.py` | structural probe depth vectors の Spearman相関 | O(T²) |

β₁の導入はそれぞれ独立した作業。本仕様書では両方をカバーする。

---

## Part A: 迷路/RAG コードベース（graph_utils.py, gedig_core.py）

### A1. graph_utils.py — 関数追加

**挿入箇所:** L410 (`compute_ged_min_proxy` の直後、`__all__` の直前)

```python
# --- L410 の後に挿入 ---

def compute_betti_1(g: nx.Graph) -> int:
    """First Betti number: β₁ = E - V + C.

    Counts independent cycles in the graph.
    For connected graphs (C=1), simplifies to E - V + 1.

    Computational cost: O(V+E) general, O(1) if connected.

    Args:
        g: Input graph.

    Returns:
        β₁ (non-negative integer). Returns 0 for empty graphs.
    """
    V = g.number_of_nodes()
    if V == 0:
        return 0
    E = g.number_of_edges()
    C = nx.number_connected_components(g)
    return E - V + C
```

**__all__ 更新:** L413

```python
__all__ = [
    "graph_efficiency",
    "spectral_score",
    "avg_shortest_path_length_safe",
    "compute_sp_gain_norm",
    "extract_k_hop_subgraph",
    "trim_terminal_edges",
    "ensure_networkx",
    "pyg_to_networkx",
    "extract_features",
    "filter_features",
    "compute_ged_min_proxy",
    "compute_betti_1",           # ← NEW
]
```

### A2. gedig_core.py — import追加

**変更箇所:** L24-36

```python
from .gedig.graph_utils import (
    graph_efficiency,
    spectral_score,
    avg_shortest_path_length_safe,
    compute_sp_gain_norm,
    extract_k_hop_subgraph,
    trim_terminal_edges,
    ensure_networkx,
    pyg_to_networkx,
    extract_features,
    filter_features,
    compute_ged_min_proxy,
    compute_betti_1,              # ← NEW
)
```

### A3. types.py — データクラスにフィールド追加

**注意:** types.py は手元にないため、推定構造に基づく。

```python
@dataclass
class HopResult:
    hop: int
    ged: float
    ig: float
    gedig: float
    struct_cost: float
    node_count: int
    edge_count: int
    sp: float
    h_component: float
    ged_raw: float = 0.0
    ged_den: float = 1.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    ig_delta: float = 0.0
    ig_den: float = 1.0
    variance_reduction: float = 0.0
    betti_1: int = 0               # ← NEW: β₁ of g_after at this hop


@dataclass
class GeDIGResult:
    # ... 既存フィールド ...
    delta_sp_rel: float = 0.0
    # ... 他の既存フィールド ...
    betti_1_before: int = 0        # ← NEW
    betti_1_after: int = 0         # ← NEW
    delta_betti_1: int = 0         # ← NEW
```

### A4. gedig_core.py calculate() — β₁計算挿入（single-hop path）

**挿入箇所:** L299 (`g2 = ensure_networkx(g_now)` の直後)

```python
        g1 = ensure_networkx(g_prev)
        g2 = ensure_networkx(g_now)

        # ---- β₁ (first Betti number) ----
        _b1_before = compute_betti_1(g1)
        _b1_after = compute_betti_1(g2)
        _delta_b1 = _b1_after - _b1_before
        # ---- end β₁ ----

        if features_prev is None:
```

**変更箇所:** L450 HopResult生成

```python
            hop0 = HopResult(
                hop=0,
                ged=delta_ged_norm,
                ig=combined_ig,
                gedig=g0_value,
                struct_cost=delta_ged_norm,
                node_count=g2.number_of_nodes(),
                edge_count=g2.number_of_edges(),
                sp=delta_sp_rel,
                h_component=delta_h_norm,
                ged_raw=float(ged_result.get('raw_ged', 0.0)),
                ged_den=float(ged_result.get('normalization_den', denom)),
                entropy_before=float(ig_result.get('entropy_before', 0.0)),
                entropy_after=float(ig_result.get('entropy_after', 0.0)),
                ig_delta=float(ig_result.get('delta_entropy', 0.0)),
                ig_den=float(ig_result.get('normalization_den', ig_fixed_den if ig_fixed_den is not None else 1.0)),
                variance_reduction=float(ig_result.get('variance_reduction', 0.0)),
                betti_1=_b1_after,                # ← NEW
            )
```

**変更箇所:** L468 GeDIGResult生成

```python
            result = GeDIGResult(
                gedig_value=g0_value,
                ged_value=delta_ged_norm,
                ig_value=combined_ig,
                raw_ged=hop0.ged_raw,
                ged_norm_den=hop0.ged_den,
                ig_raw=combined_ig,
                ig_norm_den=hop0.ig_den,
                delta_ged_norm=delta_ged_norm,
                delta_sp_rel=delta_sp_rel,
                delta_h_norm=delta_h_norm,
                structural_cost=float(ged_result.get('structural_cost', delta_ged_norm)),
                structural_improvement=float(ged_result.get('structural_improvement', -delta_ged_norm)),
                information_integration=combined_ig,
                entropy_before=hop0.entropy_before,
                entropy_after=hop0.entropy_after,
                ig_delta=hop0.ig_delta,
                variance_reduction=hop0.variance_reduction,
                computation_time=time.time() - start_time,
                version="onegauge_v1",
                hop_results={0: hop0},
                ged_min_proxy=ged_min_proxy,
                betti_1_before=_b1_before,        # ← NEW
                betti_1_after=_b1_after,           # ← NEW
                delta_betti_1=_delta_b1,           # ← NEW
            )
```

### A5. gedig_core.py calculate() — β₁計算挿入（multi-hop path）

**変更箇所:** L353 `calculate_multihop` 呼び出し部分

multihop.py が手元にないため、以下の方針で対応：

**方針1（推奨）:** `calculate_multihop` の戻り値 `result` に事後的にβ₁を設定

```python
            result = calculate_multihop(
                g1, g2, features_prev, features_now, focal_nodes, start_time,
                # ... 既存パラメータ ...
            )
            # ---- β₁ post-fill for multihop ----
            result.betti_1_before = _b1_before
            result.betti_1_after = _b1_after
            result.delta_betti_1 = _delta_b1
            # ---- end β₁ ----
```

**方針2（完全版）:** `calculate_multihop` 内で各ホップのサブグラフに対して
`compute_betti_1` を呼び、HopResult.betti_1 に設定する。
→ multihop.py にアクセス後に実施。

### A6. テスト

```python
# test_betti_1.py
import networkx as nx
from gedig.graph_utils import compute_betti_1

def test_tree():
    """木のβ₁は0"""
    g = nx.path_graph(5)
    assert compute_betti_1(g) == 0

def test_single_cycle():
    """三角形のβ₁は1"""
    g = nx.cycle_graph(3)
    assert compute_betti_1(g) == 1

def test_complete_graph():
    """K4のβ₁ = 6-4+1 = 3"""
    g = nx.complete_graph(4)
    assert compute_betti_1(g) == 3

def test_benzene():
    """ベンゼン環(6員環)のβ₁は1"""
    g = nx.cycle_graph(6)
    assert compute_betti_1(g) == 1

def test_disconnected():
    """非連結グラフ: 2つの三角形"""
    g = nx.Graph()
    g.add_edges_from([(0,1),(1,2),(2,0)])  # triangle 1
    g.add_edges_from([(3,4),(4,5),(5,3)])  # triangle 2
    # E=6, V=6, C=2 → β₁ = 6-6+2 = 2
    assert compute_betti_1(g) == 2

def test_empty():
    """空グラフ"""
    g = nx.Graph()
    assert compute_betti_1(g) == 0

def test_single_node():
    """孤立ノード"""
    g = nx.Graph()
    g.add_node(0)
    # E=0, V=1, C=1 → β₁ = 0
    assert compute_betti_1(g) == 0

def test_maze_typical():
    """迷路典型ケース: 格子グラフ10x10"""
    g = nx.grid_2d_graph(10, 10)
    # V=100, E=180, C=1 → β₁ = 180-100+1 = 81
    assert compute_betti_1(g) == 81
```

---

## Part B: Transformer コードベース（metrics.py）

### B0. 重要な設計判断

Transformer の現在の SP は `spearman_corr(depth_vectors[l], depth_vectors[l-1])`。
これはグラフ上のASPではなく、構造的プローブの深さベクトル間のSpearman相関。

β₁をTransformerに適用するには、**まずグラフを構築する必要がある**。
候補は以下の3つ：

| 方式 | グラフ構築法 | 利点 | 欠点 |
|---|---|---|---|
| (a) 注意グラフ | attention weight > 閾値の辺 | 直接的 | 閾値依存、ヘッド選択問題 |
| (b) 距離グラフ | structural probe の距離行列を閾値で辺に | SP probe と整合 | 閾値依存 |
| (c) k-NNグラフ | 各トークンの距離上位k個を辺に | 閾値不要 | k の選択 |

**推奨:** (b) 距離グラフ。理由：現在のSPが既に `pairwise_distance_matrix(z_dist)` を
計算しているため（L174-175）、追加計算はほぼゼロ。

### B1. metrics.py — β₁計算関数追加

**挿入箇所:** `compute_layer_curves` の前（L141付近）

```python
def _betti_1_from_distance_matrix(
    dist_mat: torch.Tensor,
    threshold: Optional[float] = None,
    k_neighbors: int = 0,
) -> int:
    """Compute β₁ from a pairwise distance matrix.

    Constructs a graph by connecting pairs with distance < threshold,
    then computes β₁ = E - V + C.

    Args:
        dist_mat: (T, T) pairwise distance matrix.
        threshold: Distance threshold for edge creation.
                   If None, uses median distance.
        k_neighbors: If > 0, use k-NN graph instead of threshold.

    Returns:
        β₁ (non-negative integer).
    """
    n = dist_mat.shape[0]
    if n < 2:
        return 0

    if k_neighbors > 0:
        # k-NN graph: each node connects to k nearest neighbors
        _, indices = torch.topk(dist_mat, k=min(k_neighbors + 1, n), largest=False, dim=1)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in indices[i]:
                j = int(j.item())
                if j != i:
                    G.add_edge(i, j)
    else:
        # Threshold graph
        if threshold is None:
            # Upper triangle values only (avoid diagonal)
            triu_idx = torch.triu_indices(n, n, offset=1)
            vals = dist_mat[triu_idx[0], triu_idx[1]]
            threshold = float(vals.median().item())
        adj = (dist_mat < threshold).int()
        adj.fill_diagonal_(0)
        G = nx.from_numpy_array(adj.cpu().numpy())

    E = G.number_of_edges()
    V = G.number_of_nodes()
    C = nx.number_connected_components(G)
    return E - V + C
```

### B2. metrics.py — LayerCurves にフィールド追加

**変更箇所:** L122-139

```python
@dataclass
class LayerCurves:
    H: List[Optional[float]]
    EPC: List[Optional[float]]
    SP: List[Optional[float]]
    delta_H: List[Optional[float]]
    delta_EPC: List[Optional[float]]
    delta_SP: List[Optional[float]]
    B1: List[Optional[int]] = None          # ← NEW: β₁ per layer
    delta_B1: List[Optional[int]] = None    # ← NEW: Δβ₁ per layer

    def __post_init__(self):
        n = len(self.H)
        if self.B1 is None:
            self.B1 = [None] * n
        if self.delta_B1 is None:
            self.delta_B1 = [None] * n

    def as_dict(self) -> Dict[str, List]:
        return {
            "H": self.H,
            "EPC": self.EPC,
            "SP": self.SP,
            "delta_H": self.delta_H,
            "delta_EPC": self.delta_EPC,
            "delta_SP": self.delta_SP,
            "B1": self.B1,              # ← NEW
            "delta_B1": self.delta_B1,  # ← NEW
        }
```

### B3. metrics.py — compute_layer_curves 内にβ₁計算追加

**変更箇所:** L154-205

```python
def compute_layer_curves(
    hidden_states: Sequence[torch.Tensor],
    unembed_weight: torch.Tensor,
    b_dist: torch.Tensor,
    b_depth: torch.Tensor,
    temperature: float = 1.0,
    vocab_chunk_tokens: int = 8,
    betti_k_neighbors: int = 5,       # ← NEW: k-NN parameter for β₁
) -> LayerCurves:
    """Compute H/EPC/SP/β₁ and their deltas layer by layer."""
    if len(hidden_states) < 2:
        raise ValueError("hidden_states must include at least embedding + 1 layer")

    num_layers = len(hidden_states)
    h_values: List[Optional[float]] = [None] * num_layers
    epc_values: List[Optional[float]] = [None] * num_layers
    sp_values: List[Optional[float]] = [None] * num_layers
    b1_values: List[Optional[int]] = [None] * num_layers       # ← NEW
    delta_h: List[Optional[float]] = [None] * num_layers
    delta_epc: List[Optional[float]] = [None] * num_layers
    delta_sp: List[Optional[float]] = [None] * num_layers
    delta_b1: List[Optional[int]] = [None] * num_layers        # ← NEW

    dist_mats: List[torch.Tensor] = []
    depth_vectors: List[torch.Tensor] = []

    for idx, layer_hidden in enumerate(hidden_states):
        layer_hidden = layer_hidden.to(dtype=torch.float32)
        h_values[idx] = compute_vocab_entropy(
            hidden=layer_hidden,
            unembed_weight=unembed_weight,
            temperature=temperature,
            chunk_tokens=vocab_chunk_tokens,
        )

        z_dist = layer_hidden @ b_dist.t()
        dm = pairwise_distance_matrix(z_dist)
        dist_mats.append(dm)

        z_depth = layer_hidden @ b_depth.t()
        depth = torch.sum(z_depth * z_depth, dim=-1)
        depth_vectors.append(depth)

        # ---- β₁ from distance graph (reuse dm already computed) ----
        b1_values[idx] = _betti_1_from_distance_matrix(
            dm, k_neighbors=betti_k_neighbors,
        )
        # ---- end β₁ ----

        if idx >= 1:
            prev = dist_mats[idx - 1]
            curr = dist_mats[idx]
            epc = _fro_norm(curr - prev) / (_fro_norm(prev) + EPS)
            epc_values[idx] = float(epc.item())

            sp_values[idx] = spearman_corr(depth_vectors[idx], depth_vectors[idx - 1])

            if h_values[idx - 1] is not None and h_values[idx] is not None:
                delta_h[idx] = float(h_values[idx] - h_values[idx - 1])

            # ---- Δβ₁ ----
            if b1_values[idx] is not None and b1_values[idx - 1] is not None:
                delta_b1[idx] = b1_values[idx] - b1_values[idx - 1]
            # ---- end Δβ₁ ----

        if idx >= 2:
            if epc_values[idx] is not None and epc_values[idx - 1] is not None:
                delta_epc[idx] = float(epc_values[idx] - epc_values[idx - 1])
            if sp_values[idx] is not None and sp_values[idx - 1] is not None:
                delta_sp[idx] = float(sp_values[idx] - sp_values[idx - 1])

    return LayerCurves(
        H=h_values,
        EPC=epc_values,
        SP=sp_values,
        delta_H=delta_h,
        delta_EPC=delta_epc,
        delta_SP=delta_sp,
        B1=b1_values,            # ← NEW
        delta_B1=delta_b1,       # ← NEW
    )
```

### B4. run_inference_gedig_v2.py — 記録追加

**変更箇所:** L267-273（curves_by_metric への追加）

```python
        curves_by_metric["H"].append(curves.H)
        curves_by_metric["EPC"].append(curves.EPC)
        curves_by_metric["SP"].append(curves.SP)
        curves_by_metric["delta_H"].append(curves.delta_H)
        curves_by_metric["delta_EPC"].append(curves.delta_EPC)
        curves_by_metric["delta_SP"].append(curves.delta_SP)
        curves_by_metric["F"].append(f_curve)
        curves_by_metric["B1"].append(curves.B1)              # ← NEW
        curves_by_metric["delta_B1"].append(curves.delta_B1)  # ← NEW
```

**変更箇所:** L210-213 初期化部分（defaultdictだが念のため）

```python
        curves_by_metric.setdefault("B1", [])
        curves_by_metric.setdefault("delta_B1", [])
```

### B5. 注意点：Δβ₁ の起点

迷路のΔβ₁は idx >= 1 で計算（前ステップとの差分）。
Transformerでは delta_SP が idx >= 2 で計算されているが、
delta_B1 は idx >= 1 で計算可能（β₁自体が各層で独立に計算されるため）。
上記実装では idx >= 1 で計算している。SPとの相関比較時に注意。

---

## Part C: 相関検証スクリプト

60-seed実験完走後に使う分析スクリプト。

```python
# analyze_betti_sp_correlation.py
"""
60-seed実験結果からdelta_sp_relとdelta_betti_1の相関を分析する。
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_results(result_dir: Path):
    """実験結果JSONからSPとβ₁の時系列を抽出"""
    sp_values = []
    b1_values = []

    for f in sorted(result_dir.glob("*.json")):
        data = json.loads(f.read_text())
        for step in data.get("steps", []):
            sp = step.get("delta_sp_rel")
            b1 = step.get("delta_betti_1")
            if sp is not None and b1 is not None:
                sp_values.append(sp)
                b1_values.append(b1)

    return np.array(sp_values), np.array(b1_values)


def analyze(sp: np.ndarray, b1: np.ndarray):
    """相関分析"""
    print(f"N = {len(sp)}")
    print(f"SP  range: [{sp.min():.4f}, {sp.max():.4f}]")
    print(f"B1  range: [{b1.min()}, {b1.max()}]")
    print()

    # Pearson (線形相関)
    r_pearson, p_pearson = stats.pearsonr(sp, b1)
    print(f"Pearson:  r={r_pearson:.4f}  p={p_pearson:.2e}")

    # Spearman (順位相関)
    r_spearman, p_spearman = stats.spearmanr(sp, b1)
    print(f"Spearman: r={r_spearman:.4f}  p={p_spearman:.2e}")

    print()
    print("=== 判定 ===")
    if abs(r_spearman) > 0.7:
        print(f"|r| = {abs(r_spearman):.3f} > 0.7 → β₁でSP置換可能")
        print("→ 以後の実験はβ₁定義に移行")
    elif abs(r_spearman) > 0.4:
        print(f"|r| = {abs(r_spearman):.3f} ∈ (0.4, 0.7] → 中程度相関")
        print("→ β₁はSPの部分情報。追加項として検討")
    else:
        print(f"|r| = {abs(r_spearman):.3f} < 0.4 → 低相関")
        print("→ β₁はSPと独立情報。F分解の新項候補")


if __name__ == "__main__":
    import sys
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    sp, b1 = load_results(result_dir)
    if len(sp) > 0:
        analyze(sp, b1)
    else:
        print("No data found.")
```

---

## Part D: 変更ファイル一覧

| ファイル | 変更種別 | 変更量 | 優先度 |
|---|---|---|---|
| `graph_utils.py` | 関数追加 + __all__更新 | +20行 | **P0** (即実施可) |
| `gedig_core.py` | import + calculate内β₁記録 | +15行 | **P0** |
| `types.py` | HopResult/GeDIGResult フィールド追加 | +6行 | **P0** |
| `multihop.py` | β₁ post-fill or 各ホップ記録 | +5-15行 | P1 (手元にファイルがない) |
| `metrics.py` | β₁計算関数 + LayerCurves拡張 | +50行 | P1 (Transformer実験開始時) |
| `run_inference_gedig_v2.py` | curves記録追加 | +4行 | P1 |
| `test_betti_1.py` | 新規テスト | +50行 | P0 |
| `analyze_betti_sp_correlation.py` | 新規分析スクリプト | +50行 | P1 (60-seed完走後) |

**P0 合計: 約40行の変更。**

---

## Part E: 実施順序

```
Phase 1: P0実装（即時、実験と並行）
  ├─ graph_utils.py に compute_betti_1 追加
  ├─ types.py に 3フィールド追加
  ├─ gedig_core.py に import + calculate内記録追加
  ├─ test_betti_1.py 作成・実行
  └─ 既存テスト全パスを確認（デフォルト値0なので壊れないはず）

Phase 2: 60-seed完走後
  ├─ 結果データから事後的にβ₁を計算（グラフ構造が保存されている場合）
  │   または次回実験でβ₁記録付きで再走行
  ├─ analyze_betti_sp_correlation.py 実行
  └─ 相関結果に基づき移行判断

Phase 3: Transformer実験v2.1準備
  ├─ metrics.py に _betti_1_from_distance_matrix 追加
  ├─ LayerCurves 拡張
  ├─ run_inference_gedig_v2.py 記録追加
  └─ betti_k_neighbors パラメータのチューニング
```

---

## Part F: 未解決事項

1. **types.py の実物確認が必要。** HopResult/GeDIGResult の実際のフィールド構成を確認後、
   デフォルト値付きフィールドの挿入位置を調整する必要がある可能性。

2. **multihop.py の実物確認が必要。** 各ホップのサブグラフにβ₁を記録する場合、
   サブグラフ抽出ロジックの中に挿入点を特定する必要。

3. **Transformer版のグラフ構築方法の選択。** 距離行列のk-NNグラフを推奨したが、
   k の値による感度分析が必要。k=5をデフォルトとし、k=3,5,7,10で比較推奨。

4. **60-seed実験データのグラフ構造保存状況。** β₁を事後計算できるかは、
   実験ログにG(V,E)が保存されているかに依存。未保存の場合は再走行が必要。
