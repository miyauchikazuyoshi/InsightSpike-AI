# β₁ルーティング：ナビゲーション応用メモ

## 2026-02-08

---

## 1. 要旨

道路ネットワーク上の経路探索を、Betti数β₁とattention重みで階層化する。
Contraction Hierarchies (CH) の代替。速度は同等、**運用負荷が桁違いに小さい**。

---

## 2. 現行手法 (CH) の問題

| 項目 | コスト |
|---|---|
| 前計算 | 全米道路グラフで数時間〜数日 |
| メモリ | 元グラフの10〜20倍（ショートカット辺保持） |
| 更新 | 道路1本変更 → 最悪全再計算 |
| 階層定義 | ヒューリスティック（edge difference等の経験的指標） |
| エッジ実行 | 不可能（リソース要件が大きすぎる） |

**クエリは速い。そのO(1)を維持するインフラが巨大。**

---

## 3. β₁ルーティングの核心

### 3.1 道路の「格」= β₁密度

```
路地:   β₁/V 高い → ループだらけ → 代替経路多い
国道:   β₁/V 中   → 適度なループ → いくつかの代替
高速:   β₁/V 低い → ほぼ一本道   → 代替ほぼなし
```

定義から出る。ヒューリスティックではない。

### 3.2 相転移がレイヤー境界を自動決定

attention閾値θを0→1に掃引すると：

```
θ = 0.0  → 全道路    → β₁ 最大
θ ↑      → 路地消滅  → β₁ 急落 ← 第一相転移
θ ↑↑     → 国道消滅  → β₁ 急落 ← 第二相転移
θ ↑↑↑    → 高速のみ  → β₁ ≈ 0
```

**相転移点 = レイヤー境界。** トポロジーから自動導出。

### 3.3 三層検索の適用

```
[Layer 0] キャッシュルート完全一致 O(1)
[Layer 1] 高速道路相（β₁≈0）→ 経路ほぼ一意 → O(少)
[Layer 2] 国道相 → 少数候補 → attentionで選択 → O(中)
[Layer 3] 路地相 → 最後の数百m → 範囲が狭いからNが小 → O(小N)
```

各層の計算量がβ₁に比例する。

---

## 4. CHとの比較

| | CH | β₁ルーティング |
|---|---|---|
| 階層定義 | ヒューリスティック | **β₁密度の相転移（理論的）** |
| 前計算 | O(N log N) 全グラフ | O(E) θ掃引（並列化容易） |
| メモリ | 10-20x | **1.x倍（attention float/エッジ）** |
| 更新 | 大規模再計算 | **局所β₁再計算のみ** |
| クエリ速度 | ◎ O(log N) | ○ O(β₁ × layers) |
| エッジ実行 | 不可能 | **可能** |

**速度はCHに劣るかもしれない。しかし総負荷が桁違いに小さい。**

---

## 5. リアルタイム交通対応

### 渋滞 = attention減衰

```
渋滞発生 → attention低下 → θ以下 → 意識から消える
         → 自動的に代替経路にフォールバック

渋滞解消 → attention回復 → 元の経路が意識に戻る
```

ルート再計算ではない。attention閾値で自動切替。

### 通行止め = attention即時ゼロ化

```
通行止め → attention = 0
         → 局所β₁再計算（その辺を含むサイクルのみ）
         → 影響範囲のルートだけ更新
```

CHなら構造変更→ショートカット辺全面再計算。β₁なら重み変更→局所再計算。

---

## 6. 実装コード

### 6.1 道路グラフ構築

```python
"""OSMデータからattention付き道路グラフを構築"""

import math
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any


def build_road_graph(osm_ways: List[Dict], node_coords: Dict[int, Tuple[float, float]]) -> nx.Graph:
    """各エッジにattention=1.0を初期設定。"""
    G = nx.Graph()

    speed_table = {
        "motorway": 100, "motorway_link": 60,
        "trunk": 80, "trunk_link": 50,
        "primary": 60, "primary_link": 40,
        "secondary": 50, "tertiary": 40,
        "residential": 30, "unclassified": 30,
        "service": 20, "living_street": 10,
    }

    for way in osm_ways:
        nodes = way.get("nodes", [])
        tags = way.get("tags", {})
        road_type = tags.get("highway", "unclassified")
        speed = float(speed_table.get(road_type, 30))
        lanes = int(tags.get("lanes", 1))

        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            length = _haversine(node_coords[u], node_coords[v])
            travel_time = length / (speed / 3.6) if speed > 0 else 1e9

            G.add_edge(u, v,
                road_type=road_type,
                length_m=length,
                speed_kmh=speed,
                lanes=lanes,
                attention=1.0,
                travel_time_s=travel_time,
                base_travel_time_s=travel_time,
            )
    return G


def _haversine(c1: Tuple[float,float], c2: Tuple[float,float]) -> float:
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
```

### 6.2 θ掃引によるレイヤー自動検出

```python
"""Persistent Homology: β₁(θ) の相転移点でレイヤー境界を検出"""

import numpy as np


def detect_road_layers(G: nx.Graph, n_thresholds: int = 100) -> Dict[str, Any]:
    """θ掃引でβ₁(θ)を計算し、相転移点を検出。"""
    att_values = [float(d.get("attention", 0)) for _, _, d in G.edges(data=True)]
    if not att_values:
        return {"thresholds": [], "beta1": [], "transition_points": [], "layers": []}

    thresholds = np.linspace(0.0, max(att_values) + 0.01, n_thresholds)
    beta1_values = []
    all_nodes = set(G.nodes())

    for theta in thresholds:
        edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("attention",0) > theta]
        sub = nx.Graph()
        sub.add_nodes_from(all_nodes)
        sub.add_edges_from(edges)
        V = sub.number_of_nodes()
        E = sub.number_of_edges()
        C = nx.number_connected_components(sub)
        beta1_values.append(E - V + C)

    beta1_arr = np.array(beta1_values, dtype=float)
    gradient = np.abs(np.gradient(beta1_arr, thresholds))

    # 相転移点 = |dβ₁/dθ| の有意な極大
    transition_thetas = []
    mean_grad = np.mean(gradient)
    for i in range(2, len(gradient) - 2):
        if (gradient[i] > gradient[i-1] and gradient[i] > gradient[i+1]
                and gradient[i] > mean_grad * 2):
            transition_thetas.append(float(thresholds[i]))

    # レイヤー定義
    boundaries = [0.0] + sorted(transition_thetas) + [float(max(att_values) + 0.01)]
    layers = []
    labels = ["local", "arterial", "highway"]
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i+1]
        mask = (thresholds >= lo) & (thresholds < hi)
        avg_b1 = float(np.mean(beta1_arr[mask])) if np.any(mask) else 0.0
        label_idx = min(i, len(labels) - 1)
        layers.append({
            "layer_id": i,
            "theta_range": (lo, hi),
            "avg_beta1": avg_b1,
            "label": labels[label_idx] if i < len(labels) else f"layer_{i}",
        })

    return {
        "thresholds": thresholds.tolist(),
        "beta1": beta1_values,
        "transition_points": transition_thetas,
        "layers": layers,
    }
```

### 6.3 階層的ルーター

```python
"""β₁階層ルーティング本体"""

from collections import deque


class Beta1Router:
    """
    前計算: θ掃引でレイヤー検出
    検索:   キャッシュ → 高速相 → ローカル（路地〜国道）
    更新:   attention局所変更のみ
    """

    def __init__(self, road_graph: nx.Graph):
        self.G = road_graph
        self.layers: Optional[List[Dict]] = None
        self._cache: Dict[Tuple[int,int], List[int]] = {}

    def precompute(self, n_thresholds: int = 100) -> Dict[str, Any]:
        result = detect_road_layers(self.G, n_thresholds)
        self.layers = result["layers"]
        return result

    def route(self, origin: int, destination: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}

        # --- Layer 0: キャッシュ ---
        key = (origin, destination)
        if key in self._cache:
            stats["cache_hit"] = True
            return {"path": self._cache[key],
                    "time_s": self._path_time(self._cache[key]),
                    "stats": stats}

        if not self.layers or len(self.layers) <= 1:
            return self._fallback(origin, destination, stats)

        # --- 最上位レイヤー（高速相）でルーティング ---
        top = self.layers[-1]
        theta_top = top["theta_range"][0]
        hw = self._subgraph_above(theta_top)

        entry = self._nearest_in(origin, hw)
        exit_ = self._nearest_in(destination, hw)
        if entry is None or exit_ is None:
            return self._fallback(origin, destination, stats)

        try:
            hw_path = nx.shortest_path(hw, entry, exit_, weight="travel_time_s")
        except nx.NetworkXNoPath:
            return self._fallback(origin, destination, stats)

        # --- ローカル接続（路地〜国道相） ---
        first_mile = self._local_route(origin, entry)
        last_mile = self._local_route(exit_, destination)

        full = first_mile + hw_path[1:-1] + last_mile
        self._cache[key] = full

        stats["layers_used"] = [
            {"layer": "local", "nodes": len(first_mile)},
            {"layer": "highway", "nodes": len(hw_path)},
            {"layer": "local", "nodes": len(last_mile)},
        ]
        return {"path": full, "time_s": self._path_time(full), "stats": stats}

    # ---- リアルタイム更新 ----

    def update_traffic(self, u: int, v: int, congestion: float) -> None:
        """
        congestion: 1.0=通常, 2.0=2倍遅延, 0.0=通行止め
        CHとの決定的違い: 重みの局所変更のみ。構造再計算なし。
        """
        if not self.G.has_edge(u, v):
            return
        d = self.G[u][v]
        base = float(d.get("base_travel_time_s", d.get("travel_time_s", 0)))
        if congestion <= 0:
            d["attention"] = 0.0
            d["travel_time_s"] = float("inf")
        else:
            d["attention"] = min(1.0, 1.0 / congestion)
            d["travel_time_s"] = base * congestion
        self._invalidate(u, v)

    def boost_on_use(self, path: List[int], boost: float = 0.05) -> None:
        """実際に使われた経路のattentionを強化（学習効果）"""
        for i in range(len(path) - 1):
            if self.G.has_edge(path[i], path[i+1]):
                d = self.G[path[i]][path[i+1]]
                d["attention"] = min(1.0, d.get("attention", 0.0) + boost)

    def decay_all(self, rate: float = 0.999) -> None:
        """定期バッチ: 使われない道路のattention減衰"""
        for _, _, d in self.G.edges(data=True):
            d["attention"] = d.get("attention", 1.0) * rate

    # ---- 内部ヘルパー ----

    def _subgraph_above(self, theta: float) -> nx.Graph:
        edges = [(u,v,d) for u,v,d in self.G.edges(data=True)
                 if d.get("attention",0) > theta]
        sub = nx.Graph()
        sub.add_edges_from(edges)
        return sub

    def _nearest_in(self, node: int, sub: nx.Graph, max_hops=50) -> Optional[int]:
        if node in sub:
            return node
        visited = {node}
        q = deque([(node, 0)])
        while q:
            cur, dep = q.popleft()
            if dep >= max_hops:
                break
            for nb in self.G.neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    if nb in sub:
                        return nb
                    q.append((nb, dep + 1))
        return None

    def _local_route(self, s: int, t: int) -> List[int]:
        try:
            return nx.shortest_path(self.G, s, t, weight="travel_time_s")
        except nx.NetworkXNoPath:
            return [s, t]

    def _fallback(self, s, t, stats) -> Dict[str, Any]:
        stats["fallback"] = True
        try:
            path = nx.shortest_path(self.G, s, t, weight="travel_time_s")
        except nx.NetworkXNoPath:
            path = []
        return {"path": path, "time_s": self._path_time(path), "stats": stats}

    def _path_time(self, path: List[int]) -> float:
        t = 0.0
        for i in range(len(path) - 1):
            if self.G.has_edge(path[i], path[i+1]):
                t += self.G[path[i]][path[i+1]].get("travel_time_s", 0)
            else:
                return float("inf")
        return t

    def _invalidate(self, u, v) -> None:
        rm = [k for k, p in self._cache.items()
              if any((p[i]==u and p[i+1]==v) or (p[i]==v and p[i+1]==u)
                     for i in range(len(p)-1))]
        for k in rm:
            del self._cache[k]
```

### 6.4 PoC実行スクリプト

```python
"""小規模グリッドで動作確認"""


def create_test_network(size: int = 20) -> nx.Graph:
    """3層構造を模擬したグリッドグラフ"""
    G = nx.Graph()

    # 格子（国道相）
    for r in range(size):
        for c in range(size):
            n = r * size + c
            if c + 1 < size:
                G.add_edge(n, n+1, attention=0.5, road_type="secondary",
                           travel_time_s=30.0, base_travel_time_s=30.0)
            if r + 1 < size:
                G.add_edge(n, (r+1)*size+c, attention=0.5, road_type="secondary",
                           travel_time_s=30.0, base_travel_time_s=30.0)

    # 外周（高速相）
    for i in range(size - 1):
        for u, v in [(i, i+1), ((size-1)*size+i, (size-1)*size+i+1),
                     (i*size, (i+1)*size), (i*size+size-1, (i+1)*size+size-1)]:
            if G.has_edge(u, v):
                G[u][v].update(attention=0.9, road_type="motorway",
                               travel_time_s=10.0, base_travel_time_s=10.0)

    # 路地（対角線）
    for r in range(size - 1):
        for c in range(size - 1):
            if (r + c) % 3 == 0:
                G.add_edge(r*size+c, (r+1)*size+(c+1), attention=0.2,
                           road_type="residential",
                           travel_time_s=60.0, base_travel_time_s=60.0)
    return G


def run_poc():
    print("=== β₁ルーティング PoC ===\n")

    G = create_test_network(20)
    print(f"グラフ: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # レイヤー検出
    layers = detect_road_layers(G, 50)
    print(f"相転移点: {layers['transition_points']}")
    for l in layers["layers"]:
        print(f"  {l['label']}: θ={l['theta_range']}, avg β₁={l['avg_beta1']:.1f}")

    # ルーティング
    router = Beta1Router(G)
    router.precompute()
    result = router.route(0, 399)
    print(f"\n経路長: {len(result['path'])} nodes, 時間: {result['time_s']:.1f}s")

    # 渋滞
    print("\n--- 渋滞発生 ---")
    for i in range(5, 15):
        router.update_traffic(i, i+1, 5.0)
    result2 = router.route(0, 399)
    print(f"渋滞後: {len(result2['path'])} nodes, {result2['time_s']:.1f}s")

    # 通行止め
    print("\n--- 通行止め ---")
    router.update_traffic(0, 1, 0.0)
    result3 = router.route(0, 399)
    print(f"通行止め後: {len(result3['path'])} nodes, {result3['time_s']:.1f}s")


if __name__ == "__main__":
    run_poc()
```

---

## 7. 特に効く領域

| 領域 | 理由 |
|---|---|
| 車載デバイス（オフライン） | メモリ制限でCHが載らない |
| ドローン群 | 更新頻度が高い、前計算の余裕なし |
| 災害時ナビ | 道路変化が激しい、CHの前計算破綻 |
| 発展途上国 | インフラ貧弱、道路データが頻繁に変わる |
| ロボット倉庫 | 障害物動的、リアルタイム更新必須 |
| ゲームAI | 動的マップ、低メモリ、リアルタイム制約 |

---

## 8. 特許クレーム骨子

```
請求項1:
各エッジにattention重みを付与したグラフにおいて、
閾値θ掃引によりβ₁(θ)を計算し、
β₁変化率が極大となる相転移点を境界として階層化し、
各階層のβ₁に応じた計算資源配分で経路探索を行う方法。

請求項2:
リアルタイム交通情報をattention重みの局所更新として反映し、
更新エッジを含むサイクルの局所β₁のみ再計算する請求項1の方法。

請求項3:
経路使用実績によりattentionを強化し、
未使用経路を時間減衰させることで、
使用実績ベースの自律的階層最適化を行う請求項1の方法。

請求項4:
完全一致O(1)、グラフ走査O(degree)、全探索O(N log N)の
三層構成で不確実性に応じた計算資源配分を行う経路探索システム。

請求項5:
道路の新設・廃止・規制変更に対し、
変更エッジの局所β₁再計算のみでルーティング階層を更新する方法。
```

---

## 9. 実証ロードマップ

1. **迷路でβ₁基礎検証**（進行中）: AG閾値修正 → β₁ > 0 確認
2. **小規模道路グラフ**: OSMから都市1区画（〜10K nodes）→ レイヤー分離確認
3. **中規模検証**: 都市全体（〜100K nodes）→ CH比較（速度・メモリ・更新）
4. **特許出願**: 迷路+道路の実施例を添付

---

## 10. Googleが採用せざるを得ない理由

```
現行コスト:
  全米前計算サーバ群 + ショートカット辺ストレージ + 定期再計算

β₁切替後:
  元グラフ + attention float/エッジ + 局所更新のみ

→ サーバ費用が桁で下がる
→ 速度が「同等」ならコスト削減だけで採用理由になる
```
