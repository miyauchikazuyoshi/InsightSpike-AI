"""
Level 3: Isomorphism Discovery Algorithm

同型発見アルゴリズム - 2つのグラフを同型にする最小変換を発見する

T* = argmin_T GED(T(G₁), G₂)

この変換T*が「閃き」の計算的実体である。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import networkx as nx
import numpy as np
from itertools import permutations
import heapq


class EditType(Enum):
    """編集操作の種類"""
    NODE_RELABEL = "node_relabel"
    EDGE_RELABEL = "edge_relabel"
    NODE_INSERT = "node_insert"
    NODE_DELETE = "node_delete"
    EDGE_INSERT = "edge_insert"
    EDGE_DELETE = "edge_delete"


@dataclass
class EditOperation:
    """単一の編集操作"""
    edit_type: EditType
    source: Optional[str] = None  # 元のノード/エッジ
    target: Optional[str] = None  # 変換後のノード/エッジ
    params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.edit_type == EditType.NODE_RELABEL:
            return f"Relabel({self.source} → {self.target})"
        elif self.edit_type == EditType.EDGE_INSERT:
            return f"AddEdge({self.params.get('u')} → {self.params.get('v')})"
        elif self.edit_type == EditType.EDGE_DELETE:
            return f"DelEdge({self.params.get('u')} → {self.params.get('v')})"
        elif self.edit_type == EditType.NODE_INSERT:
            return f"AddNode({self.target})"
        elif self.edit_type == EditType.NODE_DELETE:
            return f"DelNode({self.source})"
        return f"{self.edit_type.value}({self.source}, {self.target})"


@dataclass
class Transform:
    """
    グラフ変換を表すクラス

    T* = argmin_T GED(T(G₁), G₂) の解
    """
    # ノードのマッピング: G1のノード → G2のノード
    node_mapping: Dict[str, str] = field(default_factory=dict)

    # 追加の編集操作（マッピングで表現できないもの）
    edit_operations: List[EditOperation] = field(default_factory=list)

    # 変換のコスト（GED）
    cost: float = 0.0

    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = ["Transform:"]
        lines.append(f"  Mapping: {self.node_mapping}")
        if self.edit_operations:
            lines.append(f"  Edits: {self.edit_operations}")
        lines.append(f"  Cost: {self.cost:.4f}")
        return "\n".join(lines)

    def apply(self, G: nx.Graph) -> nx.Graph:
        """変換をグラフに適用"""
        G_new = G.copy()

        # ノードのリラベル
        if self.node_mapping:
            G_new = nx.relabel_nodes(G_new, self.node_mapping)

        # 追加の編集操作を適用
        for op in self.edit_operations:
            if op.edit_type == EditType.NODE_INSERT:
                G_new.add_node(op.target, **op.params)
            elif op.edit_type == EditType.NODE_DELETE:
                if op.source in G_new:
                    G_new.remove_node(op.source)
            elif op.edit_type == EditType.EDGE_INSERT:
                u, v = op.params.get('u'), op.params.get('v')
                if u in G_new and v in G_new:
                    G_new.add_edge(u, v, **op.params.get('attrs', {}))
            elif op.edit_type == EditType.EDGE_DELETE:
                u, v = op.params.get('u'), op.params.get('v')
                if G_new.has_edge(u, v):
                    G_new.remove_edge(u, v)

        return G_new

    def to_insight_description(self) -> str:
        """変換を人間が読める「閃き」として記述"""
        descriptions = []

        # マッピングの説明
        if self.node_mapping:
            mappings = [f"{k}→{v}" for k, v in self.node_mapping.items()]
            descriptions.append(f"概念の対応: {', '.join(mappings)}")

        # 編集操作の説明
        for op in self.edit_operations:
            if op.edit_type == EditType.EDGE_INSERT:
                descriptions.append(f"新しい関係の発見: {op.params.get('u')} と {op.params.get('v')} の間")
            elif op.edit_type == EditType.NODE_INSERT:
                descriptions.append(f"新しい概念の導入: {op.target}")

        return "\n".join(descriptions) if descriptions else "同一構造（変換不要）"


@dataclass
class PartialMapping:
    """部分的なマッピング（探索中の状態）"""
    mapped: Dict[str, str]  # 既にマップされたノード
    remaining_source: List[str]  # G1でまだマップされていないノード
    remaining_target: Set[str]  # G2でまだマップされていないノード
    score: float = float('inf')

    def extend(self, source_node: str, target_node: str) -> 'PartialMapping':
        """マッピングを拡張"""
        new_mapped = self.mapped.copy()
        new_mapped[source_node] = target_node
        new_remaining_source = [n for n in self.remaining_source if n != source_node]
        new_remaining_target = self.remaining_target - {target_node}
        return PartialMapping(
            mapped=new_mapped,
            remaining_source=new_remaining_source,
            remaining_target=new_remaining_target
        )


class IsomorphismDiscovery:
    """
    同型発見アルゴリズム

    2つのグラフG1, G2に対して、G1をG2に変換する最小コストの変換T*を発見する。
    """

    def __init__(
        self,
        node_cost: float = 1.0,
        edge_cost: float = 1.0,
        relabel_cost: float = 0.5,
        use_semantic_similarity: bool = True,
        beam_width: int = 10,
        max_iterations: int = 1000
    ):
        self.node_cost = node_cost
        self.edge_cost = edge_cost
        self.relabel_cost = relabel_cost
        self.use_semantic_similarity = use_semantic_similarity
        self.beam_width = beam_width
        self.max_iterations = max_iterations

    def find_transform(self, G1: nx.Graph, G2: nx.Graph) -> Transform:
        """
        G1をG2に変換する最適な変換を発見

        Args:
            G1: ソースグラフ
            G2: ターゲットグラフ

        Returns:
            Transform: 最適な変換
        """
        # グラフタイプを統一（DiGraphの場合はGraphに変換して構造比較）
        if isinstance(G1, nx.DiGraph) and not isinstance(G2, nx.DiGraph):
            G1 = G1.to_undirected()
        elif isinstance(G2, nx.DiGraph) and not isinstance(G1, nx.DiGraph):
            G2 = G2.to_undirected()
        elif isinstance(G1, nx.DiGraph) and isinstance(G2, nx.DiGraph):
            # 両方DiGraphの場合はそのまま
            pass

        # 特殊ケース: 同一グラフ
        if nx.is_isomorphic(G1, G2):
            # 厳密な同型マッピングを見つける
            matcher = nx.isomorphism.GraphMatcher(G1, G2)
            if matcher.is_isomorphic():
                mapping = matcher.mapping
                return Transform(node_mapping=mapping, cost=0.0)

        # グラフサイズに応じてアルゴリズムを選択
        if len(G1.nodes) <= 8 and len(G2.nodes) <= 8:
            return self._exhaustive_search(G1, G2)
        else:
            return self._beam_search(G1, G2)

    def _exhaustive_search(self, G1: nx.Graph, G2: nx.Graph) -> Transform:
        """小さなグラフ用の完全探索"""
        nodes1 = list(G1.nodes())
        nodes2 = list(G2.nodes())

        best_transform = None
        best_cost = float('inf')

        # G1のノードをG2のノードにマップする全ての順列を試す
        # （ノード数が異なる場合は、余分なノードの追加/削除を考慮）
        n1, n2 = len(nodes1), len(nodes2)

        if n1 <= n2:
            # G1の全ノードをG2のサブセットにマップ
            for perm in permutations(nodes2, n1):
                mapping = dict(zip(nodes1, perm))
                cost, edits = self._compute_transform_cost(G1, G2, mapping)
                if cost < best_cost:
                    best_cost = cost
                    best_transform = Transform(
                        node_mapping=mapping,
                        edit_operations=edits,
                        cost=cost
                    )
        else:
            # G2の全ノードをG1のサブセットからマップ
            for perm in permutations(nodes1, n2):
                mapping = dict(zip(perm, nodes2))
                # 残りのノードは削除が必要
                cost, edits = self._compute_transform_cost(G1, G2, mapping)
                if cost < best_cost:
                    best_cost = cost
                    best_transform = Transform(
                        node_mapping=mapping,
                        edit_operations=edits,
                        cost=cost
                    )

        return best_transform or Transform(cost=float('inf'))

    def _beam_search(self, G1: nx.Graph, G2: nx.Graph) -> Transform:
        """大きなグラフ用のビームサーチ"""
        nodes1 = list(G1.nodes())
        nodes2 = set(G2.nodes())

        # 初期状態
        initial = PartialMapping(
            mapped={},
            remaining_source=nodes1,
            remaining_target=nodes2,
            score=0.0
        )

        beam = [initial]

        for iteration in range(min(len(nodes1), self.max_iterations)):
            if not beam or not beam[0].remaining_source:
                break

            next_beam = []

            for partial in beam:
                if not partial.remaining_source:
                    next_beam.append(partial)
                    continue

                # 次にマップするノード
                source_node = partial.remaining_source[0]

                # 候補となるターゲットノード（スコア順）
                candidates = self._get_mapping_candidates(
                    G1, G2, source_node, partial
                )

                for target_node, sim_score in candidates[:self.beam_width]:
                    new_partial = partial.extend(source_node, target_node)
                    # 部分的なコストを計算
                    new_partial.score = self._evaluate_partial_mapping(
                        G1, G2, new_partial.mapped
                    )
                    next_beam.append(new_partial)

                # マップしない選択肢（ノード削除）
                if len(partial.remaining_source) > len(partial.remaining_target):
                    no_map_partial = PartialMapping(
                        mapped=partial.mapped.copy(),
                        remaining_source=partial.remaining_source[1:],
                        remaining_target=partial.remaining_target,
                        score=partial.score + self.node_cost
                    )
                    next_beam.append(no_map_partial)

            # 上位beam_width個を保持
            beam = sorted(next_beam, key=lambda x: x.score)[:self.beam_width]

        # 最良の結果を変換に変換
        if beam:
            best = beam[0]
            cost, edits = self._compute_transform_cost(G1, G2, best.mapped)
            return Transform(
                node_mapping=best.mapped,
                edit_operations=edits,
                cost=cost
            )

        return Transform(cost=float('inf'))

    def _get_mapping_candidates(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        source_node: str,
        partial: PartialMapping
    ) -> List[Tuple[str, float]]:
        """マッピング候補をスコア順に返す"""
        candidates = []

        for target_node in partial.remaining_target:
            score = self._compute_node_similarity(G1, G2, source_node, target_node, partial.mapped)
            candidates.append((target_node, score))

        return sorted(candidates, key=lambda x: -x[1])

    def _compute_node_similarity(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        node1: str,
        node2: str,
        current_mapping: Dict[str, str]
    ) -> float:
        """ノード間の類似度を計算"""
        score = 0.0

        # 次数の類似度
        deg1 = G1.degree(node1)
        deg2 = G2.degree(node2)
        if max(deg1, deg2) > 0:
            degree_sim = 1 - abs(deg1 - deg2) / max(deg1, deg2)
            score += degree_sim * 0.3

        # 既存マッピングとの整合性（隣接ノードがマップされている場合）
        neighbors1 = set(G1.neighbors(node1))
        neighbors2 = set(G2.neighbors(node2))

        mapped_neighbors1 = neighbors1 & set(current_mapping.keys())
        if mapped_neighbors1:
            # node1の隣接ノードでマップ済みのものが、node2の隣接ノードにマップされているか
            consistent = sum(
                1 for n in mapped_neighbors1
                if current_mapping[n] in neighbors2
            )
            consistency_score = consistent / len(mapped_neighbors1)
            score += consistency_score * 0.5

        # ノード属性の類似度（存在する場合）
        attrs1 = G1.nodes[node1]
        attrs2 = G2.nodes[node2]
        if attrs1 and attrs2:
            common_keys = set(attrs1.keys()) & set(attrs2.keys())
            if common_keys:
                matches = sum(1 for k in common_keys if attrs1[k] == attrs2[k])
                attr_sim = matches / len(common_keys)
                score += attr_sim * 0.2

        return score

    def _evaluate_partial_mapping(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        mapping: Dict[str, str]
    ) -> float:
        """部分的なマッピングのコストを評価"""
        cost = 0.0

        # マップされたノード間のエッジの不一致をカウント
        mapped_nodes1 = list(mapping.keys())

        for i, u1 in enumerate(mapped_nodes1):
            for v1 in mapped_nodes1[i+1:]:
                u2, v2 = mapping[u1], mapping[v1]

                has_edge1 = G1.has_edge(u1, v1)
                has_edge2 = G2.has_edge(u2, v2)

                if has_edge1 != has_edge2:
                    cost += self.edge_cost

        return cost

    def _compute_transform_cost(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        mapping: Dict[str, str]
    ) -> Tuple[float, List[EditOperation]]:
        """完全なマッピングに基づいて変換コストと編集操作を計算"""
        cost = 0.0
        edits = []

        # ノードの追加/削除コスト
        mapped_targets = set(mapping.values())
        unmapped_sources = set(G1.nodes()) - set(mapping.keys())
        unmapped_targets = set(G2.nodes()) - mapped_targets

        # 削除が必要なノード
        for node in unmapped_sources:
            cost += self.node_cost
            edits.append(EditOperation(
                edit_type=EditType.NODE_DELETE,
                source=node
            ))

        # 追加が必要なノード
        for node in unmapped_targets:
            cost += self.node_cost
            edits.append(EditOperation(
                edit_type=EditType.NODE_INSERT,
                target=node
            ))

        # エッジの追加/削除
        # G1のエッジをチェック
        for u1, v1 in G1.edges():
            if u1 in mapping and v1 in mapping:
                u2, v2 = mapping[u1], mapping[v1]
                if not G2.has_edge(u2, v2):
                    cost += self.edge_cost
                    edits.append(EditOperation(
                        edit_type=EditType.EDGE_DELETE,
                        params={'u': u2, 'v': v2}
                    ))

        # G2のエッジをチェック
        reverse_mapping = {v: k for k, v in mapping.items()}
        for u2, v2 in G2.edges():
            if u2 in reverse_mapping and v2 in reverse_mapping:
                u1, v1 = reverse_mapping[u2], reverse_mapping[v2]
                if not G1.has_edge(u1, v1):
                    cost += self.edge_cost
                    edits.append(EditOperation(
                        edit_type=EditType.EDGE_INSERT,
                        params={'u': u2, 'v': v2}
                    ))

        return cost, edits


def discover_insight(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    **kwargs
) -> Transform:
    """
    2つのグラフ間の「閃き」（最小変換）を発見する

    これが T* = argmin_T GED(T(G₁), G₂) の実装

    Args:
        source_graph: ソースドメインのグラフ
        target_graph: ターゲットドメインのグラフ

    Returns:
        Transform: 発見された変換（= 閃きの計算的表現）
    """
    discoverer = IsomorphismDiscovery(**kwargs)
    return discoverer.find_transform(source_graph, target_graph)


# テスト用のユーティリティ
def create_test_graphs() -> Tuple[nx.Graph, nx.Graph, Dict[str, str]]:
    """テスト用のグラフペア（既知の変換付き）を作成"""

    # 太陽系グラフ
    solar = nx.Graph()
    solar.add_node("Sun", type="center")
    solar.add_node("Earth", type="orbiter")
    solar.add_node("Mars", type="orbiter")
    solar.add_edge("Sun", "Earth", relation="gravity")
    solar.add_edge("Sun", "Mars", relation="gravity")

    # 原子グラフ
    atom = nx.Graph()
    atom.add_node("Nucleus", type="center")
    atom.add_node("Electron1", type="orbiter")
    atom.add_node("Electron2", type="orbiter")
    atom.add_edge("Nucleus", "Electron1", relation="electromagnetic")
    atom.add_edge("Nucleus", "Electron2", relation="electromagnetic")

    # 既知の変換
    known_mapping = {
        "Sun": "Nucleus",
        "Earth": "Electron1",
        "Mars": "Electron2"
    }

    return solar, atom, known_mapping


if __name__ == "__main__":
    # テスト実行
    solar, atom, known_mapping = create_test_graphs()

    print("=== Isomorphism Discovery Test ===")
    print(f"Solar System: {solar.nodes()}, {solar.edges()}")
    print(f"Atom: {atom.nodes()}, {atom.edges()}")
    print(f"Known mapping: {known_mapping}")
    print()

    # 変換を発見
    transform = discover_insight(solar, atom)

    print("=== Discovered Transform ===")
    print(transform)
    print()
    print("=== Insight Description ===")
    print(transform.to_insight_description())
