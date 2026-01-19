"""
Unit tests for Level 3: Isomorphism Discovery Algorithm

T* = argmin_T GED(T(G₁), G₂)
"""

import pytest
import networkx as nx
from src.insightspike.algorithms.isomorphism_discovery import (
    discover_insight,
    Transform,
    EditOperation,
    EditType,
    IsomorphismDiscovery,
    create_test_graphs,
)


class TestTransform:
    """Transform クラスのテスト"""

    def test_empty_transform(self):
        """空の変換"""
        t = Transform()
        assert t.cost == 0.0
        assert t.node_mapping == {}
        assert t.edit_operations == []

    def test_apply_node_mapping(self):
        """ノードマッピングの適用"""
        G = nx.Graph()
        G.add_nodes_from(["A", "B", "C"])
        G.add_edge("A", "B")

        t = Transform(node_mapping={"A": "X", "B": "Y", "C": "Z"})
        G_new = t.apply(G)

        assert set(G_new.nodes()) == {"X", "Y", "Z"}
        assert G_new.has_edge("X", "Y")

    def test_apply_edge_insert(self):
        """エッジ追加の適用"""
        G = nx.Graph()
        G.add_nodes_from(["A", "B"])

        t = Transform(
            edit_operations=[
                EditOperation(
                    edit_type=EditType.EDGE_INSERT,
                    params={"u": "A", "v": "B"}
                )
            ]
        )
        G_new = t.apply(G)

        assert G_new.has_edge("A", "B")

    def test_insight_description(self):
        """閃きの記述生成"""
        t = Transform(
            node_mapping={"太陽": "原子核", "惑星": "電子"},
            edit_operations=[
                EditOperation(
                    edit_type=EditType.EDGE_INSERT,
                    params={"u": "原子核", "v": "中性子"}
                )
            ]
        )
        desc = t.to_insight_description()

        assert "太陽→原子核" in desc
        assert "新しい関係の発見" in desc


class TestIsomorphismDiscovery:
    """同型発見アルゴリズムのテスト"""

    def test_identical_graphs(self):
        """同一グラフの場合"""
        G = nx.Graph()
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        transform = discover_insight(G, G.copy())

        assert transform.cost == 0.0

    def test_isomorphic_graphs(self):
        """同型グラフの場合（ラベルが異なる）"""
        G1 = nx.Graph()
        G1.add_edge("A", "B")
        G1.add_edge("B", "C")

        G2 = nx.Graph()
        G2.add_edge("X", "Y")
        G2.add_edge("Y", "Z")

        transform = discover_insight(G1, G2)

        assert transform.cost == 0.0
        assert len(transform.node_mapping) == 3

    def test_hub_spoke_pattern(self):
        """Hub-spoke パターン（太陽系 ≈ 原子）"""
        solar, atom, known_mapping = create_test_graphs()

        transform = discover_insight(solar, atom)

        assert transform.cost == 0.0
        # 中心ノードは中心にマップされるべき
        assert transform.node_mapping["Sun"] == "Nucleus"

    def test_ring_pattern(self):
        """環状パターン（ウロボロス ≈ ベンゼン）"""
        # 6ノードの環
        ring1 = nx.cycle_graph(6)
        ring1 = nx.relabel_nodes(ring1, {i: f"A{i}" for i in range(6)})

        ring2 = nx.cycle_graph(6)
        ring2 = nx.relabel_nodes(ring2, {i: f"B{i}" for i in range(6)})

        transform = discover_insight(ring1, ring2)

        assert transform.cost == 0.0

    def test_hierarchy_pattern(self):
        """階層パターン（会社 ≈ 軍隊）"""
        company = nx.DiGraph()
        company.add_edge("CEO", "VP")
        company.add_edge("VP", "Manager")

        military = nx.DiGraph()
        military.add_edge("General", "Colonel")
        military.add_edge("Colonel", "Captain")

        transform = discover_insight(company, military)

        assert transform.cost == 0.0

    def test_different_sizes(self):
        """サイズが異なるグラフ"""
        G1 = nx.path_graph(3)
        G1 = nx.relabel_nodes(G1, {i: f"A{i}" for i in range(3)})

        G2 = nx.path_graph(4)
        G2 = nx.relabel_nodes(G2, {i: f"B{i}" for i in range(4)})

        transform = discover_insight(G1, G2)

        # ノード追加が必要なのでコスト > 0
        assert transform.cost > 0
        # 追加操作があるはず
        assert any(
            op.edit_type == EditType.NODE_INSERT
            for op in transform.edit_operations
        )

    def test_missing_edge_discovery(self):
        """欠けているエッジの発見"""
        # 不完全なグラフ
        incomplete = nx.Graph()
        incomplete.add_edge("A", "B")
        incomplete.add_edge("B", "C")
        # A-Cのエッジがない

        # 完全なグラフ（三角形）
        complete = nx.Graph()
        complete.add_edge("X", "Y")
        complete.add_edge("Y", "Z")
        complete.add_edge("X", "Z")

        transform = discover_insight(incomplete, complete)

        # エッジ追加が発見されるはず
        edge_inserts = [
            op for op in transform.edit_operations
            if op.edit_type == EditType.EDGE_INSERT
        ]
        assert len(edge_inserts) == 1

    def test_extra_edge_discovery(self):
        """余分なエッジの発見"""
        # 余分なエッジがあるグラフ
        extra = nx.Graph()
        extra.add_edge("A", "B")
        extra.add_edge("B", "C")
        extra.add_edge("A", "C")  # 余分

        # シンプルなグラフ
        simple = nx.Graph()
        simple.add_edge("X", "Y")
        simple.add_edge("Y", "Z")

        transform = discover_insight(extra, simple)

        # エッジ削除が発見されるはず
        edge_deletes = [
            op for op in transform.edit_operations
            if op.edit_type == EditType.EDGE_DELETE
        ]
        assert len(edge_deletes) == 1


class TestScienceHistoryScenarios:
    """科学史シナリオの再現テスト"""

    def test_bohr_atomic_model(self):
        """ボーアの原子モデル（1913）: 太陽系 → 原子"""
        solar = nx.Graph()
        solar.add_node("Sun", role="center")
        solar.add_node("Earth", role="orbiter")
        solar.add_node("Mars", role="orbiter")
        solar.add_edge("Sun", "Earth")
        solar.add_edge("Sun", "Mars")

        atom = nx.Graph()
        atom.add_node("Nucleus", role="center")
        atom.add_node("Electron1", role="orbiter")
        atom.add_node("Electron2", role="orbiter")
        atom.add_edge("Nucleus", "Electron1")
        atom.add_edge("Nucleus", "Electron2")

        transform = discover_insight(solar, atom)

        assert transform.cost == 0.0
        assert transform.node_mapping["Sun"] == "Nucleus"

    def test_kekule_benzene(self):
        """ケクレのベンゼン環（1865）: ウロボロス → ベンゼン"""
        ouroboros = nx.cycle_graph(6)
        ouroboros = nx.relabel_nodes(
            ouroboros,
            {i: f"snake_{i}" for i in range(6)}
        )

        benzene = nx.cycle_graph(6)
        benzene = nx.relabel_nodes(
            benzene,
            {i: f"C{i}" for i in range(6)}
        )

        transform = discover_insight(ouroboros, benzene)

        assert transform.cost == 0.0

    def test_darwin_natural_selection(self):
        """ダーウィンの自然選択（1859）: マルサス経済学 → 進化論"""
        malthus = nx.DiGraph()
        malthus.add_node("Population")
        malthus.add_node("Resources")
        malthus.add_node("Competition")
        malthus.add_node("Survival")
        malthus.add_edge("Population", "Resources")
        malthus.add_edge("Resources", "Competition")
        malthus.add_edge("Competition", "Survival")

        evolution = nx.DiGraph()
        evolution.add_node("Species")
        evolution.add_node("Environment")
        evolution.add_node("Selection")
        evolution.add_node("Adaptation")
        evolution.add_edge("Species", "Environment")
        evolution.add_edge("Environment", "Selection")
        evolution.add_edge("Selection", "Adaptation")

        transform = discover_insight(malthus, evolution)

        assert transform.cost == 0.0


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_graphs(self):
        """空グラフ"""
        G1 = nx.Graph()
        G2 = nx.Graph()

        transform = discover_insight(G1, G2)

        assert transform.cost == 0.0

    def test_single_node(self):
        """単一ノード"""
        G1 = nx.Graph()
        G1.add_node("A")

        G2 = nx.Graph()
        G2.add_node("X")

        transform = discover_insight(G1, G2)

        assert transform.cost == 0.0
        assert transform.node_mapping == {"A": "X"}

    def test_disconnected_graph(self):
        """非連結グラフ"""
        G1 = nx.Graph()
        G1.add_edge("A", "B")
        G1.add_node("C")  # 孤立ノード

        G2 = nx.Graph()
        G2.add_edge("X", "Y")
        G2.add_node("Z")

        transform = discover_insight(G1, G2)

        assert transform.cost == 0.0


class TestPerformance:
    """パフォーマンステスト"""

    def test_medium_graph(self):
        """中規模グラフ（8ノード）"""
        G1 = nx.cycle_graph(8)
        G1 = nx.relabel_nodes(G1, {i: f"A{i}" for i in range(8)})

        G2 = nx.cycle_graph(8)
        G2 = nx.relabel_nodes(G2, {i: f"B{i}" for i in range(8)})

        transform = discover_insight(G1, G2)

        assert transform.cost == 0.0

    def test_larger_graph_with_beam_search(self):
        """大きめのグラフ（ビームサーチ使用）"""
        G1 = nx.barabasi_albert_graph(15, 2, seed=42)
        G1 = nx.relabel_nodes(G1, {i: f"A{i}" for i in range(15)})

        G2 = nx.barabasi_albert_graph(15, 2, seed=42)
        G2 = nx.relabel_nodes(G2, {i: f"B{i}" for i in range(15)})

        # 同じseedなので同型のはず
        transform = discover_insight(G1, G2, beam_width=5)

        # ビームサーチなので最適解は保証されないが、低コストで見つかるはず
        assert transform.cost <= 5.0  # 許容範囲


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
