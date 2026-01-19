"""
Level 3 Isomorphism Discovery - ベンチマークと検証

1. 実世界データでの検証
2. スケーラビリティテスト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import networkx as nx
import numpy as np

from src.insightspike.algorithms.isomorphism_discovery import (
    discover_insight,
    Transform,
    IsomorphismDiscovery,
)


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    name: str
    num_nodes_source: int
    num_nodes_target: int
    num_edges_source: int
    num_edges_target: int
    transform_cost: float
    execution_time_ms: float
    mapping_found: bool
    insight_description: str
    metadata: Dict[str, Any] = None


def run_single_benchmark(
    name: str,
    G1: nx.Graph,
    G2: nx.Graph,
    **kwargs
) -> BenchmarkResult:
    """単一ベンチマークを実行"""
    start_time = time.perf_counter()
    transform = discover_insight(G1, G2, **kwargs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return BenchmarkResult(
        name=name,
        num_nodes_source=len(G1.nodes()),
        num_nodes_target=len(G2.nodes()),
        num_edges_source=len(G1.edges()),
        num_edges_target=len(G2.edges()),
        transform_cost=transform.cost,
        execution_time_ms=elapsed_ms,
        mapping_found=len(transform.node_mapping) > 0,
        insight_description=transform.to_insight_description(),
        metadata=kwargs
    )


# =============================================================================
# 1. 実世界データの構築
# =============================================================================

def create_physics_graphs() -> List[Tuple[str, nx.Graph, nx.Graph]]:
    """物理学ドメインのグラフペア"""
    scenarios = []

    # シナリオ1: 電気回路 vs 水流システム
    circuit = nx.DiGraph()
    circuit.add_node("Battery", type="source", property="voltage")
    circuit.add_node("Resistor", type="dissipator", property="resistance")
    circuit.add_node("Capacitor", type="storage", property="capacitance")
    circuit.add_node("Current", type="flow")
    circuit.add_edge("Battery", "Current", relation="drives")
    circuit.add_edge("Current", "Resistor", relation="flows_through")
    circuit.add_edge("Current", "Capacitor", relation="charges")
    circuit.add_edge("Resistor", "Battery", relation="returns_to")

    water = nx.DiGraph()
    water.add_node("Pump", type="source", property="pressure")
    water.add_node("Pipe", type="dissipator", property="friction")
    water.add_node("Tank", type="storage", property="volume")
    water.add_node("Flow", type="flow")
    water.add_edge("Pump", "Flow", relation="drives")
    water.add_edge("Flow", "Pipe", relation="flows_through")
    water.add_edge("Flow", "Tank", relation="fills")
    water.add_edge("Pipe", "Pump", relation="returns_to")

    scenarios.append(("circuit_vs_water", circuit, water))

    # シナリオ2: 振り子 vs LC回路
    pendulum = nx.Graph()
    pendulum.add_node("Mass", energy_type="kinetic")
    pendulum.add_node("Height", energy_type="potential")
    pendulum.add_node("Gravity", type="force")
    pendulum.add_edge("Mass", "Height", relation="oscillates")
    pendulum.add_edge("Gravity", "Mass", relation="acts_on")

    lc_circuit = nx.Graph()
    lc_circuit.add_node("Inductor", energy_type="magnetic")
    lc_circuit.add_node("Capacitor", energy_type="electric")
    lc_circuit.add_node("EMF", type="force")
    lc_circuit.add_edge("Inductor", "Capacitor", relation="oscillates")
    lc_circuit.add_edge("EMF", "Inductor", relation="acts_on")

    scenarios.append(("pendulum_vs_lc", pendulum, lc_circuit))

    # シナリオ3: 熱伝導 vs 電気伝導
    heat = nx.DiGraph()
    heat.add_node("HotBody", property="temperature")
    heat.add_node("ColdBody", property="temperature")
    heat.add_node("Conductor", property="thermal_conductivity")
    heat.add_node("HeatFlow", type="flow")
    heat.add_edge("HotBody", "HeatFlow", relation="source")
    heat.add_edge("HeatFlow", "Conductor", relation="through")
    heat.add_edge("Conductor", "ColdBody", relation="to")

    electric = nx.DiGraph()
    electric.add_node("HighPotential", property="voltage")
    electric.add_node("LowPotential", property="voltage")
    electric.add_node("Wire", property="electrical_conductivity")
    electric.add_node("Current", type="flow")
    electric.add_edge("HighPotential", "Current", relation="source")
    electric.add_edge("Current", "Wire", relation="through")
    electric.add_edge("Wire", "LowPotential", relation="to")

    scenarios.append(("heat_vs_electric", heat, electric))

    return scenarios


def create_biology_graphs() -> List[Tuple[str, nx.Graph, nx.Graph]]:
    """生物学ドメインのグラフペア"""
    scenarios = []

    # シナリオ1: 細胞 vs 工場
    cell = nx.DiGraph()
    cell.add_node("Nucleus", role="control_center")
    cell.add_node("Ribosome", role="manufacturing")
    cell.add_node("Mitochondria", role="power_plant")
    cell.add_node("Membrane", role="boundary")
    cell.add_node("DNA", role="blueprint")
    cell.add_edge("Nucleus", "DNA", relation="contains")
    cell.add_edge("DNA", "Ribosome", relation="instructs")
    cell.add_edge("Mitochondria", "Ribosome", relation="powers")
    cell.add_edge("Membrane", "Nucleus", relation="protects")

    factory = nx.DiGraph()
    factory.add_node("Office", role="control_center")
    factory.add_node("Assembly", role="manufacturing")
    factory.add_node("Generator", role="power_plant")
    factory.add_node("Wall", role="boundary")
    factory.add_node("Blueprint", role="blueprint")
    factory.add_edge("Office", "Blueprint", relation="contains")
    factory.add_edge("Blueprint", "Assembly", relation="instructs")
    factory.add_edge("Generator", "Assembly", relation="powers")
    factory.add_edge("Wall", "Office", relation="protects")

    scenarios.append(("cell_vs_factory", cell, factory))

    # シナリオ2: 神経ネットワーク vs コンピュータネットワーク
    neural = nx.DiGraph()
    neural.add_node("Neuron1", type="processor")
    neural.add_node("Neuron2", type="processor")
    neural.add_node("Neuron3", type="processor")
    neural.add_node("Synapse1", type="connection")
    neural.add_node("Synapse2", type="connection")
    neural.add_edge("Neuron1", "Synapse1", relation="outputs")
    neural.add_edge("Synapse1", "Neuron2", relation="inputs")
    neural.add_edge("Neuron2", "Synapse2", relation="outputs")
    neural.add_edge("Synapse2", "Neuron3", relation="inputs")

    computer = nx.DiGraph()
    computer.add_node("Server1", type="processor")
    computer.add_node("Server2", type="processor")
    computer.add_node("Server3", type="processor")
    computer.add_node("Cable1", type="connection")
    computer.add_node("Cable2", type="connection")
    computer.add_edge("Server1", "Cable1", relation="outputs")
    computer.add_edge("Cable1", "Server2", relation="inputs")
    computer.add_edge("Server2", "Cable2", relation="outputs")
    computer.add_edge("Cable2", "Server3", relation="inputs")

    scenarios.append(("neural_vs_computer", neural, computer))

    # シナリオ3: 生態系 vs 経済システム
    ecosystem = nx.DiGraph()
    ecosystem.add_node("Sun", role="energy_source")
    ecosystem.add_node("Plants", role="producer")
    ecosystem.add_node("Herbivores", role="consumer1")
    ecosystem.add_node("Carnivores", role="consumer2")
    ecosystem.add_node("Decomposers", role="recycler")
    ecosystem.add_edge("Sun", "Plants", relation="energy")
    ecosystem.add_edge("Plants", "Herbivores", relation="consumed_by")
    ecosystem.add_edge("Herbivores", "Carnivores", relation="consumed_by")
    ecosystem.add_edge("Carnivores", "Decomposers", relation="decomposed_by")
    ecosystem.add_edge("Decomposers", "Plants", relation="nutrients")

    economy = nx.DiGraph()
    economy.add_node("Resources", role="energy_source")
    economy.add_node("Manufacturers", role="producer")
    economy.add_node("Retailers", role="consumer1")
    economy.add_node("Consumers", role="consumer2")
    economy.add_node("Recyclers", role="recycler")
    economy.add_edge("Resources", "Manufacturers", relation="energy")
    economy.add_edge("Manufacturers", "Retailers", relation="consumed_by")
    economy.add_edge("Retailers", "Consumers", relation="consumed_by")
    economy.add_edge("Consumers", "Recyclers", relation="decomposed_by")
    economy.add_edge("Recyclers", "Manufacturers", relation="nutrients")

    scenarios.append(("ecosystem_vs_economy", ecosystem, economy))

    return scenarios


def create_social_graphs() -> List[Tuple[str, nx.Graph, nx.Graph]]:
    """社会科学ドメインのグラフペア"""
    scenarios = []

    # シナリオ1: 封建制度 vs 企業組織
    feudal = nx.DiGraph()
    feudal.add_node("King", level=1)
    feudal.add_node("Duke1", level=2)
    feudal.add_node("Duke2", level=2)
    feudal.add_node("Baron1", level=3)
    feudal.add_node("Baron2", level=3)
    feudal.add_node("Peasant1", level=4)
    feudal.add_node("Peasant2", level=4)
    feudal.add_edge("King", "Duke1", relation="rules")
    feudal.add_edge("King", "Duke2", relation="rules")
    feudal.add_edge("Duke1", "Baron1", relation="rules")
    feudal.add_edge("Duke2", "Baron2", relation="rules")
    feudal.add_edge("Baron1", "Peasant1", relation="rules")
    feudal.add_edge("Baron2", "Peasant2", relation="rules")

    corporate = nx.DiGraph()
    corporate.add_node("CEO", level=1)
    corporate.add_node("VP1", level=2)
    corporate.add_node("VP2", level=2)
    corporate.add_node("Manager1", level=3)
    corporate.add_node("Manager2", level=3)
    corporate.add_node("Employee1", level=4)
    corporate.add_node("Employee2", level=4)
    corporate.add_edge("CEO", "VP1", relation="manages")
    corporate.add_edge("CEO", "VP2", relation="manages")
    corporate.add_edge("VP1", "Manager1", relation="manages")
    corporate.add_edge("VP2", "Manager2", relation="manages")
    corporate.add_edge("Manager1", "Employee1", relation="manages")
    corporate.add_edge("Manager2", "Employee2", relation="manages")

    scenarios.append(("feudal_vs_corporate", feudal, corporate))

    return scenarios


# =============================================================================
# 2. スケーラビリティテスト
# =============================================================================

def create_scalability_graphs(n_nodes: int, edge_prob: float = 0.1, seed: int = 42) -> Tuple[nx.Graph, nx.Graph]:
    """スケーラビリティテスト用のグラフペアを生成"""
    np.random.seed(seed)

    # 同じ構造で異なるラベルのグラフを作成
    G1 = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    G1 = nx.relabel_nodes(G1, {i: f"A{i}" for i in range(n_nodes)})

    G2 = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    G2 = nx.relabel_nodes(G2, {i: f"B{i}" for i in range(n_nodes)})

    return G1, G2


def run_scalability_test(
    node_counts: List[int] = [10, 20, 50, 100, 200, 500, 1000],
    beam_width: int = 10
) -> List[BenchmarkResult]:
    """スケーラビリティテストを実行"""
    results = []

    for n in node_counts:
        print(f"Testing with {n} nodes...")

        G1, G2 = create_scalability_graphs(n, edge_prob=0.05)

        try:
            result = run_single_benchmark(
                name=f"scale_{n}",
                G1=G1,
                G2=G2,
                beam_width=beam_width,
                max_iterations=min(n, 100)
            )
            results.append(result)
            print(f"  Time: {result.execution_time_ms:.1f}ms, Cost: {result.transform_cost:.1f}")
        except Exception as e:
            print(f"  Error: {e}")

    return results


# =============================================================================
# 3. メイン実行
# =============================================================================

def run_realworld_benchmarks() -> List[BenchmarkResult]:
    """実世界データでのベンチマークを実行"""
    results = []

    print("=" * 60)
    print("実世界データでの検証")
    print("=" * 60)

    # 物理学ドメイン
    print("\n--- 物理学ドメイン ---")
    for name, G1, G2 in create_physics_graphs():
        result = run_single_benchmark(name, G1, G2)
        results.append(result)
        print(f"{name}:")
        print(f"  Cost: {result.transform_cost}, Time: {result.execution_time_ms:.1f}ms")
        print(f"  Insight: {result.insight_description[:100]}...")

    # 生物学ドメイン
    print("\n--- 生物学ドメイン ---")
    for name, G1, G2 in create_biology_graphs():
        result = run_single_benchmark(name, G1, G2)
        results.append(result)
        print(f"{name}:")
        print(f"  Cost: {result.transform_cost}, Time: {result.execution_time_ms:.1f}ms")
        print(f"  Insight: {result.insight_description[:100]}...")

    # 社会科学ドメイン
    print("\n--- 社会科学ドメイン ---")
    for name, G1, G2 in create_social_graphs():
        result = run_single_benchmark(name, G1, G2)
        results.append(result)
        print(f"{name}:")
        print(f"  Cost: {result.transform_cost}, Time: {result.execution_time_ms:.1f}ms")
        print(f"  Insight: {result.insight_description[:100]}...")

    return results


def main():
    """メイン実行"""
    all_results = []

    # 1. 実世界データでの検証
    realworld_results = run_realworld_benchmarks()
    all_results.extend(realworld_results)

    # 2. スケーラビリティテスト
    print("\n" + "=" * 60)
    print("スケーラビリティテスト")
    print("=" * 60)
    scale_results = run_scalability_test(
        node_counts=[10, 20, 50, 100, 200, 500],
        beam_width=10
    )
    all_results.extend(scale_results)

    # 結果のサマリ
    print("\n" + "=" * 60)
    print("結果サマリ")
    print("=" * 60)

    # 実世界データの成功率
    realworld_success = sum(1 for r in realworld_results if r.transform_cost == 0)
    print(f"\n実世界データ: {realworld_success}/{len(realworld_results)} 完全同型発見")

    # スケーラビリティ
    print("\nスケーラビリティ:")
    for r in scale_results:
        nodes = r.num_nodes_source
        time_ms = r.execution_time_ms
        print(f"  {nodes:4d} nodes: {time_ms:8.1f}ms")

    # 結果を保存
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    results_data = [asdict(r) for r in all_results]
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存: {output_dir / 'benchmark_results.json'}")

    return all_results


if __name__ == "__main__":
    main()
