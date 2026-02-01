"""
Analogy Detection Benchmark

Tests whether structural similarity correctly identifies known analogies
and rejects non-analogous pairs.

Usage:
    python -m experiments.structural_similarity.analogy_benchmark
"""

import sys
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from insightspike.config.models import StructuralSimilarityConfig
from insightspike.algorithms.structural_similarity import (
    StructuralSimilarityEvaluator,
)


# =============================================================================
# Graph Generators
# =============================================================================

def create_hub_spoke(name: str, num_spokes: int, domain: str) -> nx.Graph:
    """Create hub-and-spoke graph (solar system, atom, etc.)"""
    G = nx.Graph()
    hub = f"{name}_hub"
    G.add_node(hub, domain=domain, role="hub", name=name)

    for i in range(num_spokes):
        spoke = f"{name}_spoke_{i}"
        G.add_node(spoke, domain=domain, role="spoke")
        G.add_edge(hub, spoke, relation="orbits")

    return G


def create_hierarchy(name: str, depth: int, branching: int, domain: str) -> nx.Graph:
    """Create hierarchical tree (company, military, etc.)"""
    G = nx.balanced_tree(branching, depth)

    # Relabel nodes
    mapping = {i: f"{name}_node_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name

    # Mark root
    root = f"{name}_node_0"
    G.nodes[root]["role"] = "root"

    return G


def create_branching_network(name: str, levels: int, domain: str) -> nx.Graph:
    """Create branching network (blood vessels, river, etc.)"""
    G = nx.Graph()

    # Create binary tree-like branching
    node_id = 0
    current_level = [f"{name}_{node_id}"]
    G.add_node(current_level[0], domain=domain, role="source", name=name)
    node_id += 1

    for level in range(levels):
        next_level = []
        for parent in current_level:
            # Each node branches into 2
            for _ in range(2):
                child = f"{name}_{node_id}"
                G.add_node(child, domain=domain, role="branch")
                G.add_edge(parent, child, relation="flows_to")
                next_level.append(child)
                node_id += 1
        current_level = next_level

    return G


def create_chain(name: str, length: int, domain: str) -> nx.Graph:
    """Create linear chain graph"""
    G = nx.path_graph(length)
    mapping = {i: f"{name}_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name

    return G


def create_clique(name: str, size: int, domain: str) -> nx.Graph:
    """Create fully connected graph"""
    G = nx.complete_graph(size)
    mapping = {i: f"{name}_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name

    return G


def create_ring(name: str, size: int, domain: str) -> nx.Graph:
    """Create ring/cycle graph"""
    G = nx.cycle_graph(size)
    mapping = {i: f"{name}_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name

    return G


def create_grid(name: str, rows: int, cols: int, domain: str) -> nx.Graph:
    """Create grid graph"""
    G = nx.grid_2d_graph(rows, cols)
    mapping = {(i, j): f"{name}_{i}_{j}" for i, j in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name

    return G


def create_bipartite(name: str, left_size: int, right_size: int, domain: str) -> nx.Graph:
    """Create complete bipartite graph."""
    G = nx.complete_bipartite_graph(left_size, right_size)
    mapping = {}
    for i in range(left_size):
        mapping[i] = f"{name}_left_{i}"
    for j in range(right_size):
        mapping[left_size + j] = f"{name}_right_{j}"
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name
        if "_left_" in node:
            G.nodes[node]["role"] = "left"
        else:
            G.nodes[node]["role"] = "right"

    return G


def create_wheel(name: str, size: int, domain: str) -> nx.Graph:
    """Create wheel graph (hub + ring)."""
    G = nx.wheel_graph(size)
    mapping = {i: f"{name}_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name

    hub = f"{name}_0"
    if hub in G:
        G.nodes[hub]["role"] = "hub"

    return G


def create_lollipop(name: str, clique_size: int, path_length: int, domain: str) -> nx.Graph:
    """Create lollipop graph (clique + tail)."""
    G = nx.lollipop_graph(clique_size, path_length)
    mapping = {i: f"{name}_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name

    return G


def create_ladder(name: str, rungs: int, domain: str) -> nx.Graph:
    """Create ladder graph (2 x rungs grid)."""
    G = nx.grid_2d_graph(2, rungs)
    mapping = {(i, j): f"{name}_{i}_{j}" for i, j in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        G.nodes[node]["domain"] = domain
        G.nodes[node]["name"] = name

    return G


# =============================================================================
# Utilities
# =============================================================================

def select_center_nodes(G: nx.Graph, k: int = 1) -> List[str]:
    """Select candidate centers using role hints or degree fallback."""
    if G.number_of_nodes() == 0:
        return []

    role_priority = {"hub", "root", "source", "center"}
    role_nodes = [n for n, data in G.nodes(data=True) if data.get("role") in role_priority]
    if role_nodes:
        return sorted(role_nodes)[:k]

    # Fallback: highest-degree nodes (deterministic tie-breaker)
    ranked = sorted(G.degree(), key=lambda item: (-item[1], str(item[0])))
    return [n for n, _ in ranked[:k]]


def stable_seed(name: str, offset: int = 0) -> int:
    """Deterministic seed from name (stable across runs)."""
    return sum(ord(c) for c in name) + offset


def noise_suffix(add_fraction: float, remove_fraction: float) -> str:
    """Short label for noise levels."""
    add_tag = int(round(add_fraction * 100))
    remove_tag = int(round(remove_fraction * 100))
    if add_tag == remove_tag:
        return f"{add_tag:02d}"
    return f"{add_tag:02d}a{remove_tag:02d}r"


def apply_edge_noise(
    G: nx.Graph,
    add_fraction: float = 0.0,
    remove_fraction: float = 0.0,
    seed: int = 0,
) -> nx.Graph:
    """Apply deterministic edge add/remove noise to a copy of the graph."""
    rng = np.random.default_rng(seed)
    H = G.copy()

    if remove_fraction > 0 and H.number_of_edges() > 0:
        edges = list(H.edges())
        remove_count = min(len(edges), int(round(remove_fraction * len(edges))))
        if remove_count > 0:
            remove_idx = rng.choice(len(edges), size=remove_count, replace=False)
            for idx in remove_idx:
                u, v = edges[int(idx)]
                if H.has_edge(u, v):
                    H.remove_edge(u, v)

    if add_fraction > 0 and H.number_of_nodes() >= 2:
        existing = {tuple(sorted(e)) for e in H.edges()}
        nodes = list(H.nodes())
        candidates = []
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                pair = (u, v)
                if pair not in existing:
                    candidates.append(pair)

        add_base = max(1, len(existing))
        add_count = min(len(candidates), int(round(add_fraction * add_base)))
        if add_count > 0:
            add_idx = rng.choice(len(candidates), size=add_count, replace=False)
            for idx in add_idx:
                u, v = candidates[int(idx)]
                H.add_edge(u, v, relation="noise")

    return H


# =============================================================================
# Test Cases
# =============================================================================

@dataclass
class AnalogyTestCase:
    """A test case for analogy detection."""
    name: str
    graph1_generator: callable
    graph1_args: dict
    graph2_generator: callable
    graph2_args: dict
    expected_analogy: bool
    description: str
    center1: Optional[str] = None
    center2: Optional[str] = None
    graph1_noise: Optional[Dict[str, float]] = None
    graph2_noise: Optional[Dict[str, float]] = None
    same_domain: bool = False
    group_id: Optional[str] = None


# Known analogy pairs (should detect as analogous)
ANALOGY_PAIRS = [
    AnalogyTestCase(
        name="solar_system_atom",
        graph1_generator=create_hub_spoke,
        graph1_args={"name": "solar_system", "num_spokes": 8, "domain": "astronomy"},
        graph2_generator=create_hub_spoke,
        graph2_args={"name": "atom", "num_spokes": 6, "domain": "physics"},
        expected_analogy=True,
        description="Solar system ≈ Atom (hub-spoke pattern)",
    ),
    AnalogyTestCase(
        name="company_military",
        graph1_generator=create_hierarchy,
        graph1_args={"name": "company", "depth": 3, "branching": 3, "domain": "business"},
        graph2_generator=create_hierarchy,
        graph2_args={"name": "military", "depth": 3, "branching": 3, "domain": "military"},
        expected_analogy=True,
        description="Company org ≈ Military hierarchy (tree pattern)",
    ),
    AnalogyTestCase(
        name="blood_river",
        graph1_generator=create_branching_network,
        graph1_args={"name": "blood_vessels", "levels": 4, "domain": "biology"},
        graph2_generator=create_branching_network,
        graph2_args={"name": "river_delta", "levels": 4, "domain": "geography"},
        expected_analogy=True,
        description="Blood vessels ≈ River system (branching pattern)",
    ),
    AnalogyTestCase(
        name="benzene_ouroboros",
        graph1_generator=create_ring,
        graph1_args={"name": "benzene", "size": 6, "domain": "chemistry"},
        graph2_generator=create_ring,
        graph2_args={"name": "ouroboros", "size": 8, "domain": "mythology"},
        expected_analogy=True,
        description="Benzene ring ≈ Ouroboros (cyclic pattern)",
    ),
    AnalogyTestCase(
        name="city_grid_circuit",
        graph1_generator=create_grid,
        graph1_args={"name": "city_blocks", "rows": 4, "cols": 4, "domain": "urban"},
        graph2_generator=create_grid,
        graph2_args={"name": "circuit_board", "rows": 5, "cols": 5, "domain": "electronics"},
        expected_analogy=True,
        description="City grid ≈ Circuit board (grid pattern)",
    ),
    AnalogyTestCase(
        name="actors_movies_students_courses",
        graph1_generator=create_bipartite,
        graph1_args={"name": "actors_movies", "left_size": 4, "right_size": 6, "domain": "entertainment"},
        graph2_generator=create_bipartite,
        graph2_args={"name": "students_courses", "left_size": 5, "right_size": 7, "domain": "education"},
        expected_analogy=True,
        description="Actors-Movies ≈ Students-Courses (bipartite pattern)",
        center1="actors_movies_left_0",
        center2="students_courses_left_0",
    ),
    AnalogyTestCase(
        name="bicycle_wheel_router",
        graph1_generator=create_wheel,
        graph1_args={"name": "bicycle_wheel", "size": 10, "domain": "mechanics"},
        graph2_generator=create_wheel,
        graph2_args={"name": "router_wheel", "size": 12, "domain": "networking"},
        expected_analogy=True,
        description="Bicycle wheel ≈ Router wheel (hub-ring pattern)",
        center1="bicycle_wheel_0",
        center2="router_wheel_0",
    ),
    AnalogyTestCase(
        name="supply_chain_relay",
        graph1_generator=create_chain,
        graph1_args={"name": "supply_chain", "length": 8, "domain": "logistics"},
        graph2_generator=create_chain,
        graph2_args={"name": "relay_race", "length": 10, "domain": "sports"},
        expected_analogy=True,
        description="Supply chain ≈ Relay race (chain pattern)",
        center1="supply_chain_3",
        center2="relay_race_4",
    ),
]

# Non-analogy pairs (should NOT detect as analogous)
NON_ANALOGY_PAIRS = [
    AnalogyTestCase(
        name="solar_chain",
        graph1_generator=create_hub_spoke,
        graph1_args={"name": "solar_system", "num_spokes": 8, "domain": "astronomy"},
        graph2_generator=create_chain,
        graph2_args={"name": "food_chain", "length": 10, "domain": "biology"},
        expected_analogy=False,
        description="Solar system vs Chain (different structures)",
    ),
    AnalogyTestCase(
        name="tree_clique",
        graph1_generator=create_hierarchy,
        graph1_args={"name": "org_chart", "depth": 3, "branching": 2, "domain": "business"},
        graph2_generator=create_clique,
        graph2_args={"name": "friend_group", "size": 8, "domain": "social"},
        expected_analogy=False,
        description="Hierarchy vs Clique (tree vs complete)",
    ),
    AnalogyTestCase(
        name="ring_hub",
        graph1_generator=create_ring,
        graph1_args={"name": "cycle", "size": 6, "domain": "math"},
        graph2_generator=create_hub_spoke,
        graph2_args={"name": "star", "num_spokes": 6, "domain": "math"},
        expected_analogy=False,
        description="Ring vs Star (cycle vs hub-spoke)",
    ),
    AnalogyTestCase(
        name="grid_chain",
        graph1_generator=create_grid,
        graph1_args={"name": "checkerboard", "rows": 4, "cols": 4, "domain": "games"},
        graph2_generator=create_chain,
        graph2_args={"name": "timeline", "length": 16, "domain": "history"},
        expected_analogy=False,
        description="Grid vs Chain (2D vs 1D)",
    ),
    AnalogyTestCase(
        name="branching_clique",
        graph1_generator=create_branching_network,
        graph1_args={"name": "tree", "levels": 3, "domain": "nature"},
        graph2_generator=create_clique,
        graph2_args={"name": "team", "size": 10, "domain": "sports"},
        expected_analogy=False,
        description="Branching vs Clique (sparse vs dense)",
    ),
    AnalogyTestCase(
        name="bipartite_clique",
        graph1_generator=create_bipartite,
        graph1_args={"name": "actors_movies", "left_size": 4, "right_size": 6, "domain": "entertainment"},
        graph2_generator=create_clique,
        graph2_args={"name": "brainstorm", "size": 10, "domain": "cognitive"},
        expected_analogy=False,
        description="Bipartite vs Clique (two-part vs complete)",
        center1="actors_movies_left_0",
        center2="brainstorm_0",
    ),
    AnalogyTestCase(
        name="wheel_chain",
        graph1_generator=create_wheel,
        graph1_args={"name": "bicycle_wheel", "size": 10, "domain": "mechanics"},
        graph2_generator=create_chain,
        graph2_args={"name": "supply_chain", "length": 10, "domain": "logistics"},
        expected_analogy=False,
        description="Wheel vs Chain (hub-ring vs linear)",
        center1="bicycle_wheel_0",
        center2="supply_chain_4",
    ),
    AnalogyTestCase(
        name="lollipop_ring",
        graph1_generator=create_lollipop,
        graph1_args={"name": "lollipop_cluster", "clique_size": 5, "path_length": 4, "domain": "social"},
        graph2_generator=create_ring,
        graph2_args={"name": "ring_structure", "size": 9, "domain": "chemistry"},
        expected_analogy=False,
        description="Lollipop vs Ring (clique+tail vs cycle)",
        center1="lollipop_cluster_0",
        center2="ring_structure_0",
    ),
    AnalogyTestCase(
        name="wheel_ring",
        graph1_generator=create_wheel,
        graph1_args={"name": "bicycle_wheel", "size": 10, "domain": "mechanics"},
        graph2_generator=create_ring,
        graph2_args={"name": "ring_structure", "size": 10, "domain": "chemistry"},
        expected_analogy=False,
        description="Wheel vs Ring (hub-ring vs cycle)",
        center1="bicycle_wheel_0",
        center2="ring_structure_0",
    ),
    AnalogyTestCase(
        name="ladder_chain",
        graph1_generator=create_ladder,
        graph1_args={"name": "rail_ladder", "rungs": 6, "domain": "transport"},
        graph2_generator=create_chain,
        graph2_args={"name": "queue_chain", "length": 12, "domain": "social"},
        expected_analogy=False,
        description="Ladder vs Chain (2-track vs linear)",
        center1="rail_ladder_0_3",
        center2="queue_chain_6",
    ),
]

# Hard negatives: structurally similar but same-domain (should be rejected when cross-domain only)
HARD_NEGATIVE_PAIRS = [
    AnalogyTestCase(
        name="solar_system_exoplanet_same_domain",
        graph1_generator=create_hub_spoke,
        graph1_args={"name": "solar_system", "num_spokes": 8, "domain": "astronomy"},
        graph2_generator=create_hub_spoke,
        graph2_args={"name": "exoplanet_system", "num_spokes": 5, "domain": "astronomy"},
        expected_analogy=False,
        description="Same-domain hub-spoke (should be filtered when cross-domain only)",
        same_domain=True,
    ),
    AnalogyTestCase(
        name="company_startup_same_domain",
        graph1_generator=create_hierarchy,
        graph1_args={"name": "company", "depth": 3, "branching": 3, "domain": "business"},
        graph2_generator=create_hierarchy,
        graph2_args={"name": "startup", "depth": 3, "branching": 2, "domain": "business"},
        expected_analogy=False,
        description="Same-domain hierarchy (should be filtered when cross-domain only)",
        same_domain=True,
    ),
    AnalogyTestCase(
        name="benzene_cyclohexane_same_domain",
        graph1_generator=create_ring,
        graph1_args={"name": "benzene", "size": 6, "domain": "chemistry"},
        graph2_generator=create_ring,
        graph2_args={"name": "cyclohexane", "size": 6, "domain": "chemistry"},
        expected_analogy=False,
        description="Same-domain ring (should be filtered when cross-domain only)",
        same_domain=True,
    ),
    AnalogyTestCase(
        name="students_courses_same_domain",
        graph1_generator=create_bipartite,
        graph1_args={"name": "students_courses", "left_size": 5, "right_size": 7, "domain": "education"},
        graph2_generator=create_bipartite,
        graph2_args={"name": "teachers_classes", "left_size": 4, "right_size": 6, "domain": "education"},
        expected_analogy=False,
        description="Same-domain bipartite (should be filtered when cross-domain only)",
        same_domain=True,
        center1="students_courses_left_0",
        center2="teachers_classes_left_0",
    ),
    AnalogyTestCase(
        name="bicycle_wheel_same_domain",
        graph1_generator=create_wheel,
        graph1_args={"name": "bicycle_wheel", "size": 10, "domain": "mechanics"},
        graph2_generator=create_wheel,
        graph2_args={"name": "gear_wheel", "size": 8, "domain": "mechanics"},
        expected_analogy=False,
        description="Same-domain wheel (should be filtered when cross-domain only)",
        same_domain=True,
        center1="bicycle_wheel_0",
        center2="gear_wheel_0",
    ),
]


def build_noisy_variants(
    test_cases: List[AnalogyTestCase],
    add_fraction: float,
    remove_fraction: float,
    suffix: str = "noisy",
    seed_offset: int = 0,
) -> List[AnalogyTestCase]:
    """Create noisy variants of analogy pairs for robustness testing."""
    noisy_cases = []
    for tc in test_cases:
        group_id = tc.group_id or tc.name
        noisy_cases.append(
            AnalogyTestCase(
                name=f"{tc.name}_{suffix}",
                graph1_generator=tc.graph1_generator,
                graph1_args=tc.graph1_args,
                graph2_generator=tc.graph2_generator,
                graph2_args=tc.graph2_args,
                expected_analogy=tc.expected_analogy,
                description=f"{tc.description} ({suffix})",
                center1=tc.center1,
                center2=tc.center2,
                graph1_noise={
                    "add_fraction": add_fraction,
                    "remove_fraction": remove_fraction,
                    "seed_offset": 11 + seed_offset,
                },
                graph2_noise={
                    "add_fraction": add_fraction,
                    "remove_fraction": remove_fraction,
                    "seed_offset": 29 + seed_offset,
                },
                same_domain=tc.same_domain,
                group_id=group_id,
            )
        )
    return noisy_cases


def build_test_cases(
    include_hard_negatives: bool = True,
    include_noisy: bool = True,
    noise_add_fraction: float = 0.05,
    noise_remove_fraction: float = 0.05,
    noise_levels: Optional[List[float]] = None,
    include_non_analogy_noise: bool = True,
    include_hard_negative_noise: bool = True,
) -> List[AnalogyTestCase]:
    """Build the default benchmark dataset with optional noise variants."""
    base_cases = list(ANALOGY_PAIRS) + list(NON_ANALOGY_PAIRS)
    if include_hard_negatives:
        base_cases = base_cases + list(HARD_NEGATIVE_PAIRS)

    for tc in base_cases:
        if not tc.group_id:
            tc.group_id = tc.name

    all_cases: List[AnalogyTestCase] = list(base_cases)
    if include_noisy:
        if noise_levels:
            noise_pairs = [(float(level), float(level)) for level in noise_levels]
        else:
            noise_pairs = [(noise_add_fraction, noise_remove_fraction)]

        for idx, (add_fraction, remove_fraction) in enumerate(noise_pairs):
            level_tag = noise_suffix(add_fraction, remove_fraction)
            all_cases.extend(
                build_noisy_variants(
                    ANALOGY_PAIRS,
                    add_fraction=add_fraction,
                    remove_fraction=remove_fraction,
                    suffix=f"noisy_pos_{level_tag}",
                    seed_offset=1000 * idx,
                )
            )
            if include_non_analogy_noise:
                all_cases.extend(
                    build_noisy_variants(
                        NON_ANALOGY_PAIRS,
                        add_fraction=add_fraction,
                        remove_fraction=remove_fraction,
                        suffix=f"noisy_neg_{level_tag}",
                        seed_offset=1000 * idx + 100,
                    )
                )
            if include_hard_negatives and include_hard_negative_noise:
                all_cases.extend(
                    build_noisy_variants(
                        HARD_NEGATIVE_PAIRS,
                        add_fraction=add_fraction,
                        remove_fraction=remove_fraction,
                        suffix=f"noisy_same_domain_{level_tag}",
                        seed_offset=1000 * idx + 200,
                    )
                )

    return all_cases


def split_cases(
    test_cases: List[AnalogyTestCase],
    tune_ratio: float = 0.5,
    seed: int = 13,
) -> Tuple[List[AnalogyTestCase], List[AnalogyTestCase]]:
    """Split cases into tuning and evaluation sets, grouped by base case."""
    groups: Dict[str, List[AnalogyTestCase]] = {}
    for tc in test_cases:
        group_id = tc.group_id or tc.name
        groups.setdefault(group_id, []).append(tc)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    if not group_ids:
        return [], []

    tune_count = int(round(tune_ratio * len(group_ids)))
    tune_count = max(1, min(len(group_ids) - 1, tune_count))
    tune_group_ids = set(group_ids[:tune_count])

    tune_cases: List[AnalogyTestCase] = []
    eval_cases: List[AnalogyTestCase] = []
    for gid in group_ids:
        if gid in tune_group_ids:
            tune_cases.extend(groups[gid])
        else:
            eval_cases.extend(groups[gid])

    return tune_cases, eval_cases


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single test case."""
    test_case: AnalogyTestCase
    similarity: float
    detected_as_analogy: bool
    correct: bool
    is_cross_domain: bool
    method: str
    centers1: List[str] = field(default_factory=list)
    centers2: List[str] = field(default_factory=list)
    aggregation: str = "mean"
    similarity_scores: List[float] = field(default_factory=list)


def run_benchmark(
    threshold: float = 0.9,
    method: str = "motif",
    cross_domain_only: bool = True,
    center_count: int = 1,
    include_noisy: bool = True,
    noise_add_fraction: float = 0.05,
    noise_remove_fraction: float = 0.05,
    noise_levels: Optional[List[float]] = None,
    include_hard_negatives: bool = True,
    include_non_analogy_noise: bool = True,
    include_hard_negative_noise: bool = True,
    test_cases: Optional[List[AnalogyTestCase]] = None,
    label: Optional[str] = None,
) -> Tuple[List[BenchmarkResult], Dict[str, float]]:
    """Run the analogy detection benchmark.

    Args:
        threshold: Similarity threshold for analogy detection
        method: Similarity method to use
        cross_domain_only: Only consider cross-domain pairs as analogies
        center_count: Number of candidate centers per graph
        include_noisy: Add noisy variants of analogy pairs
        noise_add_fraction: Fraction of edges to add as noise
        noise_remove_fraction: Fraction of edges to remove as noise
        noise_levels: Noise fractions to mix (add/remove set to same value)
        include_hard_negatives: Include same-domain structural matches
        include_non_analogy_noise: Add noisy variants of non-analogy pairs
        include_hard_negative_noise: Add noisy variants of same-domain pairs
        test_cases: Optional pre-built case list (overrides include flags)
        label: Optional label for the benchmark heading

    Returns:
        Tuple of (individual results, aggregate metrics)
    """
    config = StructuralSimilarityConfig(
        enabled=True,
        method=method,
        analogy_threshold=threshold,
        cross_domain_only=cross_domain_only,
        max_signature_hops=3,
    )
    evaluator = StructuralSimilarityEvaluator(config)

    center_count = max(1, int(center_count))
    if test_cases is None:
        all_test_cases = build_test_cases(
            include_hard_negatives=include_hard_negatives,
            include_noisy=include_noisy,
            noise_add_fraction=noise_add_fraction,
            noise_remove_fraction=noise_remove_fraction,
            noise_levels=noise_levels,
            include_non_analogy_noise=include_non_analogy_noise,
            include_hard_negative_noise=include_hard_negative_noise,
        )
    else:
        all_test_cases = test_cases
        include_noisy = any(tc.graph1_noise or tc.graph2_noise for tc in all_test_cases)
        include_hard_negatives = any(tc.same_domain for tc in all_test_cases)
    results: List[BenchmarkResult] = []

    print(f"\n{'='*70}")
    if label:
        print(f"Analogy Detection Benchmark ({label})")
    else:
        print(f"Analogy Detection Benchmark")
    print(f"Method: {method}, Threshold: {threshold}, Cross-domain only: {cross_domain_only}")
    print(f"Center selection: role/degree (k={center_count})")
    if include_noisy:
        if noise_levels:
            levels = ", ".join(f"{level:.2f}" for level in noise_levels)
            print(f"Noise: on (levels=[{levels}])")
        else:
            print(f"Noise: on (add={noise_add_fraction:.2f}, remove={noise_remove_fraction:.2f})")
    else:
        print("Noise: off")
    if include_hard_negatives:
        print("Hard negatives: on (same-domain structural matches)")
    else:
        print("Hard negatives: off")
    print(f"{'='*70}\n")

    for tc in all_test_cases:
        # Generate graphs
        g1 = tc.graph1_generator(**tc.graph1_args)
        g2 = tc.graph2_generator(**tc.graph2_args)

        # Apply deterministic noise if configured
        seed_base = stable_seed(tc.name)
        if tc.graph1_noise:
            g1 = apply_edge_noise(
                g1,
                add_fraction=float(tc.graph1_noise.get("add_fraction", 0.0)),
                remove_fraction=float(tc.graph1_noise.get("remove_fraction", 0.0)),
                seed=seed_base + int(tc.graph1_noise.get("seed_offset", 0)),
            )
        if tc.graph2_noise:
            g2 = apply_edge_noise(
                g2,
                add_fraction=float(tc.graph2_noise.get("add_fraction", 0.0)),
                remove_fraction=float(tc.graph2_noise.get("remove_fraction", 0.0)),
                seed=seed_base + int(tc.graph2_noise.get("seed_offset", 0)),
            )

        # Get center nodes
        centers1 = [tc.center1] if tc.center1 else select_center_nodes(g1, k=center_count)
        centers2 = [tc.center2] if tc.center2 else select_center_nodes(g2, k=center_count)

        # Evaluate (aggregate over centers)
        similarity_scores: List[float] = []
        if centers1 and centers2:
            for c1 in centers1:
                for c2 in centers2:
                    sim_result = evaluator.evaluate(g1, g2, center1=c1, center2=c2)
                    similarity_scores.append(sim_result.similarity)
        else:
            sim_result = evaluator.evaluate(g1, g2)
            similarity_scores.append(sim_result.similarity)

        similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0
        is_cross_domain = evaluator._is_cross_domain(g1, g2)
        detected_as_analogy = similarity >= threshold and (is_cross_domain if cross_domain_only else True)
        expected_analogy = tc.expected_analogy
        if not cross_domain_only and tc.same_domain:
            expected_analogy = True

        # Record result
        result = BenchmarkResult(
            test_case=tc,
            similarity=similarity,
            detected_as_analogy=detected_as_analogy,
            correct=(detected_as_analogy == expected_analogy),
            is_cross_domain=is_cross_domain,
            method=method,
            centers1=centers1,
            centers2=centers2,
            aggregation="mean",
            similarity_scores=similarity_scores,
        )
        results.append(result)

        # Print result
        status = "✓" if result.correct else "✗"
        expected = "analogy" if expected_analogy else "not analogy"
        detected = "analogy" if result.detected_as_analogy else "not analogy"

        print(f"{status} {tc.name}")
        print(f"  {tc.description}")
        print(f"  Similarity: {result.similarity:.3f} | Expected: {expected} | Detected: {detected}")
        print(f"  Cross-domain: {result.is_cross_domain}")
        if center_count != 1 or tc.center1 or tc.center2:
            print(f"  Centers: g1={result.centers1} g2={result.centers2}")
        flags = []
        if tc.graph1_noise or tc.graph2_noise:
            flags.append("noise")
        if tc.same_domain:
            flags.append("same-domain")
        if flags:
            print(f"  Flags: {', '.join(flags)}")
        print()

    # Calculate metrics
    def is_expected_true(r: BenchmarkResult) -> bool:
        if not cross_domain_only and r.test_case.same_domain:
            return True
        return r.test_case.expected_analogy

    true_positives = sum(1 for r in results if is_expected_true(r) and r.detected_as_analogy)
    false_positives = sum(1 for r in results if not is_expected_true(r) and r.detected_as_analogy)
    true_negatives = sum(1 for r in results if not is_expected_true(r) and not r.detected_as_analogy)
    false_negatives = sum(1 for r in results if is_expected_true(r) and not r.detected_as_analogy)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(results)
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    metrics = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "accuracy": accuracy,
    }

    # Print summary
    print(f"{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total test cases: {len(results)}")
    print(f"Correct: {sum(1 for r in results if r.correct)} / {len(results)}")
    print()
    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives:  {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print()
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"FPR:       {fpr:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"{'='*70}\n")

    return results, metrics


def run_threshold_sweep(
    method: str = "motif",
    thresholds: List[float] = None,
    test_cases: Optional[List[AnalogyTestCase]] = None,
    selection_mode: str = "f1",
    recall_min: float = 0.95,
    **kwargs: Any,
) -> Dict[float, Dict[str, float]]:
    """Run benchmark across multiple thresholds to find optimal."""
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'#'*70}")
    print("Threshold Sweep")
    print(f"{'#'*70}\n")

    all_metrics = {}
    for threshold in thresholds:
        _, metrics = run_benchmark(
            threshold=threshold,
            method=method,
            cross_domain_only=True,
            test_cases=test_cases,
            **kwargs,
        )
        all_metrics[threshold] = metrics

    best_threshold = select_best_threshold(
        all_metrics,
        selection_mode=selection_mode,
        recall_min=recall_min,
    )

    print(f"\n{'='*70}")
    print("Threshold Comparison")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12} {'Accuracy':<12}")
    print("-" * 72)
    for t in thresholds:
        m = all_metrics[t]
        marker = " ← best" if t == best_threshold else ""
        print(f"{t:<12.2f} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {m['fpr']:<12.3f} {m['accuracy']:<12.3f}{marker}")
    print(f"{'='*70}\n")

    return all_metrics


def select_best_threshold(
    metrics_by_threshold: Dict[float, Dict[str, float]],
    selection_mode: str = "f1",
    recall_min: float = 0.95,
) -> float:
    """Select a threshold based on metrics."""
    if selection_mode == "fpr_min_recall":
        candidates = [
            (t, m) for t, m in metrics_by_threshold.items()
            if m.get("recall", 0.0) >= recall_min
        ]
        if not candidates:
            selection_mode = "f1"
        else:
            # Minimize FPR, then maximize precision/F1, then prefer higher threshold.
            candidates.sort(
                key=lambda item: (
                    item[1].get("fpr", 1.0),
                    -item[1].get("precision", 0.0),
                    -item[1].get("f1", 0.0),
                    -item[0],
                )
            )
            return candidates[0][0]

    if selection_mode == "precision":
        candidates = list(metrics_by_threshold.items())
        if recall_min is not None and recall_min > 0:
            filtered = [
                (t, m) for t, m in candidates
                if m.get("recall", 0.0) >= recall_min
            ]
            if filtered:
                candidates = filtered
        return max(
            candidates,
            key=lambda item: (
                item[1].get("precision", 0.0),
                -item[1].get("fpr", 1.0),
                item[1].get("f1", 0.0),
                item[0],
            ),
        )[0]

    # Default: maximize F1, then accuracy, then higher threshold.
    return max(
        metrics_by_threshold.keys(),
        key=lambda t: (
            metrics_by_threshold[t].get("f1", 0.0),
            metrics_by_threshold[t].get("accuracy", 0.0),
            t,
        ),
    )


def run_tuned_benchmark(
    method: str = "motif",
    thresholds: List[float] = None,
    tune_ratio: float = 0.5,
    seed: int = 13,
    seeds: Optional[List[int]] = None,
    selection_mode: str = "fpr_min_recall",
    recall_min: float = 0.95,
    **kwargs: Any,
) -> Tuple[float, List[BenchmarkResult], Dict[str, float]]:
    """Tune threshold on a validation split, then evaluate on holdout."""
    if thresholds is None:
        thresholds = [0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96]
    if seeds is None:
        seeds = [seed]
    noise_levels = kwargs.get("noise_levels")
    if noise_levels is None and kwargs.get("include_noisy", True):
        noise_levels = [0.05, 0.15, 0.30]
        kwargs["noise_levels"] = noise_levels

    all_cases = build_test_cases(
        include_hard_negatives=kwargs.get("include_hard_negatives", True),
        include_noisy=kwargs.get("include_noisy", True),
        noise_add_fraction=kwargs.get("noise_add_fraction", 0.05),
        noise_remove_fraction=kwargs.get("noise_remove_fraction", 0.05),
        noise_levels=noise_levels,
        include_non_analogy_noise=kwargs.get("include_non_analogy_noise", True),
        include_hard_negative_noise=kwargs.get("include_hard_negative_noise", True),
    )

    seed_thresholds: List[float] = []
    seed_metrics: List[Dict[str, float]] = []
    last_threshold = thresholds[0]
    last_results: List[BenchmarkResult] = []
    last_metrics: Dict[str, float] = {}

    for seed_value in seeds:
        tune_cases, eval_cases = split_cases(all_cases, tune_ratio=tune_ratio, seed=seed_value)

        print(f"\n{'#'*70}")
        print(f"Tuning Threshold (validation split, seed={seed_value})")
        print(f"{'#'*70}")
        print(f"Groups: total={len({tc.group_id or tc.name for tc in all_cases})} "
              f"tune={len({tc.group_id or tc.name for tc in tune_cases})} "
              f"eval={len({tc.group_id or tc.name for tc in eval_cases})}")
        print(f"Selection: mode={selection_mode}, recall_min={recall_min:.2f}")

        all_metrics = run_threshold_sweep(
            method=method,
            thresholds=thresholds,
            test_cases=tune_cases,
            selection_mode=selection_mode,
            recall_min=recall_min,
            **kwargs,
        )
        best_threshold = select_best_threshold(
            all_metrics,
            selection_mode=selection_mode,
            recall_min=recall_min,
        )

        print(f"\n{'='*70}")
        print(f"Holdout Evaluation (threshold={best_threshold:.2f}, seed={seed_value})")
        print(f"{'='*70}")
        results, metrics = run_benchmark(
            threshold=best_threshold,
            method=method,
            cross_domain_only=True,
            test_cases=eval_cases,
            label=f"holdout seed={seed_value}",
            **kwargs,
        )

        seed_thresholds.append(best_threshold)
        seed_metrics.append(metrics)
        last_threshold = best_threshold
        last_results = results
        last_metrics = metrics

    if len(seed_metrics) > 1:
        print(f"\n{'='*70}")
        print("Stability Summary (holdout)")
        print(f"{'='*70}")
        from collections import Counter
        counts = Counter(seed_thresholds)
        threshold_summary = ", ".join(f"{t:.2f}x{counts[t]}" for t in sorted(counts.keys()))
        print(f"Selected thresholds: {threshold_summary}")
        for key in ("precision", "recall", "f1", "fpr", "accuracy"):
            values = np.array([m.get(key, 0.0) for m in seed_metrics], dtype=np.float32)
            print(f"{key:<9} mean={values.mean():.3f} std={values.std():.3f}")
        print(f"{'='*70}\n")

    return last_threshold, last_results, last_metrics


if __name__ == "__main__":
    # Tune threshold on validation split, then evaluate on holdout
    best_threshold, results, metrics = run_tuned_benchmark(
        method="motif",
        selection_mode="precision",
        recall_min=0.85,
        seeds=[13, 23, 37],
        noise_levels=[0.05, 0.15, 0.30],
    )
