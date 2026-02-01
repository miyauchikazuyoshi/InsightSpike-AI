"""
Science History Simulation

Simulates historical scientific discoveries that were based on analogical reasoning.
Tests whether geDIG with structural similarity can "rediscover" these insights.

Historical discoveries simulated:
1. Bohr's atomic model (1913) - Analogy: Solar system → Atom
2. Kekulé's benzene ring (1865) - Analogy: Ouroboros (snake biting tail) → Benzene structure
3. Darwin's natural selection (1859) - Analogy: Malthus population theory → Species competition

Usage:
    python -m experiments.structural_similarity.science_history_simulation
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from insightspike.config.models import StructuralSimilarityConfig
from insightspike.algorithms.structural_similarity import StructuralSimilarityEvaluator
from insightspike.algorithms.gedig_core import GeDIGCore


# =============================================================================
# Historical Discovery Scenarios
# =============================================================================

@dataclass
class HistoricalDiscovery:
    """Represents a historical scientific discovery based on analogy."""
    name: str
    year: int
    scientist: str
    description: str
    source_domain: str
    source_knowledge: nx.Graph
    target_domain: str
    target_problem: nx.Graph
    expected_insight: str
    center_source: Optional[str] = None
    center_target: Optional[str] = None


def create_solar_system_knowledge() -> nx.Graph:
    """Knowledge about the solar system (pre-1913)."""
    G = nx.Graph()

    # Sun as central body
    G.add_node("sun", domain="astronomy", role="center",
               properties=["massive", "gravitational_source"])

    # Planets orbiting
    planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn"]
    for planet in planets:
        G.add_node(planet, domain="astronomy", role="orbiting_body",
                   properties=["smaller_mass", "stable_orbit"])
        G.add_edge("sun", planet, relation="gravitational_attraction")
        G.add_edge(planet, "sun", relation="orbits_around")

    # Key concept: stable orbits due to balance of forces
    G.add_node("orbital_mechanics", domain="astronomy", role="principle",
               properties=["centripetal_force", "gravitational_balance"])

    return G


def create_atom_problem_1913() -> nx.Graph:
    """The atomic structure problem before Bohr (1913)."""
    G = nx.Graph()

    # Known: positively charged nucleus (from Rutherford 1911)
    G.add_node("nucleus", domain="physics", role="center",
               properties=["positive_charge", "small", "massive"])

    # Known: negatively charged electrons
    electrons = ["electron_1", "electron_2", "electron_3"]
    for e in electrons:
        G.add_node(e, domain="physics", role="particle",
                   properties=["negative_charge", "light"])
        G.add_edge("nucleus", e, relation="electrostatic_attraction")

    # Problem: How do electrons remain stable? (classical physics predicts collapse)
    G.add_node("stability_problem", domain="physics", role="unknown",
               properties=["why_no_collapse", "discrete_spectra"])

    return G


def create_ouroboros_knowledge() -> nx.Graph:
    """Knowledge about the Ouroboros symbol."""
    G = nx.Graph()

    # Cyclic structure
    segments = ["head", "body_1", "body_2", "body_3", "body_4", "tail"]
    for i, seg in enumerate(segments):
        G.add_node(seg, domain="mythology", role="segment",
                   properties=["continuous", "cyclic"])
        next_seg = segments[(i + 1) % len(segments)]
        G.add_edge(seg, next_seg, relation="connected_to")

    # Key insight: tail connects back to head
    G.add_node("self_reference", domain="mythology", role="principle",
               properties=["circular", "self_contained", "eternal_return"])

    return G


def create_benzene_problem_1865() -> nx.Graph:
    """The benzene structure problem before Kekulé (1865)."""
    G = nx.Graph()

    # Known: C6H6 formula
    # Known: carbon usually has 4 bonds
    carbons = ["C1", "C2", "C3", "C4", "C5", "C6"]
    for c in carbons:
        G.add_node(c, domain="chemistry", role="atom",
                   properties=["carbon", "four_bonds"])

    # Linear arrangements tried but didn't work
    for i in range(len(carbons) - 1):
        G.add_edge(carbons[i], carbons[i + 1], relation="bond")

    # Problem: structure doesn't match properties
    G.add_node("structure_problem", domain="chemistry", role="unknown",
               properties=["unusual_stability", "symmetric_reactions"])

    return G


def create_benzene_solution() -> nx.Graph:
    """Kekulé's benzene ring solution."""
    G = nx.Graph()

    carbons = ["C1", "C2", "C3", "C4", "C5", "C6"]
    for c in carbons:
        G.add_node(c, domain="chemistry", role="atom",
                   properties=["carbon", "aromatic"])

    # Ring structure!
    for i in range(len(carbons)):
        next_c = carbons[(i + 1) % len(carbons)]
        G.add_edge(carbons[i], next_c, relation="bond")

    return G


def create_malthus_theory() -> nx.Graph:
    """Malthus's population theory (1798)."""
    G = nx.Graph()

    # Population growth
    G.add_node("population", domain="economics", role="entity",
               properties=["exponential_growth", "reproducing"])

    # Limited resources
    G.add_node("resources", domain="economics", role="constraint",
               properties=["limited", "linear_growth"])

    # Competition for survival
    G.add_node("competition", domain="economics", role="process",
               properties=["struggle", "survival"])
    G.add_node("death", domain="economics", role="outcome",
               properties=["population_control", "weaker_eliminated"])

    G.add_edge("population", "resources", relation="consumes")
    G.add_edge("resources", "competition", relation="scarcity_causes")
    G.add_edge("competition", "death", relation="leads_to")

    return G


def create_species_variation_problem() -> nx.Graph:
    """Darwin's problem: explaining species diversity (pre-1859)."""
    G = nx.Graph()

    # Observed: species variations
    species = ["finch_A", "finch_B", "finch_C"]
    for s in species:
        G.add_node(s, domain="biology", role="species",
                   properties=["variation", "adapted_traits"])

    # Observed: some survive, some don't
    G.add_node("survival_patterns", domain="biology", role="observation",
               properties=["differential_survival", "adaptation"])

    # Unknown: mechanism
    G.add_node("selection_mechanism", domain="biology", role="unknown",
               properties=["how_traits_selected", "inheritance"])

    for s in species:
        G.add_edge(s, "survival_patterns", relation="exhibits")

    return G


def select_focus_node(G: nx.Graph, preferred: Optional[str]) -> Optional[str]:
    """Pick a stable focus node for evaluation and focal subgraphs."""
    if preferred and preferred in G:
        return preferred
    if G.number_of_nodes() == 0:
        return None
    ranked = sorted(G.degree(), key=lambda item: (-item[1], str(item[0])))
    return ranked[0][0] if ranked else None


def attach_analogy_insight(
    G: nx.Graph,
    discovery: HistoricalDiscovery,
    focus_node: Optional[str],
) -> Optional[str]:
    """Attach an analogy insight node so focal structure actually changes."""
    node_id = "analogy_insight"
    if node_id in G:
        suffix = 1
        while f"{node_id}_{suffix}" in G:
            suffix += 1
        node_id = f"{node_id}_{suffix}"

    G.add_node(
        node_id,
        domain=discovery.target_domain,
        source_domain=discovery.source_domain,
        role="analogy",
        properties=["structural_mapping"],
    )

    if focus_node and focus_node in G:
        G.add_edge(focus_node, node_id, relation="analogy_hint")
        neighbors = [n for n in G.neighbors(focus_node) if n != node_id]
        if neighbors:
            G.add_edge(node_id, neighbors[0], relation="analogy_bridge")

    return node_id


# =============================================================================
# Simulation Runner
# =============================================================================

def create_discoveries() -> List[HistoricalDiscovery]:
    """Create all historical discovery scenarios."""
    return [
        HistoricalDiscovery(
            name="Bohr's Atomic Model",
            year=1913,
            scientist="Niels Bohr",
            description="Electrons orbit nucleus like planets orbit sun",
            source_domain="astronomy",
            source_knowledge=create_solar_system_knowledge(),
            target_domain="physics",
            target_problem=create_atom_problem_1913(),
            expected_insight="Electrons have stable orbits at discrete energy levels",
            center_source="sun",
            center_target="nucleus",
        ),
        HistoricalDiscovery(
            name="Kekulé's Benzene Ring",
            year=1865,
            scientist="August Kekulé",
            description="Benzene has cyclic ring structure like snake biting its tail",
            source_domain="mythology",
            source_knowledge=create_ouroboros_knowledge(),
            target_domain="chemistry",
            target_problem=create_benzene_problem_1865(),
            expected_insight="Carbon atoms form closed ring, explaining stability",
            center_source="head",
            center_target="C1",
        ),
        HistoricalDiscovery(
            name="Darwin's Natural Selection",
            year=1859,
            scientist="Charles Darwin",
            description="Species competition mirrors Malthusian population struggle",
            source_domain="economics",
            source_knowledge=create_malthus_theory(),
            target_domain="biology",
            target_problem=create_species_variation_problem(),
            expected_insight="Survival of the fittest through competitive selection",
            center_source="competition",
            center_target="survival_patterns",
        ),
    ]


@dataclass
class SimulationResult:
    """Result of simulating a discovery."""
    discovery: HistoricalDiscovery
    structural_similarity: float
    is_analogy_detected: bool
    gedig_without_ss: float
    gedig_with_internal_ss: float
    gedig_with_external_bonus: float
    gedig_with_total_bonus: float
    spike_detected: bool
    insight_bonus: float


def simulate_discovery(
    discovery: HistoricalDiscovery,
    ss_config: StructuralSimilarityConfig,
    connect_analogy_insight: bool = True,
    internal_cross_domain_only: bool = False,
    apply_external_bonus: bool = True,
) -> SimulationResult:
    """Simulate the discovery process.

    Process:
    1. Scientist has source knowledge (e.g., solar system)
    2. Scientist encounters target problem (e.g., atomic structure)
    3. System detects structural similarity
    4. If analogy found, insight bonus is added to geDIG
    """
    # Create evaluator
    evaluator = StructuralSimilarityEvaluator(ss_config)

    # Step 1: Evaluate structural similarity between source and target
    focus_node = select_focus_node(discovery.target_problem, discovery.center_target)
    sim_result = evaluator.evaluate(
        discovery.source_knowledge,
        discovery.target_problem,
        center1=discovery.center_source,
        center2=focus_node,
    )

    # Step 2: Calculate geDIG with and without structural similarity
    # Before: knowledge graph without the insight
    g_before = discovery.target_problem.copy()

    # After: knowledge graph with source pattern applied
    g_after = discovery.target_problem.copy()
    if connect_analogy_insight:
        attach_analogy_insight(g_after, discovery, focus_node)
    else:
        g_after.add_node(
            "analogy_insight",
            domain=discovery.target_domain,
            source_domain=discovery.source_domain,
            properties=["structural_mapping"],
        )

    # Use geDIG with SS enabled (internal SS within target domain)
    ss_config_for_gedig = ss_config.dict()
    ss_config_for_gedig["cross_domain_only"] = internal_cross_domain_only
    core_with_ss = GeDIGCore(
        enable_multihop=True,
        max_hops=2,
        lambda_weight=1.0,
        structural_similarity_config=ss_config_for_gedig,
    )

    core_without_ss = GeDIGCore(
        enable_multihop=True,
        max_hops=2,
        lambda_weight=1.0,
        structural_similarity_config={"enabled": False},
    )

    # Calculate geDIG values
    focal = {focus_node} if focus_node else set()

    result_with_ss = core_with_ss.calculate(
        g_prev=g_before,
        g_now=g_after,
        focal_nodes=focal or None,
    )

    result_without_ss = core_without_ss.calculate(
        g_prev=g_before,
        g_now=g_after,
        focal_nodes=focal or None,
    )

    # Calculate insight bonus
    insight_bonus = 0.0
    if apply_external_bonus and sim_result.is_analogy:
        insight_bonus = ss_config.analogy_weight * sim_result.similarity

    lambda_weight = core_without_ss.lambda_weight
    gedig_without_ss = result_without_ss.gedig_value
    gedig_with_internal_ss = result_with_ss.gedig_value
    gedig_with_external_bonus = gedig_without_ss - (lambda_weight * insight_bonus)
    gedig_with_total_bonus = gedig_with_internal_ss - (lambda_weight * insight_bonus)

    return SimulationResult(
        discovery=discovery,
        structural_similarity=sim_result.similarity,
        is_analogy_detected=sim_result.is_analogy,
        gedig_without_ss=gedig_without_ss,
        gedig_with_internal_ss=gedig_with_internal_ss,
        gedig_with_external_bonus=gedig_with_external_bonus,
        gedig_with_total_bonus=gedig_with_total_bonus,
        spike_detected=gedig_with_total_bonus < -0.1,  # Simplified spike detection
        insight_bonus=insight_bonus,
    )


def run_simulation(
    threshold: float = 0.6,
    analogy_weight: float = 0.3,
    internal_cross_domain_only: bool = False,
    connect_analogy_insight: bool = True,
    apply_external_bonus: bool = True,
) -> List[SimulationResult]:
    """Run simulation for all historical discoveries."""
    config = StructuralSimilarityConfig(
        enabled=True,
        method="signature",
        analogy_threshold=threshold,
        analogy_weight=analogy_weight,
        cross_domain_only=True,
        max_signature_hops=3,
    )

    discoveries = create_discoveries()
    results = []

    print(f"\n{'='*80}")
    print("SCIENCE HISTORY SIMULATION")
    print("Can geDIG 'rediscover' historical insights through structural analogy?")
    print(f"Settings: internal_cross_domain_only={internal_cross_domain_only}, "
          f"connect_analogy_insight={connect_analogy_insight}, "
          f"external_bonus={'on' if apply_external_bonus else 'off'}")
    print(f"{'='*80}\n")

    for discovery in discoveries:
        print(f"\n{'─'*80}")
        print(f"📜 {discovery.name} ({discovery.year})")
        print(f"   Scientist: {discovery.scientist}")
        print(f"   {discovery.description}")
        print(f"{'─'*80}")

        result = simulate_discovery(
            discovery,
            config,
            connect_analogy_insight=connect_analogy_insight,
            internal_cross_domain_only=internal_cross_domain_only,
            apply_external_bonus=apply_external_bonus,
        )
        results.append(result)

        # Print results
        analogy_status = "✓ DETECTED" if result.is_analogy_detected else "✗ NOT DETECTED"
        print(f"\n   Source: {discovery.source_domain.upper()} knowledge")
        print(f"   Target: {discovery.target_domain.upper()} problem")
        print(f"\n   Structural Similarity: {result.structural_similarity:.3f}")
        print(f"   Analogy Detection: {analogy_status}")
        print(f"\n   geDIG (without SS):          {result.gedig_without_ss:.4f}")
        print(f"   geDIG (internal SS):         {result.gedig_with_internal_ss:.4f}")
        print(f"   geDIG (external SS bonus):   {result.gedig_with_external_bonus:.4f}")
        print(f"   geDIG (total SS):            {result.gedig_with_total_bonus:.4f}")
        print(f"   Insight Bonus (external):    {result.insight_bonus:.4f}")

        if result.is_analogy_detected:
            print(f"\n   💡 INSIGHT: {discovery.expected_insight}")
        else:
            print(f"\n   ⚠️  Analogy not detected. Historical insight may require lower threshold.")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    detected = sum(1 for r in results if r.is_analogy_detected)
    print(f"Discoveries simulated: {len(results)}")
    print(f"Analogies detected: {detected}/{len(results)}")

    print(f"\n{'─'*40}")
    print(f"{'Discovery':<30} {'Similarity':>10} {'Detected':>10}")
    print(f"{'─'*40}")
    for r in results:
        status = "✓" if r.is_analogy_detected else "✗"
        print(f"{r.discovery.name:<30} {r.structural_similarity:>10.3f} {status:>10}")
    print(f"{'='*80}\n")

    return results


def analyze_what_if(results: List[SimulationResult]) -> None:
    """Analyze: What if the scientist had NOT made the analogical connection?"""
    print(f"\n{'='*80}")
    print("COUNTERFACTUAL ANALYSIS")
    print("What if the scientist had NOT seen the analogy?")
    print(f"{'='*80}\n")

    for r in results:
        if r.is_analogy_detected:
            delta = r.gedig_without_ss - r.gedig_with_total_bonus
            print(f"📜 {r.discovery.name}:")
            print(f"   Without analogy insight: geDIG = {r.gedig_without_ss:.4f}")
            print(f"   With analogy insight:    geDIG = {r.gedig_with_total_bonus:.4f}")
            print(f"   Insight contribution:    Δ = {delta:.4f}")
            print(f"   → The analogy made the solution more 'valuable' by the geDIG metric")
            print()


if __name__ == "__main__":
    # Run simulation
    results = run_simulation(threshold=0.5, analogy_weight=0.3)

    # Counterfactual analysis
    analyze_what_if(results)
