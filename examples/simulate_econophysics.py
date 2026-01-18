
"""
Econophysics Eureka Simulation: Fluid Dynamics -> Financial Markets
===================================================================

This script demonstrates a more complex "Eureka Moment":
Mapping the structural properties of "Fluid Turbulence" (Physics) to 
"Market Panic/Crash" (Economics).

Hypothesis:
A financial crash structurally resembles fluid turbulence (vortices/loops of panic).
If the AI knows about Turbulence, it should recognize a Crash as a similar phenomenon.
"""

import logging
import os
import sys
import networkx as nx

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from insightspike.metrics.improved_gedig_metrics import calculate_gedig_metrics
from insightspike.algorithms.structure_enricher import StructureEnricher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EconophysicsSim")

def print_header(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

def create_turbulence_graph(n=10):
    """
    Create a graph representing Fluid Turbulence.
    Characterized by:
    - Many small feedback loops (vortices)
    - High local clustering (eddies)
    - Chaotic but interconnected
    """
    # Watts-Strogatz small-world graph simulates local clustering (eddies) + shortcuts
    G = nx.watts_strogatz_graph(n, k=4, p=0.3, seed=42)
    
    # Label generic physics entities
    mapping = {i: f"Particle_{i}" for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    
    nx.set_node_attributes(G, "Physics/FluidDynamics", "domain")
    # Identify specific loop structures as 'Vortices'
    cycles = sorted(nx.simple_cycles(G.to_directed()), key=len)[:3]
    for cycle in cycles:
        for node in cycle:
            G.nodes[node]['type'] = 'Vortex_Particle'
            
    return G, "Particle_0"

def create_market_crash_graph(n=10):
    """
    Create a graph representing a Market Crash (Panic).
    Characterized by:
    - Traders selling to each other in loops (Panic Spirals)
    - High interdependence (Contagion)
    - Structurally very similar to the Turbulence graph
    """
    # Use same parameters but different seed/labels to ensure structural similarity
    # but not exact identity. 
    # Note: Econophysics posits these *are* mathematically similar.
    G = nx.watts_strogatz_graph(n, k=4, p=0.3, seed=99) 
    
    # Label finance entities
    mapping = {i: f"Trader_{i}" for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    
    nx.set_node_attributes(G, "Economics/Finance", "domain")
    
    # In finance, loops are 'Panic Spirals' or 'Margin Calls'
    cycles = sorted(nx.simple_cycles(G.to_directed()), key=len)[:3]
    for cycle in cycles:
        for node in cycle:
            G.nodes[node]['type'] = 'Panic_Seller'
            
    return G, "Trader_0"

def main():
    enricher = StructureEnricher()
    
    # --- PHASE 1: PHYSICS DOMAIN (Knowledge Base) ---
    print_header("PHASE 1: Learning 'Fluid Turbulence' (Physics)")
    g_physics, p_center = create_turbulence_graph(15)
    
    enricher.enrich_graph(g_physics, center_node=p_center)
    
    # Show what the AI sees
    sig_p = g_physics.nodes[p_center].get("structure_signature")
    print(f"Domain: {g_physics.nodes[p_center]['domain']}")
    print(f"Structure Signature (Abstract): {sig_p[:4]}...") 
    print("AI Internal Representation: 'Complex feedback loops detected (Vortices).'")

    # --- PHASE 2: ECONOMICS DOMAIN (New Observation) ---
    print_header("PHASE 2: Observing 'Market Crash' (Economics)")
    g_economy, e_center = create_market_crash_graph(15)
    
    enricher.enrich_graph(g_economy, center_node=e_center)
    
    sig_e = g_economy.nodes[e_center].get("structure_signature")
    print(f"Domain: {g_economy.nodes[e_center]['domain']}")
    print(f"Structure Signature (Abstract): {sig_e[:4]}...")
    print("AI Internal Representation: 'Complex feedback loops detected (Panic Spirals).'")
    
    # --- PHASE 3: DISCOVERY ---
    print_header("PHASE 3: Cross-Domain Discovery (Econophysics)")
    
    # The AI compares the new "Market" structure to its known "Physics" library
    context = {
        "prototype_graph": g_physics,
        "prototype_center": p_center,
        "center_node": e_center,
        "description": "Comparing Financial Contagion to Fluid Turbulence"
    }
    
    metrics = calculate_gedig_metrics(
        g_physics, # Known Prototype
        g_economy, # Target
        config={
            "enable_similarity": True, 
            "similarity_threshold": 0.65, # Stricter threshold for complex graphs
            "similarity_weight": 1.0      # Maximum bonus for complex discoveries
        },
        analogy_context=context
    )
    
    print(f"Comparing '{g_physics.nodes[p_center]['domain']}' <==> '{g_economy.nodes[e_center]['domain']}'")
    print(f"Structural Similarity: {metrics.to_dict().get('analogy_bonus', 0.0)/1.5:.4f} (normalized)")
    
    if metrics.is_analogy:
        print("\n>>> EUREKA! MOMENT DETECTED <<<")
        print("The AI has discovered that Market Panics behave like Fluid Turbulence.")
        print(f"Insight Score Spike: {metrics.insight_score:.4f}")
        print("-" * 30)
        print("Implication: 'We can apply Navier-Stokes equations to option pricing?'")
        print("New Field Proposed: Econophysics")
    else:
        print("\nNo analogy detected. The structures were too different.")

if __name__ == "__main__":
    main()
