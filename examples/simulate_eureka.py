
"""
Eureka Machine Simulation: Cross-Domain Structural Analogy
==========================================================

This script simulates an AI realizing a valid structural analogy between two completely
different domains: "Astronomy" (Solar System) and "Quantum Physics" (Atom Model).

Step 1: Learn "Solar System" structure (Hub-and-Spoke).
Step 2: Enrich it to understand 'Sun' is a HUB.
Step 3: Encounter "Rutherford Atom" structure (Nucleus + Electrons).
Step 4: Enrich it to understand 'Nucleus' is a HUB.
Step 5: Compare and Trigger "Eureka!" (High Insight Score).
"""

import networkx as nx
import logging
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from insightspike.metrics.improved_gedig_metrics import calculate_gedig_metrics
from insightspike.algorithms.structure_enricher import StructureEnricher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EurekaSim")

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def create_solar_system():
    G = nx.Graph()
    center = "Sun"
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
    for p in planets:
        G.add_edge(center, p)
    
    # Domain attributes
    nx.set_node_attributes(G, "Astronomy", "domain")
    nx.set_node_attributes(G, "Star", "type") # Sun
    for p in planets:
        G.nodes[p]["type"] = "Planet"
    return G, center

def create_atom_model():
    G = nx.Graph()
    center = "Nucleus"
    electrons = ["e1", "e2", "e3", "e4", "e5", "e6"]
    for e in electrons:
        G.add_edge(center, e)
        
    # Domain attributes (completely different from Astronomy!)
    nx.set_node_attributes(G, "QuantumPhysics", "domain")
    nx.set_node_attributes(G, "Proton", "type") # Nucleus
    for e in electrons:
        G.nodes[e]["type"] = "Electron"
    return G, center

def main():
    enricher = StructureEnricher()
    
    print_header("PHASE 1: Learning the 'Solar System'")
    g_solar, sun = create_solar_system()
    
    # 1. Structure Enrichment (AI acquires abstract concept)
    enricher.enrich_graph(g_solar, center_node=sun)
    role_sun = g_solar.nodes[sun].get("structure_role")
    print(f"Entities: {g_solar.nodes()}")
    print(f"Observation: 'Sun' connects to {g_solar.degree(sun)} planets.")
    print(f"AI Concept Abstraction -> Sun is a '{role_sun.upper()}'")

    print_header("PHASE 2: Encountering the 'Atom Model'")
    g_atom, nucleus = create_atom_model()
    
    # 2. Structure Enrichment (AI analyzes new domain)
    enricher.enrich_graph(g_atom, center_node=nucleus)
    role_nuc = g_atom.nodes[nucleus].get("structure_role")
    print(f"Entities: {g_atom.nodes()}")
    print(f"Observation: 'Nucleus' connects to {g_atom.degree(nucleus)} electrons.")
    print(f"AI Concept Abstraction -> Nucleus is a '{role_nuc.upper()}'")
    
    print_header("PHASE 3: The Eureka Moment (Mapping)")
    
    # The system compares the new observation (Atom) against its knowledge base (Solar System)
    # Context: We suspect Nucleus might be like Sun because they are both HUBs.
    analogy_context = {
        "prototype_graph": g_solar,
        "prototype_center": sun,
        "center_node": nucleus
    }
    
    # We treat g_solar as "Previous Knowledge" and g_atom as "New Observation"
    # Note: Usually geDIG compares t-1 to t, but here we use it for Cross-Domain matching.
    # We pass g_solar as 'graph_before' (the reference) and g_atom as 'graph_after' (the target).
    
    metrics = calculate_gedig_metrics(
        g_solar, 
        g_atom, 
        config={
            "enable_similarity": True,
            "similarity_threshold": 0.5,
            "similarity_weight": 1.0 # High weight for Eureka
        },
        analogy_context=analogy_context
    )
    
    print(f"Comparing Domain '{g_solar.nodes['Sun']['domain']}' <-> '{g_atom.nodes['Nucleus']['domain']}'")
    print(f"Structural Similarity Score: {metrics.to_dict().get('analogy_bonus', 0.0):.4f} (approx)")
    
    if metrics.is_analogy:
        print("\nResult: >>> EUREKA! <<<")
        print("The AI has detected a fundamental structural identity across domains.")
        print(f"Insight Score Spike: {metrics.insight_score:.4f}")
        print("Interpretation: 'The Atom is like a miniature Solar System.'")
    else:
        print("\nResult: No analogy detected.")

if __name__ == "__main__":
    main()
