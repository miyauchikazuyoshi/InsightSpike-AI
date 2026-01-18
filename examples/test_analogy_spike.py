
"""
Verification Script for Structural Analogy Spike
===============================================

This script demonstrates how finding a structural analogy triggers a "Eureka" spike
in the geDIG metrics.

Scenario:
1. We have a "Prototype" structure (e.g., Solar System: Sun + Planets).
2. We observe a new graph update that forms a similar structure (Atom: Nucleus + Electrons).
3. We expect:
    - High Structural Similarity
    - Positive Analogy Bonus
    - Spike in Insight Score (Eureka!)
"""

import logging
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from insightspike.metrics.improved_gedig_metrics import calculate_gedig_metrics

def create_star_graph(n_nodes: int, hub: str = "hub") -> nx.Graph:
    """Create a star graph (hub + spokes)."""
    G = nx.Graph()
    for i in range(1, n_nodes):
        G.add_edge(hub, f"spoke_{i}")
    return G

def create_chain_graph(n_nodes: int) -> nx.Graph:
    """Create a chain graph (1-2-3-...)."""
    G = nx.Graph()
    for i in range(n_nodes - 1):
        G.add_edge(f"node_{i}", f"node_{i+1}")
    return G

def main():
    print("=== Testing Structural Analogy Spike (Eureka Moment) ===\n")

    # 1. Define a Prototype (The "Mental Model")
    # A Star graph representing "Centralized Control" or "Solar System"
    prototype = create_star_graph(6, hub="Central")
    print(f"Prototype: Star Graph ({prototype.number_of_nodes()} nodes)")

    # 2. Define Previous State (Before)
    # Just unrelated scattered nodes
    graph_before = nx.Graph()
    graph_before.add_nodes_from(["n1", "n2", "n3", "n4", "n5", "n6"])
    
    # 3. Define New State (After) - The "Discovery"
    # A new structure that looks just like the prototype!
    graph_after = create_star_graph(6, hub="n1") # n1 becomes the hub
    # Add some domain Attribute to simulate "Different Domain"
    nx.set_node_attributes(graph_after, "Atom", "domain")
    nx.set_node_attributes(prototype, "Space", "domain")
    
    print(f"Graph After: New Star Graph in 'Atom' domain")

    # 4. Calculate geDIG WITHOUT Analogy Context (Baseline)
    print("\n--- Baseline: No Analogy Context ---")
    metrics_base = calculate_gedig_metrics(
        graph_before, 
        graph_after,
        config={"enable_similarity": False} # Disable explicit similarity for baseline
    )
    print(f"Insight Score: {metrics_base.insight_score:.3f}")
    print(f"Spike Detected: {metrics_base.spike_detected}")
    print(f"Analogy Bonus: {metrics_base.analogy_bonus:.3f}")

    # 5. Calculate geDIG WITH Analogy Context (Eureka!)
    print("\n--- Experiment: With Analogy Context (Eureka!) ---")
    
    # Context telling the system: "Compare this new thing to our known prototype"
    context = {
        "prototype_graph": prototype,
        "prototype_center": "Central",
        "center_node": "n1"
    }
    
    metrics_eureka = calculate_gedig_metrics(
        graph_before, 
        graph_after,
        config={
            "enable_similarity": True,
            "similarity_threshold": 0.6,
            "similarity_weight": 0.8  # High weight for dramatic effect
        },
        analogy_context=context
    )
    
    print(f"Insight Score: {metrics_eureka.insight_score:.3f}")
    print(f"Spike Detected: {metrics_eureka.spike_detected}")
    print(f"Analogy Bonus: {metrics_eureka.analogy_bonus:.3f}")
    print(f"Is Analogy: {metrics_eureka.is_analogy}")
    

    # 6. Verification
    if metrics_eureka.insight_score > metrics_base.insight_score:
        print("\n[SUCCESS] Analogy boosted the Insight Score!")
    else:
        print("\n[FAILURE] Analogy did not boost the score.")
        
    if metrics_eureka.is_analogy:
        print("[SUCCESS] Analogy was correctly detected.")
    else:
        print("[FAILURE] Analogy was NOT detected.")

    # 7. Test Decoupled Structure Enrichment
    print("\n--- Testing Decoupled Structure Enrichment ---")
    try:
        from insightspike.algorithms.structure_enricher import StructureEnricher
        
        enricher = StructureEnricher()
        # Enrich the 'graph_after' (which has the star structure)
        enricher.enrich_graph(graph_after, center_node="n1")
        
        node_data = graph_after.nodes["n1"]
        role = node_data.get("structure_role")
        signature = node_data.get("structure_signature")
        
        print(f"Node 'n1' Role: {role}")
        print(f"Node 'n1' Signature Length: {len(signature) if signature else 0}")
        
        if role == "hub":
            print("[SUCCESS] StructureEnricher correctly identified 'n1' as a hub.")
        else:
            print(f"[FAILURE] StructureEnricher identified 'n1' as {role}, expected 'hub'.")
             
    except ImportError:
        print("[WARNING] Could not import StructureEnricher (maybe not implemented yet).")
    except Exception as e:
        print(f"[FAILURE] Enrichment test failed with error: {e}")


if __name__ == "__main__":
    main()
