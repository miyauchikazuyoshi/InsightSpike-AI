#!/usr/bin/env python3
"""Multi-domain knowledge base for testing multi-hop geDIG evaluation."""

from dataclasses import dataclass
from typing import List


@dataclass
class MultiDomainKnowledge:
    """Knowledge item with clear domain and cross-domain connections."""
    text: str
    domain: str
    subdomain: str
    concepts: List[str]
    connects_to: List[str]  # Other domains this knowledge connects to
    depth: str  # basic, intermediate, advanced


def create_multidomain_knowledge_base():
    """Create a diverse multi-domain knowledge base with intentional connections."""
    
    knowledge_items = [
        # === COMPUTER SCIENCE DOMAIN ===
        MultiDomainKnowledge(
            text="Algorithms are step-by-step procedures for solving problems, fundamental to both computer science and everyday cooking recipes.",
            domain="Computer Science",
            subdomain="Algorithms",
            concepts=["algorithm", "procedure", "problem_solving"],
            connects_to=["Cooking", "Mathematics"],
            depth="basic"
        ),
        MultiDomainKnowledge(
            text="Binary trees organize data hierarchically, similar to family trees in genealogy or organizational charts in business.",
            domain="Computer Science",
            subdomain="Data Structures",
            concepts=["binary_tree", "hierarchy", "data_structure"],
            connects_to=["Biology", "Business"],
            depth="intermediate"
        ),
        MultiDomainKnowledge(
            text="Machine learning models learn patterns from data, much like how the human brain forms neural pathways through experience.",
            domain="Computer Science",
            subdomain="Machine Learning",
            concepts=["machine_learning", "pattern_recognition", "neural_networks"],
            connects_to=["Neuroscience", "Psychology"],
            depth="advanced"
        ),
        
        # === BIOLOGY DOMAIN ===
        MultiDomainKnowledge(
            text="DNA uses a four-letter code (ATCG) to store genetic information, analogous to how computers use binary code.",
            domain="Biology",
            subdomain="Genetics",
            concepts=["DNA", "genetic_code", "information_storage"],
            connects_to=["Computer Science", "Information Theory"],
            depth="intermediate"
        ),
        MultiDomainKnowledge(
            text="Evolution through natural selection optimizes organisms for survival, similar to genetic algorithms in computer science.",
            domain="Biology",
            subdomain="Evolution",
            concepts=["evolution", "natural_selection", "optimization"],
            connects_to=["Computer Science", "Mathematics"],
            depth="advanced"
        ),
        MultiDomainKnowledge(
            text="Ecosystems maintain balance through feedback loops, a concept also crucial in control systems and economics.",
            domain="Biology",
            subdomain="Ecology",
            concepts=["ecosystem", "feedback_loops", "balance"],
            connects_to=["Economics", "Engineering"],
            depth="intermediate"
        ),
        
        # === PHYSICS DOMAIN ===
        MultiDomainKnowledge(
            text="Quantum entanglement shows that particles can be correlated regardless of distance, challenging classical notions of locality.",
            domain="Physics",
            subdomain="Quantum Mechanics",
            concepts=["quantum_entanglement", "non_locality", "correlation"],
            connects_to=["Philosophy", "Information Theory"],
            depth="advanced"
        ),
        MultiDomainKnowledge(
            text="Thermodynamics' entropy concept applies to information theory, measuring disorder in both physical and digital systems.",
            domain="Physics",
            subdomain="Thermodynamics",
            concepts=["entropy", "disorder", "information"],
            connects_to=["Computer Science", "Information Theory"],
            depth="intermediate"
        ),
        MultiDomainKnowledge(
            text="Wave-particle duality demonstrates that light exhibits both wave and particle properties depending on observation.",
            domain="Physics",
            subdomain="Quantum Physics",
            concepts=["wave_particle_duality", "observation", "complementarity"],
            connects_to=["Philosophy", "Neuroscience"],
            depth="advanced"
        ),
        
        # === ECONOMICS DOMAIN ===
        MultiDomainKnowledge(
            text="Supply and demand curves determine market equilibrium, similar to how neural networks find optimal weights.",
            domain="Economics",
            subdomain="Microeconomics",
            concepts=["supply_demand", "equilibrium", "optimization"],
            connects_to=["Computer Science", "Mathematics"],
            depth="basic"
        ),
        MultiDomainKnowledge(
            text="Game theory models strategic interactions, applicable to AI agents, evolutionary biology, and political science.",
            domain="Economics",
            subdomain="Game Theory",
            concepts=["game_theory", "strategy", "nash_equilibrium"],
            connects_to=["Computer Science", "Biology", "Politics"],
            depth="intermediate"
        ),
        MultiDomainKnowledge(
            text="Market bubbles exhibit emergent behavior from individual actions, like flocking in birds or traffic jams.",
            domain="Economics",
            subdomain="Behavioral Economics",
            concepts=["market_bubble", "emergent_behavior", "collective_action"],
            connects_to=["Biology", "Physics", "Psychology"],
            depth="advanced"
        ),
        
        # === PSYCHOLOGY DOMAIN ===
        MultiDomainKnowledge(
            text="Cognitive biases affect decision-making in humans, which must be considered when designing AI systems.",
            domain="Psychology",
            subdomain="Cognitive Psychology",
            concepts=["cognitive_bias", "decision_making", "heuristics"],
            connects_to=["Computer Science", "Economics"],
            depth="intermediate"
        ),
        MultiDomainKnowledge(
            text="Memory consolidation during sleep resembles batch processing in machine learning systems.",
            domain="Psychology",
            subdomain="Neuroscience",
            concepts=["memory_consolidation", "sleep", "learning"],
            connects_to=["Computer Science", "Biology"],
            depth="advanced"
        ),
        MultiDomainKnowledge(
            text="Social networks influence behavior through peer effects, similar to message passing in graph neural networks.",
            domain="Psychology",
            subdomain="Social Psychology",
            concepts=["social_networks", "peer_influence", "behavior"],
            connects_to=["Computer Science", "Sociology"],
            depth="intermediate"
        ),
        
        # === PHILOSOPHY DOMAIN ===
        MultiDomainKnowledge(
            text="The Chinese Room argument questions whether symbol manipulation equals understanding, relevant to AI consciousness debates.",
            domain="Philosophy",
            subdomain="Philosophy of Mind",
            concepts=["chinese_room", "consciousness", "understanding"],
            connects_to=["Computer Science", "Psychology"],
            depth="advanced"
        ),
        MultiDomainKnowledge(
            text="Occam's Razor favors simpler explanations, a principle used in model selection and scientific theory.",
            domain="Philosophy",
            subdomain="Philosophy of Science",
            concepts=["occams_razor", "simplicity", "parsimony"],
            connects_to=["Computer Science", "Physics", "Biology"],
            depth="basic"
        ),
        
        # === MATHEMATICS DOMAIN ===
        MultiDomainKnowledge(
            text="Fractals show self-similarity at different scales, appearing in nature, markets, and data compression algorithms.",
            domain="Mathematics",
            subdomain="Geometry",
            concepts=["fractals", "self_similarity", "scale_invariance"],
            connects_to=["Biology", "Economics", "Computer Science"],
            depth="intermediate"
        ),
        MultiDomainKnowledge(
            text="Graph theory provides tools for analyzing networks, from social connections to neural pathways to internet topology.",
            domain="Mathematics",
            subdomain="Discrete Mathematics",
            concepts=["graph_theory", "networks", "connectivity"],
            connects_to=["Computer Science", "Biology", "Sociology"],
            depth="intermediate"
        ),
        
        # === LINGUISTICS DOMAIN ===
        MultiDomainKnowledge(
            text="Language models like transformers process sequences similarly to how humans parse grammatical structures.",
            domain="Linguistics",
            subdomain="Computational Linguistics",
            concepts=["language_models", "syntax", "semantics"],
            connects_to=["Computer Science", "Psychology"],
            depth="advanced"
        ),
        MultiDomainKnowledge(
            text="Zipf's law describes word frequency distributions, appearing also in city sizes and internet traffic.",
            domain="Linguistics",
            subdomain="Quantitative Linguistics",
            concepts=["zipfs_law", "power_law", "frequency_distribution"],
            connects_to=["Mathematics", "Economics", "Physics"],
            depth="intermediate"
        ),
        
        # === MEDICINE DOMAIN ===
        MultiDomainKnowledge(
            text="Epidemiological models of disease spread use similar mathematics to information diffusion in social networks.",
            domain="Medicine",
            subdomain="Epidemiology",
            concepts=["epidemiology", "disease_spread", "SIR_model"],
            connects_to=["Mathematics", "Sociology", "Computer Science"],
            depth="intermediate"
        ),
        MultiDomainKnowledge(
            text="Personalized medicine uses genetic data and ML to tailor treatments, combining biology with data science.",
            domain="Medicine",
            subdomain="Precision Medicine",
            concepts=["personalized_medicine", "genomics", "data_driven"],
            connects_to=["Biology", "Computer Science", "Statistics"],
            depth="advanced"
        ),
        
        # === MUSIC DOMAIN ===
        MultiDomainKnowledge(
            text="Musical harmony follows mathematical ratios, with frequencies in simple integer relationships creating consonance.",
            domain="Music",
            subdomain="Music Theory",
            concepts=["harmony", "frequency_ratios", "consonance"],
            connects_to=["Mathematics", "Physics"],
            depth="basic"
        ),
        MultiDomainKnowledge(
            text="Algorithmic composition uses AI to generate music, blending creativity with computational rules.",
            domain="Music",
            subdomain="Computer Music",
            concepts=["algorithmic_composition", "generative_music", "creativity"],
            connects_to=["Computer Science", "Mathematics"],
            depth="advanced"
        )
    ]
    
    return knowledge_items


def create_multidomain_queries():
    """Create queries that benefit from multi-hop reasoning across domains."""
    
    queries = [
        # === QUERIES REQUIRING MULTI-HOP ===
        
        # Biology -> CS -> Math (3-hop)
        ("How do genetic algorithms relate to natural selection and mathematical optimization?", "advanced"),
        
        # Physics -> Philosophy -> CS (3-hop)
        ("What does quantum mechanics tell us about the nature of computation and consciousness?", "advanced"),
        
        # Economics -> Biology -> CS (3-hop)
        ("How do market dynamics resemble ecosystem balance and can this inform distributed systems?", "advanced"),
        
        # Psychology -> CS -> Biology (3-hop)
        ("How does human memory consolidation compare to machine learning and neural plasticity?", "advanced"),
        
        # Music -> Math -> Physics (3-hop)
        ("Why do mathematical ratios in music create harmonic resonance in physical systems?", "intermediate"),
        
        # === QUERIES WITH CLEAR DOMAIN BRIDGES ===
        
        # CS <-> Biology bridge
        ("How do DNA storage mechanisms inspire digital data compression techniques?", "intermediate"),
        
        # Physics <-> Economics bridge
        ("Can thermodynamic entropy explain market inefficiencies?", "advanced"),
        
        # Math <-> Biology <-> Medicine bridge
        ("How do fractal patterns in biological systems inform medical imaging analysis?", "advanced"),
        
        # === QUERIES CREATING SHORTCUTS ===
        
        # Should create direct CS-Medicine connection
        ("How can deep learning revolutionize early disease detection?", "intermediate"),
        
        # Should create direct Physics-Psychology connection
        ("Does quantum uncertainty principle apply to human decision-making?", "advanced"),
        
        # === QUERIES DETECTING LOOPS ===
        
        # CS -> Math -> Economics -> CS (loop)
        ("How do optimization algorithms in CS use economic game theory which relies on computational models?", "advanced"),
        
        # Biology -> CS -> Psychology -> Biology (loop)
        ("How do neural networks model brain function which evolved through biological processes?", "advanced"),
        
        # === SINGLE-DOMAIN QUERIES (baseline) ===
        
        ("What are the basic principles of supply and demand?", "basic"),
        ("How does a binary tree work?", "basic"),
        ("What is DNA composed of?", "basic"),
        
        # === NOVEL CROSS-DOMAIN QUERIES ===
        
        # Linguistics -> Medicine (new connection)
        ("Can language patterns predict onset of neurological disorders?", "advanced"),
        
        # Music -> Economics (new connection)
        ("How does streaming affect music market dynamics?", "intermediate"),
        
        # Philosophy -> Medicine (new connection)
        ("What ethical frameworks guide personalized medicine decisions?", "intermediate"),
        
        # === QUERIES TESTING REDUNDANCY ===
        
        # Already covered by existing knowledge
        ("What is natural selection?", "basic"),
        ("What are fractals?", "basic"),
        ("What is game theory?", "basic")
    ]
    
    return queries


def visualize_domain_connections(knowledge_items):
    """Create a visualization of domain connections."""
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = nx.Graph()
    
    # Add nodes for each domain
    domains = set()
    for item in knowledge_items:
        domains.add(item.domain)
        for connected_domain in item.connects_to:
            domains.add(connected_domain)
    
    # Add edges based on connections
    domain_connections = {}
    for item in knowledge_items:
        for connected_domain in item.connects_to:
            edge = tuple(sorted([item.domain, connected_domain]))
            domain_connections[edge] = domain_connections.get(edge, 0) + 1
    
    # Create graph
    for domain in domains:
        G.add_node(domain)
    
    for (d1, d2), weight in domain_connections.items():
        G.add_edge(d1, d2, weight=weight)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.7)
    
    # Draw edges with width based on connection count
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights], alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("Multi-Domain Knowledge Graph Connections", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../results/multidomain_graph.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Domain Statistics:")
    print(f"  Total domains: {len(domains)}")
    print(f"  Total connections: {len(domain_connections)}")
    print(f"  Average connections per domain: {len(domain_connections) * 2 / len(domains):.1f}")
    
    # Find most connected domains
    domain_degree = {}
    for domain in domains:
        domain_degree[domain] = G.degree(domain)
    
    sorted_domains = sorted(domain_degree.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🔗 Most connected domains:")
    for domain, degree in sorted_domains[:5]:
        print(f"  {domain}: {degree} connections")


if __name__ == "__main__":
    # Create knowledge base
    kb = create_multidomain_knowledge_base()
    queries = create_multidomain_queries()
    
    print(f"📚 Created multi-domain knowledge base:")
    print(f"  Total items: {len(kb)}")
    print(f"  Domains: {len(set(item.domain for item in kb))}")
    
    # Count cross-domain connections
    total_connections = sum(len(item.connects_to) for item in kb)
    print(f"  Cross-domain connections: {total_connections}")
    
    print(f"\n❓ Created test queries:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Multi-hop queries: {sum(1 for q, _ in queries if '->' in q or 'bridge' in q.lower())}")
    
    # Show domain distribution
    from collections import Counter
    domain_counts = Counter(item.domain for item in kb)
    print(f"\n📊 Domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} items")
    
    # Visualize connections
    visualize_domain_connections(kb)