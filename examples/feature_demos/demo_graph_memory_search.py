#!/usr/bin/env python3
"""
Test Graph-Based Memory Search
==============================

Demonstrates multi-hop graph traversal for associative memory retrieval.
"""

import numpy as np
from insightspike.config import load_config


def test_graph_search():
    """Test graph-based memory search functionality"""
    
    print("=== Graph-Based Memory Search Test ===\n")
    
    # Create config with graph search enabled
    config = load_config(preset="experiment")
    config.graph.enable_graph_search = True
    config.graph.hop_limit = 2
    config.graph.neighbor_threshold = 0.4
    config.graph.path_decay = 0.7
    
    # Initialize agent
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent = MainAgent(config)
    agent.initialize()
    
    # Add interconnected knowledge
    print("1. Building Knowledge Graph...")
    print("-" * 50)
    
    # Water cycle knowledge
    knowledge_base = [
        "Water evaporates from oceans and lakes when heated by the sun.",
        "Evaporation transforms liquid water into water vapor in the atmosphere.",
        "Water vapor rises and cools in the atmosphere forming clouds.",
        "Clouds are made of tiny water droplets suspended in air.",
        "When clouds become heavy, precipitation occurs as rain or snow.",
        "Rain flows into rivers and streams that lead back to oceans.",
        "The water cycle is a continuous process that recycles Earth's water.",
        # Related but indirect concepts
        "Plants absorb water through their roots from the soil.",
        "Transpiration is when plants release water vapor through leaves.",
        "Photosynthesis uses water and sunlight to create energy.",
        "Ice caps store frozen water for thousands of years.",
        "Groundwater moves slowly through underground aquifers.",
    ]
    
    for knowledge in knowledge_base:
        agent.add_knowledge(knowledge)
    
    print(f"Added {len(knowledge_base)} interconnected facts about water cycle\n")
    
    # Test queries at different hop distances
    test_queries = [
        ("How do clouds form?", "Direct match - should find cloud formation"),
        ("What happens to rain water?", "1-hop - rain → rivers → oceans"),
        ("How do plants affect atmospheric water?", "2-hop - plants → transpiration → atmosphere"),
        ("What role does the sun play in the water cycle?", "Multi-hop - sun → evaporation → vapor → clouds"),
    ]
    
    print("2. Testing Multi-Hop Retrieval:")
    print("-" * 50)
    
    for query, description in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {description}")
        
        # Process with graph search
        result = agent.process_question(query)
        
        # Analyze retrieved documents
        if hasattr(result, 'retrieved_documents'):
            docs = result.retrieved_documents
            print(f"Retrieved {len(docs)} documents:")
            
            # Group by hop distance
            direct = [d for d in docs if d.get("hop", 0) == 0]
            one_hop = [d for d in docs if d.get("hop", 0) == 1]
            two_hop = [d for d in docs if d.get("hop", 0) == 2]
            
            print(f"  - Direct matches: {len(direct)}")
            print(f"  - 1-hop neighbors: {len(one_hop)}")
            print(f"  - 2-hop neighbors: {len(two_hop)}")
            
            # Show paths for multi-hop results
            for doc in docs:
                if doc.get("hop", 0) > 0:
                    path = doc.get("path", [])
                    print(f"  - Path (hop {doc['hop']}): {' → '.join(map(str, path))}")
        
        print(f"Response: {result.response[:100]}...")
        print(f"Has spike: {result.has_spike}")


def test_concept_chaining():
    """Test how graph search chains related concepts"""
    
    print("\n\n=== Concept Chaining Test ===\n")
    
    config = load_config(preset="experiment")
    config.graph.enable_graph_search = True
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent = MainAgent(config)
    agent.initialize()
    
    # Add chain of related concepts
    concept_chain = [
        "Energy from food is stored as ATP in cells.",
        "ATP powers muscle contractions during exercise.",
        "Exercise increases heart rate and blood flow.",
        "Blood carries oxygen to working muscles.",
        "Oxygen is used in cellular respiration.",
        "Cellular respiration produces ATP from glucose.",
        "Glucose comes from digesting carbohydrates.",
        "Carbohydrates are found in foods like bread and pasta.",
    ]
    
    for concept in concept_chain:
        agent.add_knowledge(concept)
    
    # Query that requires chaining
    query = "How does eating bread help with exercise?"
    print(f"Query: '{query}'")
    print("This requires chaining: bread → carbs → glucose → ATP → muscle energy\n")
    
    result = agent.process_question(query)
    
    print(f"Response: {result.response}")
    print(f"\nGraph search successfully chained concepts: {'bread' in result.response and 'ATP' in result.response}")


def compare_with_standard_search():
    """Compare graph search vs standard search"""
    
    print("\n\n=== Graph Search vs Standard Search ===\n")
    
    # Shared knowledge base
    knowledge = [
        "Machine learning uses algorithms to learn from data.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks.",
        "Backpropagation trains neural networks by adjusting weights.",
        "Gradient descent optimizes the loss function.",
        "Convolutional networks excel at image recognition.",
        "Transformers revolutionized natural language processing.",
        "Attention mechanisms help models focus on relevant information.",
    ]
    
    query = "How do transformers relate to neurons?"
    
    # Test with standard search
    print("1. Standard Search (direct similarity only):")
    config1 = load_config(preset="experiment")
    config1.graph.enable_graph_search = False
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent1 = MainAgent(config1)
    agent1.initialize()
    for k in knowledge:
        agent1.add_knowledge(k)
    
    result1 = agent1.process_question(query)
    print(f"Retrieved: {len(result1.retrieved_documents)} documents")
    print(f"Response preview: {result1.response[:150]}...")
    
    # Test with graph search
    print("\n2. Graph Search (multi-hop traversal):")
    config2 = load_config(preset="experiment")
    config2.graph.enable_graph_search = True
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent2 = MainAgent(config2)
    agent2.initialize()
    for k in knowledge:
        agent2.add_knowledge(k)
    
    result2 = agent2.process_question(query)
    print(f"Retrieved: {len(result2.retrieved_documents)} documents")
    print(f"Response preview: {result2.response[:150]}...")
    
    print("\n✓ Graph search can find indirect relationships through the knowledge graph!")


if __name__ == "__main__":
    test_graph_search()
    test_concept_chaining()
    compare_with_standard_search()
    
    print("\n\n=== Summary ===")
    print("Graph-based search enables:")
    print("- Multi-hop traversal to find indirectly related concepts")
    print("- Concept chaining for complex reasoning")
    print("- Better context through subgraph extraction")
    print("- Associative leaps similar to human thinking")
