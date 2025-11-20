#!/usr/bin/env python3
"""
Subgraph Context in Prompts Example
===================================

Shows how graph neighborhoods are included in LLM prompts.
"""

from insightspike.config import load_config
from insightspike.config import load_config


def demonstrate_subgraph_prompts():
    """Show how subgraph context appears in different prompt modes"""
    
    print("=== Subgraph Context in LLM Prompts ===\n")
    
    # Setup knowledge graph
    config = load_config(preset="graph_enhanced")
    config.llm.prompt_style = "detailed"  # Show detailed format
    config.llm.include_metadata = True
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent = MainAgent(config)
    agent.initialize()
    
    # Create a knowledge network
    print("1. Building Knowledge Network...")
    print("-" * 50)
    
    knowledge = [
        # Core machine learning concepts
        "Machine learning algorithms learn patterns from data.",
        "Supervised learning uses labeled training data.",
        "Neural networks are composed of interconnected layers.",
        "Deep learning uses multiple hidden layers.",
        
        # Related concepts
        "Backpropagation calculates gradients for training.",
        "Gradient descent optimizes model parameters.",
        "Loss functions measure prediction errors.",
        "Activation functions introduce non-linearity.",
        
        # Applications
        "Computer vision uses CNNs for image recognition.",
        "Natural language processing uses transformers.",
        "Reinforcement learning learns through rewards.",
        "Transfer learning reuses pretrained models.",
    ]
    
    for k in knowledge:
        agent.add_knowledge(k)
    
    print(f"Added {len(knowledge)} interconnected ML concepts\n")
    
    # Query that will trigger graph exploration
    query = "How do transformers relate to gradient descent?"
    print(f"2. Query: '{query}'")
    print("This should explore the graph: transformers → neural networks → training → gradient descent\n")
    
    # Process to see subgraph in action
    result = agent.process_question(query)
    
    print("3. Subgraph Context in Prompt:")
    print("-" * 50)
    
    # The subgraph context would appear in the prompt as:
    print("""
Example prompt with subgraph context (detailed mode):

Context:
Machine learning algorithms learn patterns from data.
Neural networks are composed of interconnected layers.
Natural language processing uses transformers.
Gradient descent optimizes model parameters.

[Knowledge Graph Context]
Central concepts: 3
Related concepts within 1 hops: 8
Connections: 12
Key relationships:
  - Natural language processing uses transformers... → Neural networks are composed of interconnected...
  - Neural networks are composed of interconnected... → Deep learning uses multiple hidden layers...
  - Deep learning uses multiple hidden layers... → Backpropagation calculates gradients for training...
  - Backpropagation calculates gradients for training... → Gradient descent optimizes model parameters...
  - Gradient descent optimizes model parameters... → Loss functions measure prediction errors...

Question: How do transformers relate to gradient descent?

Answer:
""")
    
    print("\n4. Benefits of Subgraph Context:")
    print("-" * 50)
    print("✓ Shows conceptual paths between query terms")
    print("✓ Provides local graph structure to LLM")
    print("✓ Enables reasoning about indirect relationships")
    print("✓ Gives topology hints (hub concepts, clusters)")


def compare_prompt_styles():
    """Compare how subgraph appears in different prompt styles"""
    
    print("\n\n=== Subgraph in Different Prompt Styles ===\n")
    
    # Sample subgraph context
    subgraph = {
        "center_nodes": [2, 5, 7],
        "nodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "edges": [(2, 3), (3, 5), (5, 6), (6, 7), (7, 8)],
        "radius": 1,
        "concept_map": [
            "Water evaporates from oceans... → Water vapor rises and cools...",
            "Water vapor rises and cools... → Clouds are made of tiny droplets...",
            "Clouds are made of tiny droplets... → Precipitation occurs as rain...",
        ]
    }
    
    print("1. DETAILED Mode (research/large models):")
    print("-" * 60)
    print("""
[Knowledge Graph Context]
Central concepts: 3
Related concepts within 1 hops: 10
Connections: 5
Key relationships:
  - Water evaporates from oceans... → Water vapor rises and cools...
  - Water vapor rises and cools... → Clouds are made of tiny droplets...
  - Clouds are made of tiny droplets... → Precipitation occurs as rain...
""")
    
    print("\n2. STANDARD Mode (production/medium models):")
    print("-" * 60)
    print("""
[Graph Context: 10 related concepts]
""")
    
    print("\n3. MINIMAL Mode (small models):")
    print("-" * 60)
    print("(Subgraph context omitted for space constraints)")
    
    print("\n\nKey Points:")
    print("- Detailed mode: Full graph structure with relationships")
    print("- Standard mode: Just indicates graph context exists")
    print("- Minimal mode: Skips graph context entirely")


def show_subgraph_impact():
    """Demonstrate impact of subgraph on reasoning"""
    
    print("\n\n=== Impact of Subgraph Context ===\n")
    
    print("Without Subgraph Context:")
    print("-" * 40)
    print("LLM sees: [doc1, doc2, doc3, ...]")
    print("Must infer relationships from text alone")
    print("")
    
    print("With Subgraph Context:")
    print("-" * 40)
    print("LLM sees: [doc1, doc2, doc3, ...] + graph structure")
    print("Can follow explicit paths:")
    print("  A → B → C → D")
    print("Understands which concepts are hubs")
    print("Sees clustering of related ideas")
    
    print("\n\nExample Reasoning Enhancement:")
    print("Query: 'How does photosynthesis relate to climate?'")
    print("\nSubgraph shows path:")
    print("photosynthesis → oxygen production → atmosphere → greenhouse gases → climate")
    print("\nThis helps LLM construct coherent multi-step explanations!")


if __name__ == "__main__":
    demonstrate_subgraph_prompts()
    compare_prompt_styles()
    show_subgraph_impact()
    
    print("\n\n✓ Subgraph context enriches prompts with structural knowledge!")
