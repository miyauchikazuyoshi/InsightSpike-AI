#!/usr/bin/env python3
"""
Test Enhanced Episode Management with Graph-Aware Integration and Splitting
=========================================================================

Demonstrates the new features:
1. Graph-informed episode integration
2. Automatic conflict-based splitting
3. Self-organizing knowledge structure
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.core.agents.main_agent_enhanced import EnhancedMainAgent


def demonstrate_enhanced_features():
    """Demonstrate the enhanced episode management features"""
    
    print("=== Enhanced Episode Management Demo ===\n")
    
    # Initialize enhanced agent
    print("1. Initializing Enhanced Agent...")
    agent = EnhancedMainAgent()
    agent.initialize()
    
    # Configure for demo
    print("\n2. Configuring episode management...")
    agent.configure_episode_management(
        integration_config={
            'similarity_threshold': 0.75,  # Lower for more integration
            'graph_weight': 0.3,
            'graph_connection_bonus': 0.1,
            'enable_graph_integration': True
        },
        splitting_config={
            'conflict_threshold': 0.6,  # Lower for more splitting
            'max_episode_length': 200,  # Shorter for demo
            'enable_auto_split': True
        }
    )
    
    # Test documents that should demonstrate features
    test_documents = [
        # Cluster 1: AI/ML
        "Machine learning algorithms are fundamental to artificial intelligence.",
        "Deep learning neural networks have revolutionized AI applications.",
        "Neural network architectures like transformers dominate NLP.",
        
        # Mixed document (should potentially split)
        "AI is transforming climate science through predictive models. Quantum computing offers new computational paradigms. Biology benefits from machine learning analysis.",
        
        # Cluster 2: Climate
        "Climate change requires urgent global action and policy changes.",
        "Environmental modeling helps predict future climate scenarios.",
        
        # Another mixed document
        "Machine learning helps analyze climate data. Quantum algorithms solve optimization problems. Biological systems inspire new AI architectures.",
        
        # Cluster 3: Quantum
        "Quantum computing leverages superposition for parallel computation.",
        "Quantum algorithms promise exponential speedups for certain problems.",
    ]
    
    print("\n3. Processing documents...")
    print("-" * 50)
    
    for i, text in enumerate(test_documents):
        print(f"\nDocument {i+1}: {text[:60]}...")
        
        # Add episode
        result = agent.add_episode_with_graph_update(text)
        
        if result['success']:
            print(f"  ✓ Success - Total episodes: {result['total_episodes']}")
            
            # Show integration info
            integration_info = result.get('integration_info', {})
            if integration_info.get('total_integrations', 0) > 0:
                print(f"  → Integrated! Total integrations: {integration_info['total_integrations']}")
                print(f"    Integration rate: {integration_info['integration_rate']:.1%}")
            
            # Show splitting info
            splitting_info = result.get('splitting_info', {})
            if splitting_info.get('episodes_split', 0) > 0:
                print(f"  ⚡ Split occurred! Episodes split: {splitting_info['episodes_split']}")
                print(f"    New episodes created: {splitting_info['new_episodes_created']}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Trigger global optimization
    print("\n\n4. Triggering global conflict resolution...")
    optimization_result = agent.trigger_global_optimization()
    if optimization_result['success']:
        print(f"  Episodes split: {optimization_result['episodes_split']}")
        print(f"  New episodes created: {optimization_result['new_episodes']}")
    
    # Show final statistics
    print("\n5. Final Statistics:")
    print("-" * 50)
    
    stats = agent.get_stats()
    memory_stats = stats.get('memory_stats', {})
    
    print(f"  Total episodes: {memory_stats.get('total_episodes', 0)}")
    print(f"  Total integrations: {memory_stats.get('total_integrations', 0)}")
    
    # Get enhanced stats
    l2_stats = agent.l2_memory.get_enhanced_stats()
    
    integration_stats = l2_stats.get('integration_stats', {})
    print(f"\n  Integration Statistics:")
    print(f"    Total attempts: {integration_stats.get('total_attempts', 0)}")
    print(f"    Successful integrations: {integration_stats.get('successful_integrations', 0)}")
    print(f"    Graph-assisted: {integration_stats.get('graph_assisted', 0)}")
    print(f"    Graph bonus applied: {integration_stats.get('graph_bonus_applied', 0)}")
    
    splitting_stats = l2_stats.get('splitting_stats', {})
    print(f"\n  Splitting Statistics:")
    print(f"    Conflicts detected: {splitting_stats.get('conflicts_detected', 0)}")
    print(f"    Episodes split: {splitting_stats.get('episodes_split', 0)}")
    print(f"    Total new episodes: {splitting_stats.get('total_new_episodes', 0)}")
    
    # Show some episode details
    print("\n6. Episode Graph Information:")
    print("-" * 50)
    
    # Check a few episodes
    for i in range(min(3, memory_stats.get('total_episodes', 0))):
        info = agent.get_episode_graph_info(i)
        print(f"\n  Episode {i}:")
        print(f"    Text: {info.get('text', 'N/A')}")
        print(f"    Connections: {info.get('connection_count', 0)}")
        print(f"    Conflict score: {info.get('conflict_score', 0.0):.3f}")
        if info.get('needs_splitting'):
            print(f"    ⚠️  Needs splitting!")


def test_specific_scenarios():
    """Test specific integration and splitting scenarios"""
    
    print("\n\n=== Specific Scenario Tests ===\n")
    
    agent = EnhancedMainAgent()
    agent.initialize()
    
    # Scenario 1: High similarity documents (should integrate)
    print("Scenario 1: Testing integration of similar documents")
    print("-" * 50)
    
    similar_docs = [
        "Deep learning models use neural networks for pattern recognition.",
        "Neural networks in deep learning recognize complex patterns.",
        "Pattern recognition through deep neural network models."
    ]
    
    for doc in similar_docs:
        result = agent.add_episode_with_graph_update(doc)
        print(f"Added: {doc[:40]}... Episodes: {result['total_episodes']}")
    
    # Scenario 2: Long mixed document (should split)
    print("\n\nScenario 2: Testing splitting of long mixed document")
    print("-" * 50)
    
    long_mixed = (
        "Artificial intelligence revolutionizes data analysis. "
        "Climate science benefits from predictive modeling. "
        "Quantum computing promises exponential speedups. "
        "Machine learning algorithms process big data. "
        "Environmental research uses AI for predictions. "
        "Quantum algorithms solve optimization problems."
    )
    
    result = agent.add_episode_with_graph_update(long_mixed)
    print(f"Added long document. Episodes: {result['total_episodes']}")
    
    splitting_info = result.get('splitting_info', {})
    if splitting_info.get('episodes_split', 0) > 0:
        print(f"✓ Document was split! New episodes: {splitting_info['new_episodes_created']}")
    
    # Final count
    final_stats = agent.l2_memory.get_enhanced_stats()
    print(f"\nFinal episode count: {len(agent.l2_memory.episodes)}")
    print(f"Integration rate: {final_stats.get('integration_rate', 0):.1%}")


if __name__ == "__main__":
    # Run main demo
    demonstrate_enhanced_features()
    
    # Run specific tests
    test_specific_scenarios()
    
    print("\n\n✅ Enhanced episode management demo complete!")
    print("\nKey features demonstrated:")
    print("- Graph-aware integration reduces redundancy")
    print("- Conflict-based splitting maintains coherence")
    print("- Self-organizing knowledge structure")
    print("- Dynamic attention-like behavior")