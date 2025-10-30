#!/usr/bin/env python3
"""
Test Adaptive Learning Mechanism
================================

Demonstrates how the system learns from successful patterns and adjusts strategies.
"""

import time
from insightspike.config import load_config


def test_adaptive_learning():
    """Test adaptive learning over multiple queries"""
    
    print("=== Adaptive Learning Test ===\n")
    
    # Create agent with learning enabled
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    config = load_config(preset="adaptive_learning")
    agent = MainAgent(config)
    agent.initialize()
    
    # Add diverse knowledge base
    print("1. Building Knowledge Base...")
    print("-" * 50)
    
    knowledge = [
        # Physics concepts
        "Force equals mass times acceleration (F=ma).",
        "Energy cannot be created or destroyed, only transformed.",
        "Light travels at 299,792,458 meters per second in vacuum.",
        "Gravity is the weakest of the four fundamental forces.",
        
        # Biology concepts
        "DNA contains the genetic instructions for life.",
        "Photosynthesis converts light energy into chemical energy.",
        "Evolution occurs through natural selection.",
        "Cells are the basic unit of life.",
        
        # Computer science
        "Algorithms are step-by-step procedures for solving problems.",
        "Big O notation describes algorithm time complexity.",
        "Machine learning models learn patterns from data.",
        "Neural networks are inspired by biological neurons.",
    ]
    
    for k in knowledge:
        agent.add_knowledge(k)
    
    print(f"Added {len(knowledge)} facts across multiple domains\n")
    
    # Test queries to observe learning
    test_queries = [
        # Similar queries to establish patterns
        ("How fast does light travel?", "direct_match"),
        ("What is the speed of light?", "similar_to_previous"),
        ("Tell me about light's velocity", "related_concept"),
        
        # Different domain to test adaptation
        ("What is DNA?", "domain_shift"),
        ("How does genetic information work?", "complex_related"),
        
        # Back to physics to see if it remembers
        ("Explain Newton's second law", "return_to_physics"),
        ("What connects force and acceleration?", "conceptual_link"),
        
        # Cross-domain to test multi-hop
        ("How do neural networks relate to biology?", "cross_domain"),
    ]
    
    print("2. Running Queries and Observing Learning:")
    print("-" * 50)
    
    for i, (query, query_type) in enumerate(test_queries, 1):
        print(f"\nQuery {i}: '{query}' ({query_type})")
        
        # Get current strategy parameters
        config_before = agent._get_config_snapshot()
        
        # Process query
        result = agent.process_question(query)
        
        # Get updated strategy parameters
        config_after = agent._get_config_snapshot()
        
        # Show learning adjustments
        if config_before != config_after:
            print("  Strategy adjusted:")
            for param, old_val in config_before.items():
                new_val = config_after[param]
                if old_val != new_val:
                    print(f"    {param}: {old_val} → {new_val}")
        
        print(f"  Quality: {result.reasoning_quality:.3f}")
        print(f"  Spike: {result.has_spike}")
        print(f"  Response preview: {result.response[:80]}...")
        
        # Show learning progress periodically
        if i % 3 == 0:
            report = agent.strategy_optimizer.report_performance()
            print(f"\n  Learning Progress:")
            print(f"    Performance: {report['current_performance']:.3f}")
            print(f"    Exploration rate: {report['exploration_rate']:.3f}")
            print(f"    Total patterns: {report['total_patterns']}")


def test_pattern_similarity():
    """Test how similar patterns influence strategy"""
    
    print("\n\n=== Pattern Similarity Test ===\n")
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    config = load_config(preset="adaptive_learning")
    agent = MainAgent(config)
    agent.initialize()
    
    # Add knowledge
    for i in range(10):
        agent.add_knowledge(f"Fact {i}: Some information about topic {i % 3}")
    
    print("Testing how similar questions benefit from learned patterns:\n")
    
    # Ask similar questions
    similar_queries = [
        "Tell me about topic 1",
        "What do you know about topic 1?",
        "Explain topic 1 to me",
        "Information on topic 1 please",
    ]
    
    for i, query in enumerate(similar_queries, 1):
        result = agent.process_question(query)
        
        # Find similar patterns
        similar_patterns = agent.pattern_logger.find_similar_patterns(query)
        
        print(f"Query {i}: '{query}'")
        print(f"  Found {len(similar_patterns)} similar patterns")
        if similar_patterns:
            best_pattern, similarity = similar_patterns[0]
            print(f"  Best match: '{best_pattern.question}' (similarity: {similarity:.3f})")
            print(f"  Previous reward: {best_pattern.reward:.3f}")
        print(f"  Current quality: {result.reasoning_quality:.3f}\n")


def test_strategy_performance():
    """Test different strategy performance tracking"""
    
    print("\n\n=== Strategy Performance Analysis ===\n")
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    config = load_config(preset="adaptive_learning")
    agent = MainAgent(config)
    agent.initialize()
    
    # Add varied knowledge
    for i in range(20):
        agent.add_knowledge(f"Knowledge item {i} with various complexity levels")
    
    # Run queries to gather performance data
    print("Running queries to analyze strategy performance...\n")
    
    queries = [
        "Simple direct question",
        "Complex multi-part question that requires deep reasoning",
        "Another simple query",
        "Question requiring graph traversal and associations",
    ]
    
    for query in queries:
        agent.process_question(query)
    
    # Get performance report
    performance = agent.pattern_logger.get_strategy_performance()
    
    print("Strategy Performance Report:")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Avg Reward':<12} {'Spike Rate':<12} {'Count':<8}")
    print("-" * 60)
    
    for strategy, metrics in performance.items():
        print(
            f"{strategy:<20} "
            f"{metrics['avg_reward']:<12.3f} "
            f"{metrics['spike_rate']:<12.3f} "
            f"{metrics['count']:<8}"
        )
    
    print("\nInsights:")
    print("- High threshold vs Low threshold performance")
    print("- Single hop vs Multi-hop effectiveness")
    print("- Path decay impact on retrieval quality")


if __name__ == "__main__":
    test_adaptive_learning()
    test_pattern_similarity()
    test_strategy_performance()
    
    print("\n\n=== Summary ===")
    print("Adaptive learning enables:")
    print("- Strategy optimization based on rewards")
    print("- Pattern recognition for similar queries")
    print("- Performance tracking across strategies")
    print("- Exploration/exploitation balance")
    print("- Continuous improvement over time")
