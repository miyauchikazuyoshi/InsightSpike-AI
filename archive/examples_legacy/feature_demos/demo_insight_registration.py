#!/usr/bin/env python3
"""
Test Insight Auto-Registration
==============================

Tests the automatic insight registration when spikes are detected.
"""

from insightspike.config import load_config
from insightspike.detection.insight_registry import get_insight_registry


def test_insight_registration():
    """Test automatic insight registration on spike detection"""
    
    print("=== Testing Insight Auto-Registration ===\n")
    
    # Create agent with experiment config
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    config = load_config(preset="experiment")
    agent = MainAgent(config)
    
    # Get insight registry
    registry = get_insight_registry()
    initial_count = len(registry.insights)
    print(f"Initial insights in registry: {initial_count}")
    
    # Add some knowledge that might trigger insights
    knowledge_base = [
        "Water freezes at 0 degrees Celsius.",
        "Ice is the solid form of water.",
        "Steam is water vapor at high temperature.",
        "The three states of water are ice, liquid, and vapor.",
        "Temperature determines the state of water.",
    ]
    
    for knowledge in knowledge_base:
        agent.add_knowledge(knowledge)
    
    # Ask questions that might generate insights
    test_questions = [
        "How does temperature affect the state of water?",
        "What connects ice, water, and steam?",
        "Why does water change from liquid to solid at 0 degrees?",
        "What is the relationship between the three states of water?",
    ]
    
    print("\nProcessing questions to trigger insights...\n")
    
    for question in test_questions:
        print(f"Q: {question}")
        result = agent.process_question(question, verbose=False)
        
        print(f"Response: {result.response[:100]}...")
        print(f"Spike detected: {result.spike_detected}")
        print(f"Reasoning quality: {result.reasoning_quality:.3f}")
        
        # Check if new insights were registered
        current_count = len(registry.insights)
        if current_count > initial_count:
            new_insights = current_count - initial_count
            print(f"✓ {new_insights} new insights registered!")
            initial_count = current_count  # Update for next iteration
        
        print("-" * 50)
    
    # Display insight statistics
    print("\n=== Insight Registry Statistics ===")
    stats = registry.get_optimization_stats()
    
    print(f"Total insights: {stats.get('total_insights', 0)}")
    print(f"Average quality score: {stats.get('avg_quality_score', 0):.3f}")
    print(f"Average GED improvement: {stats.get('avg_ged_improvement', 0):.3f}")
    print(f"Average IG improvement: {stats.get('avg_ig_improvement', 0):.3f}")
    
    # Show recent insights
    recent_insights = registry.get_recent_insights(limit=5)
    if recent_insights:
        print("\n=== Recent Insights ===")
        for insight in recent_insights:
            print(f"- {insight.text}")
            print(f"  Quality: {insight.quality_score:.3f}, Type: {insight.relationship_type}")
    
    # Test insight retrieval for future queries
    print("\n=== Testing Insight Retrieval ===")
    test_query = "What happens to water at different temperatures?"
    
    result = agent.process_question(test_query, verbose=True)
    
    # Check if insights were used in the response
    retrieved_docs = result.retrieved_documents
    insight_docs = [doc for doc in retrieved_docs if doc.get("is_insight", False)]
    
    if insight_docs:
        print(f"\n✓ {len(insight_docs)} insights were retrieved and used!")
        for doc in insight_docs:
            print(f"  - {doc['text']}")
    else:
        print("\n⚠️  No insights were retrieved for this query")


if __name__ == "__main__":
    test_insight_registration()
