#!/usr/bin/env python3
"""
Test Insight Configuration Settings
===================================

Demonstrates how to enable/disable insight features via configuration.
"""

from insightspike.config import load_config


def test_with_insights_enabled():
    """Test with insights enabled (default)"""
    print("=== Testing WITH Insights Enabled ===\n")
    
    config = load_config(preset="experiment")
    print(f"Insight registration enabled: {config.processing.enable_insight_registration}")
    print(f"Insight search enabled: {config.processing.enable_insight_search}")
    print(f"Max insights per query: {config.processing.max_insights_per_query}\n")
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent = MainAgent(config)
    agent.add_knowledge("Water boils at 100 degrees Celsius.")
    
    result = agent.process_question("What happens to water at high temperature?")
    print(f"Response: {result.response[:100]}...")
    print(f"Spike detected: {result.spike_detected}")
    
    # Check if insights were searched
    insight_docs = [doc for doc in result.retrieved_documents if doc.get("is_insight", False)]
    print(f"Insights in retrieval: {len(insight_docs)}")


def test_with_insights_disabled():
    """Test with insights disabled"""
    print("\n=== Testing WITH Insights Disabled ===\n")
    
    config = load_config(preset="minimal")  # Uses minimal preset
    print(f"Insight registration enabled: {config.processing.enable_insight_registration}")
    print(f"Insight search enabled: {config.processing.enable_insight_search}")
    print(f"Layer1 bypass enabled: {config.processing.enable_layer1_bypass}\n")
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent = MainAgent(config)
    agent.add_knowledge("Water boils at 100 degrees Celsius.")
    
    result = agent.process_question("What happens to water at high temperature?")
    print(f"Response: {result.response[:100]}...")
    print(f"Spike detected: {result.spike_detected}")
    
    # Check if insights were searched (should be none)
    insight_docs = [doc for doc in result.retrieved_documents if doc.get("is_insight", False)]
    print(f"Insights in retrieval: {len(insight_docs)}")


def test_custom_configuration():
    """Test with custom insight settings"""
    print("\n=== Testing with Custom Configuration ===\n")
    
    config = load_config(preset="experiment")
    # Custom settings
    config.processing.enable_insight_registration = True  # Register insights
    config.processing.enable_insight_search = False      # But don't search them
    config.processing.max_insights_per_query = 10        # Higher limit
    
    print(f"Insight registration enabled: {config.processing.enable_insight_registration}")
    print(f"Insight search enabled: {config.processing.enable_insight_search}")
    print(f"Max insights per query: {config.processing.max_insights_per_query}\n")
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent = MainAgent(config)
    agent.add_knowledge("Ice melts at 0 degrees Celsius.")
    
    result = agent.process_question("What happens to ice when heated?")
    print(f"Response: {result.response[:100]}...")
    print(f"Spike detected: {result.spike_detected}")
    
    # Insights should not be in retrieval (search disabled)
    insight_docs = [doc for doc in result.retrieved_documents if doc.get("is_insight", False)]
    print(f"Insights in retrieval: {len(insight_docs)} (should be 0)")


if __name__ == "__main__":
    test_with_insights_enabled()
    test_with_insights_disabled()
    test_custom_configuration()
    
    print("\n✓ Configuration tests completed!")
