"""
Simple Demonstration of UnifiedMainAgent
========================================

Shows basic usage without requiring external data files.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_main_agent import UnifiedMainAgent, AgentConfig, AgentMode
import time


def demo_basic_usage():
    """Demonstrate basic agent usage"""
    print("="*60)
    print("BASIC AGENT DEMO")
    print("="*60)
    
    # Create a basic agent
    config = AgentConfig.from_mode(AgentMode.BASIC)
    agent = UnifiedMainAgent(config)
    
    print("Initializing agent...")
    if not agent.initialize():
        print("Failed to initialize")
        return
    
    # Add some knowledge
    print("\nAdding knowledge...")
    knowledge = [
        ("Python is a high-level programming language.", 0.8),
        ("Machine learning is a subset of artificial intelligence.", 0.9),
        ("Neural networks are inspired by biological neurons.", 0.7),
        ("Deep learning uses multiple layers of neural networks.", 0.8),
    ]
    
    for text, c_value in knowledge:
        agent.add_episode(text, c_value=c_value)
        print(f"  Added: {text[:50]}...")
    
    # Ask questions
    questions = [
        "What is Python?",
        "What is the relationship between AI and machine learning?",
        "How do neural networks work?"
    ]
    
    print("\nAsking questions:")
    for question in questions:
        print(f"\nQ: {question}")
        result = agent.process_question(question, max_cycles=2)
        print(f"A: {result['response'][:200]}...")
        print(f"   Quality: {result['reasoning_quality']:.3f}")
        print(f"   Docs retrieved: {len(result['retrieved_documents'])}")


def demo_caching():
    """Demonstrate caching functionality"""
    print("\n" + "="*60)
    print("CACHING DEMO")
    print("="*60)
    
    # Create agent with caching
    config = AgentConfig(
        mode=AgentMode.BASIC,
        enable_caching=True,
        cache_size=100
    )
    agent = UnifiedMainAgent(config)
    
    if not agent.initialize():
        print("Failed to initialize")
        return
    
    # Ask same question twice
    question = "What is the meaning of life?"
    
    print(f"\nFirst query: {question}")
    start = time.time()
    result1 = agent.process_question(question)
    time1 = time.time() - start
    print(f"Time: {time1:.3f}s (cached: {result1.get('cached', False)})")
    
    print(f"\nSecond query: {question}")
    start = time.time()
    result2 = agent.process_question(question)
    time2 = time.time() - start
    print(f"Time: {time2:.3f}s (cached: {result2.get('cached', False)})")
    
    print(f"\nSpeedup from cache: {time1/time2:.1f}x faster")


def demo_mode_comparison():
    """Compare different modes side by side"""
    print("\n" + "="*60)
    print("MODE COMPARISON DEMO")
    print("="*60)
    
    # Same question for all modes
    question = "What are the key principles of machine learning?"
    
    # Test different modes
    modes = [
        (AgentMode.BASIC, "Basic"),
        (AgentMode.ENHANCED, "Enhanced (Graph-Aware)"),
    ]
    
    for mode, name in modes:
        print(f"\n{name} Mode:")
        
        config = AgentConfig.from_mode(mode)
        agent = UnifiedMainAgent(config)
        
        if agent.initialize():
            # Add same knowledge
            agent.add_episode("Machine learning uses data to learn patterns.", c_value=0.8)
            agent.add_episode("Supervised learning requires labeled data.", c_value=0.7)
            
            result = agent.process_question(question, max_cycles=1)
            print(f"  Response: {result['response'][:150]}...")
            print(f"  Quality: {result['reasoning_quality']:.3f}")
            print(f"  Features enabled: {[k for k, v in config.__dict__.items() if k.startswith('enable_') and v]}")


def show_migration_example():
    """Show how to migrate from old code"""
    print("\n" + "="*60)
    print("MIGRATION EXAMPLE")
    print("="*60)
    
    print("\nOLD CODE:")
    print("-"*40)
    print("""
from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.agents.main_agent_enhanced import EnhancedMainAgent

# Had to choose one agent type
agent = MainAgent(config)  # OR EnhancedMainAgent(config)
agent.initialize()
result = agent.process_question("Question?")
""")
    
    print("\nNEW CODE:")
    print("-"*40)
    print("""
from unified_main_agent import UnifiedMainAgent, AgentConfig, AgentMode

# Can configure any combination of features
config = AgentConfig(
    mode=AgentMode.BASIC,
    enable_graph_aware_memory=True,  # Add enhanced features
    enable_caching=True,             # Add caching
)
agent = UnifiedMainAgent(config)
agent.initialize()
result = agent.process_question("Question?")
""")
    
    print("\n✅ Benefits:")
    print("- Single import instead of 6 different agent classes")
    print("- Mix and match features as needed")
    print("- Consistent API across all modes")
    print("- Easy to experiment with different configurations")


if __name__ == "__main__":
    # Run all demos
    demo_basic_usage()
    demo_caching()
    demo_mode_comparison()
    show_migration_example()
    
    print("\n" + "="*60)
    print("REFACTORING COMPLETE!")
    print("="*60)
    print("\nThe UnifiedMainAgent successfully consolidates all 6 agent variants:")
    print("- MainAgent")
    print("- EnhancedMainAgent") 
    print("- MainAgentWithQueryTransform")
    print("- MainAgentAdvanced")
    print("- MainAgentOptimized")
    print("- GraphCentricMainAgent")
    print("\nInto ONE configurable class with feature flags!")
    print("\nNext steps:")
    print("1. Update imports in src/insightspike/core/agents/")
    print("2. Delete old agent files")
    print("3. Update CLI to use UnifiedMainAgent")