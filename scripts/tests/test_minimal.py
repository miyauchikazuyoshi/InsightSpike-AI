#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["ANTHROPIC_API_KEY"] = "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"

from insightspike.implementations.agents.main_agent import MainAgent

# Test 1: Baseline
print("\n=== Test 1: Baseline ===")
config1 = {
    "llm": {"provider": "mock"},  # Use mock to avoid API issues
    "graph": {"enable_message_passing": False}
}
agent1 = MainAgent(config1)
agent1.add_knowledge("Test knowledge")
result1 = agent1.process_question("Test question")
print(f"Baseline works: {hasattr(result1, 'response')}")

# Test 2: Message Passing
print("\n=== Test 2: Message Passing ===")
config2 = {
    "llm": {"provider": "mock"},
    "graph": {
        "enable_message_passing": True,
        "message_passing": {"alpha": 0.5, "iterations": 3}
    }
}
agent2 = MainAgent(config2)
agent2.add_knowledge("Test knowledge")
result2 = agent2.process_question("Test question")
print(f"Message passing works: {hasattr(result2, 'response')}")

# Check if message passing was used
if hasattr(agent2.l3_graph, 'message_passing_enabled'):
    print(f"Message passing enabled: {agent2.l3_graph.message_passing_enabled}")

print("\n✓ All tests completed")