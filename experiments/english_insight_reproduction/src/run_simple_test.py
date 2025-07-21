#!/usr/bin/env python3
"""Simple test of InsightSpike with DistilGPT2"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.insightspike.config import load_config
from src.insightspike.implementations.agents.main_agent import MainAgent

# Load config
config = load_config(config_path=str(Path(__file__).parent.parent / "config_experiment.yaml"))

# Initialize agent
print("Initializing MainAgent...")
agent = MainAgent(config)
print("✓ Agent initialized")

# Add some knowledge
print("\nAdding knowledge...")
knowledge_items = [
    "Energy is the capacity to do work.",
    "Energy can change forms but cannot be created or destroyed.",
    "Entropy is a measure of energy degradation.",
    "Information and entropy have a deep mathematical relationship."
]

for item in knowledge_items:
    agent.add_knowledge(item)
    print(f"  Added: {item[:50]}...")

# Test question
print("\nTesting question...")
result = agent.process_question("How are energy and information related?")

print(f"\nResult:")
print(f"  Has spike: {result.get('has_spike', False)}")
print(f"  Spike confidence: {result.get('spike_info', {}).get('confidence', 0):.3f}")
print(f"  Response: {result.get('response', 'No response')[:200]}...")

print("\n✅ Test complete!")