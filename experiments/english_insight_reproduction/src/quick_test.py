#!/usr/bin/env python3
"""Quick test with 2 knowledge items and 1 question"""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

print("Quick InsightSpike Test", flush=True)
print("=" * 40, flush=True)

# Import without warnings
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    from src.insightspike.config import load_config
    from src.insightspike.implementations.agents.main_agent import MainAgent
finally:
    sys.stderr.close()
    sys.stderr = stderr_backup

# Load config
config_path = Path(__file__).parent.parent / "config_experiment.yaml"
config = load_config(config_path=str(config_path))

# Initialize
print("1. Initializing agent...")
agent = MainAgent(config)
print("   ✓ Agent ready")

# Add knowledge
print("\n2. Adding knowledge...")
k1 = agent.add_knowledge("Energy is the capacity to do work.")
print(f"   ✓ Added episode {k1.get('episode_id', 'N/A')}")

k2 = agent.add_knowledge("Information and entropy have a deep mathematical relationship.")
print(f"   ✓ Added episode {k2.get('episode_id', 'N/A')}")

# Process question
print("\n3. Processing question...")
print("   Question: How are energy and information related?")

import time
start = time.time()
result = agent.process_question("How are energy and information related?")
elapsed = time.time() - start

print(f"\n4. Results:")
print(f"   - Time: {elapsed:.2f}s")
print(f"   - Has spike: {result.get('has_spike', False)}")
print(f"   - Response: {result.get('response', 'No response')[:100]}...")

print("\n✅ Test complete!")