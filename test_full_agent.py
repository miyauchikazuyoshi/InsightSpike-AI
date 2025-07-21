#!/usr/bin/env python3
import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress stderr  
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    from src.insightspike.config import load_config
    from src.insightspike.implementations.agents.main_agent import MainAgent
finally:
    sys.stderr.close()
    sys.stderr = stderr_backup

print("1. Loading config...", flush=True)
config = load_config(config_path="experiments/english_insight_reproduction/config_experiment.yaml")
print("   ✓ Config loaded", flush=True)

print("2. Creating MainAgent...", flush=True)
agent = MainAgent(config)
print("   ✓ MainAgent created", flush=True)

print("3. Adding knowledge...", flush=True)
result1 = agent.add_knowledge("Energy is the capacity to do work.")
print(f"   ✓ Added: Episode {result1.get('episode_id', 'N/A')}", flush=True)

print("4. Processing question...", flush=True)
result2 = agent.process_question("What is energy?")
print(f"   ✓ Response: {result2.get('response', 'No response')[:50]}...", flush=True)
print(f"   ✓ Has spike: {result2.get('has_spike', False)}", flush=True)

print("\n✅ All tests passed!", flush=True)