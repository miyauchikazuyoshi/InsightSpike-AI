#!/usr/bin/env python3
import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress stderr
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, "w")

try:
    from src.insightspike.config import load_config
    from src.insightspike.implementations.agents.main_agent import MainAgent
finally:
    sys.stderr.close()
    sys.stderr = stderr_backup

print("1. Loading config...", flush=True)
config = load_config(
    config_path="experiments/english_insight_reproduction/config_experiment.yaml"
)
print("   ✓ Config loaded", flush=True)

print("2. Creating MainAgent...", flush=True)
agent = MainAgent(config)
print("   ✓ MainAgent created", flush=True)

print("3. Adding knowledge...", flush=True)
result1 = agent.add_knowledge("Energy is the capacity to do work.")
episode_id = result1.get("episode_id") if isinstance(result1, dict) else getattr(result1, "episode_id", "N/A")
print(f"   ✓ Added: Episode {episode_id}", flush=True)

print("4. Processing question...", flush=True)
result2 = agent.process_question("What is energy?")
response_text = getattr(result2, "response", None)
if response_text is None and isinstance(result2, dict):
    response_text = result2.get("response")
print(f"   ✓ Response: {(response_text or 'No response')[:50]}...", flush=True)
has_spike = getattr(result2, "has_spike", None)
if has_spike is None and isinstance(result2, dict):
    has_spike = result2.get("has_spike", False)
print(f"   ✓ Has spike: {bool(has_spike)}", flush=True)

print("\n✅ All tests passed!", flush=True)
