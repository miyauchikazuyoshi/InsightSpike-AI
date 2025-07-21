#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Step 1: Import config", flush=True)
from src.insightspike.config import load_config

print("Step 2: Load config", flush=True)
config = load_config(config_path="experiments/english_insight_reproduction/config_experiment.yaml")

print("Step 3: Import MainAgent", flush=True)
from src.insightspike.implementations.agents.main_agent import MainAgent

print("Step 4: Create agent", flush=True)
agent = MainAgent(config)

print("Step 5: Add knowledge", flush=True)
result = agent.add_knowledge("Test knowledge")
print(f"Result: {result}", flush=True)

print("Done!", flush=True)