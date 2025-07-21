#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.insightspike.config import load_config
config = load_config(config_path="experiments/english_insight_reproduction/config_experiment.yaml")

print("Config loaded, creating layers...", flush=True)

# Test layer initialization
print("1. Testing Layer2 (Memory)...", flush=True)
from src.insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager
layer2 = L2MemoryManager(config)
print("   ✓ Layer2 created", flush=True)

print("2. Testing Layer3 (Graph)...", flush=True)
from src.insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
layer3 = L3GraphReasoner(config)
print("   ✓ Layer3 created", flush=True)

print("3. Testing Layer4 (LLM)...", flush=True)
from src.insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface
layer4 = L4LLMInterface(config)
print("   ✓ Layer4 created", flush=True)

print("\nAll layers initialized successfully!", flush=True)