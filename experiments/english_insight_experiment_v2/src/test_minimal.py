#!/usr/bin/env python3
"""Minimal test to isolate the issue"""

import numpy as np
from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
from insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager, MemoryConfig, MemoryMode

print("1. Creating config...")
config = ConfigPresets.experiment()
config.llm.temperature = 0.7
config.llm.max_tokens = 100

print("2. Creating datastore...")
datastore = FileSystemDataStore(base_path="./data")

print("3. Creating memory manager...")
memory_config = MemoryConfig(
    mode=MemoryMode.SCALABLE,
    embedding_dim=384,
    use_graph_integration=True,
    use_importance_scoring=True
)
memory = L2MemoryManager(config=memory_config, legacy_config=config)

print("4. Adding some episodes...")
memory.store_episode("Test episode 1", c_value=0.5, metadata={"phase": 1})
memory.store_episode("Test episode 2", c_value=0.7, metadata={"phase": 2})
print(f"   Memory has {len(memory.episodes)} episodes")

print("5. Creating agent...")
agent = MainAgent(config=config, datastore=datastore)

print("6. Replacing memory...")
agent.l2_memory = memory

print("7. Initializing agent...")
agent.initialize()

print("8. Processing question...")
result = agent.process_question("What is entropy?", max_cycles=1)

print("9. Result:")
print(f"   Response: {result.response[:100]}...")
print(f"   Spike detected: {result.spike_detected}")

print("\nDone!")