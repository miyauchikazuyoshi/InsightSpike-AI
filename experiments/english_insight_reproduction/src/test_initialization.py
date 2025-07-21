#!/usr/bin/env python3
"""Test initialization of components"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=== Testing InsightSpike Initialization ===")

# 1. Test config loading
print("\n1. Testing config loading...")
try:
    from src.insightspike.config import load_config
    config = load_config(config_path=str(Path(__file__).parent.parent / "config_experiment.yaml"))
    print("✓ Config loaded successfully")
    print(f"  - Environment: {config.environment}")
    print(f"  - LLM Provider: {config.llm.provider}")
    print(f"  - LLM Model: {config.llm.model}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Test SQLite datastore
print("\n2. Testing SQLite datastore...")
try:
    from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
    db_path = Path(__file__).parent.parent / "data" / "test.db"
    db_path.parent.mkdir(exist_ok=True)
    datastore = SQLiteDataStore(str(db_path))
    print("✓ SQLite datastore initialized")
except Exception as e:
    print(f"✗ SQLite initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Test MainAgent initialization
print("\n3. Testing MainAgent initialization...")
try:
    from src.insightspike.implementations.agents.main_agent import MainAgent
    agent = MainAgent(config)
    print("✓ MainAgent initialized successfully")
    
    # Test layers
    print("\n4. Testing layer initialization...")
    print(f"  - Layer2 (Memory): {agent.layer2_memory_manager is not None}")
    print(f"  - Layer3 (Graph): {agent.layer3_graph_reasoner is not None}")
    print(f"  - Layer4 (LLM): {agent.layer4_llm_interface is not None}")
    
except Exception as e:
    print(f"✗ MainAgent initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test simple knowledge addition
print("\n5. Testing knowledge addition...")
try:
    result = agent.add_knowledge("Test knowledge: Energy is the capacity to do work.")
    print("✓ Knowledge added successfully")
    print(f"  - Episode ID: {result.get('episode_id', 'N/A')}")
except Exception as e:
    print(f"✗ Knowledge addition failed: {e}")
    import traceback
    traceback.print_exc()

# 5. Test simple question
print("\n6. Testing question processing...")
try:
    result = agent.process_question("What is energy?")
    print("✓ Question processed successfully")
    print(f"  - Has spike: {result.get('has_spike', False)}")
    print(f"  - Response preview: {result.get('response', 'No response')[:100]}...")
except Exception as e:
    print(f"✗ Question processing failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Initialization test complete ===")
print("\nIf all tests passed, the experiment should work!")