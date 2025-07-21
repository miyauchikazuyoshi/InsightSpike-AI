#!/usr/bin/env python3
"""Minimal test with distilgpt2"""

import os
import sys
import logging
from pathlib import Path

# Suppress warnings
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import silently
import sys
from io import StringIO

# Capture stderr during imports
old_stderr = sys.stderr
sys.stderr = StringIO()

try:
    from src.insightspike.config import load_config
    from src.insightspike.implementations.agents.main_agent import MainAgent
finally:
    sys.stderr = old_stderr

print("=== Minimal InsightSpike Test ===")

# Load config
config_path = Path(__file__).parent.parent / "config_experiment.yaml"
config = load_config(config_path=str(config_path))
print(f"✓ Config loaded (provider: {config.llm.provider}, model: {config.llm.model})")

# Initialize agent
print("Initializing agent...")
agent = MainAgent(config)
print("✓ Agent initialized")

# Add knowledge
print("\nAdding knowledge...")
agent.add_knowledge("Energy is the capacity to do work.")
agent.add_knowledge("Information and entropy have a deep mathematical relationship.")
print("✓ Knowledge added")

# Test question
print("\nProcessing question...")
result = agent.process_question("How are energy and information related?")

print(f"\n=== Results ===")
print(f"Has spike: {result.get('has_spike', False)}")
print(f"Response: {result.get('response', 'No response')[:100]}...")

print("\n✅ Test completed successfully!")