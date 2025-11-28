#!/usr/bin/env python3
import os
import sys
import time
import importlib.util
import pytest

def _has_torch():
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False

if not _has_torch():
    pytest.skip("torch/transformers stack not available in this environment", allow_module_level=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Step-by-step test", flush=True)

# Step 1: Config
print("\n1. Loading config...", flush=True)
start = time.time()
from src.insightspike.config import load_config

config = load_config(
    config_path="experiments/english_insight_reproduction/config_experiment.yaml"
)
print(f"   ✓ Config loaded ({time.time()-start:.2f}s)", flush=True)

# Step 2: ProviderFactory
print("\n2. Importing ProviderFactory...", flush=True)
start = time.time()
from src.insightspike.providers import ProviderFactory

print(f"   ✓ Imported ({time.time()-start:.2f}s)", flush=True)

# Step 3: Create provider
print("\n3. Creating LocalProvider...", flush=True)
start = time.time()
try:
    provider = ProviderFactory.create("local", config.llm.dict())
    print(f"   ✓ Provider created ({time.time()-start:.2f}s)", flush=True)
except Exception as e:
    print(f"   ✗ Failed: {e}", flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 4: Test generation
print("\n4. Testing generation...", flush=True)
start = time.time()
response = provider.generate("Test")
print(f"   ✓ Generated: '{response[:50]}...' ({time.time()-start:.2f}s)", flush=True)

print("\n✅ All tests passed!", flush=True)
