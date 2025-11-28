#!/usr/bin/env python3
import os
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

print("Testing LocalProvider...")

from src.insightspike.providers import ProviderFactory

print("Creating LocalProvider...")
try:
    provider = ProviderFactory.create("local", {"model": "distilgpt2"})
    print("✓ LocalProvider created")

    print("\nGenerating text...")
    response = provider.generate("Energy is")
    print(f"✓ Generated: {response}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
