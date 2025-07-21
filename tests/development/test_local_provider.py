#!/usr/bin/env python3
import os

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
