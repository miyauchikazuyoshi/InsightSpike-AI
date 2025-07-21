#!/usr/bin/env python3
"""Test DistilGPT2 directly"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Testing DistilGPT2...")

try:
    from transformers import pipeline
    
    print("Creating pipeline...")
    generator = pipeline('text-generation', model='distilgpt2', device='cpu')
    
    print("Generating text...")
    result = generator("Energy is", max_length=20, num_return_sequences=1)
    
    print(f"Result: {result[0]['generated_text']}")
    print("✓ DistilGPT2 works!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()