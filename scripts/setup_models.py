#!/usr/bin/env python3
"""
Pre-download and cache models for faster initialization
=======================================================

This script downloads commonly used models to avoid initialization delays.
"""

import os
import sys
from pathlib import Path

print("="*60)
print("InsightSpike Model Setup")
print("="*60)

# Models to pre-download
MODELS = {
    "embeddings": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2"
    ],
    "llms": [
        "distilgpt2",
        "gpt2"  # Optional: larger model
    ]
}

print("\n1. Checking transformers cache...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    
    # Get cache directory
    cache_dir = os.getenv("TRANSFORMERS_CACHE", str(Path.home() / ".cache" / "huggingface"))
    print(f"   Cache directory: {cache_dir}")
    
except ImportError as e:
    print(f"   ✗ Error: {e}")
    print("   Please run: poetry install")
    sys.exit(1)

# Download LLM models
print("\n2. Pre-downloading LLM models...")
for model_name in MODELS["llms"]:
    print(f"\n   Downloading {model_name}...")
    try:
        # Download tokenizer
        print(f"   - Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model
        print(f"   - Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Test generation
        print(f"   - Testing generation...")
        inputs = tokenizer("Test", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=10)
        
        print(f"   ✓ {model_name} ready!")
        
        # Clear from memory
        del model
        del tokenizer
        
    except Exception as e:
        print(f"   ✗ Failed to download {model_name}: {e}")

# Download embedding models
print("\n3. Pre-downloading embedding models...")
for model_name in MODELS["embeddings"]:
    print(f"\n   Downloading {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        
        # Test encoding
        test_embedding = model.encode(["test"])
        print(f"   ✓ {model_name} ready! (dim={test_embedding.shape[1]})")
        
        # Clear from memory
        del model
        
    except Exception as e:
        print(f"   ✗ Failed to download {model_name}: {e}")

print("\n" + "="*60)
print("Setup complete!")
print("="*60)

print("""
Models are now cached locally. Future initializations will be much faster!

To use in InsightSpike:
- LLMs: config.llm.model = "distilgpt2"
- Embeddings: Already configured in config

Run experiments with:
  poetry run python experiments/your_experiment.py
""")

# Create a marker file to indicate setup is complete
marker_file = Path(__file__).parent / ".models_downloaded"
marker_file.touch()
print(f"\n✓ Created marker file: {marker_file}")