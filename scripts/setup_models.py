#!/usr/bin/env python3
"""
Setup Models for InsightSpike-AI
================================

Downloads and caches all required models:
- Sentence Transformers for embeddings
- TinyLlama for text generation
"""

import os
import sys
from pathlib import Path


def setup_sentence_transformer():
    """Download and cache sentence transformer model."""
    print("\n=== Setting up Sentence Transformer ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = 'all-MiniLM-L6-v2'
        print(f"📥 Downloading {model_name}...")
        
        # This will download and cache the model
        model = SentenceTransformer(model_name)
        
        # Test it
        test_embedding = model.encode("Test sentence")
        print(f"✓ Model downloaded (embedding dim: {len(test_embedding)})")
        
        del model
        return True
        
    except ImportError:
        print("❌ sentence-transformers not installed!")
        print("Please run: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def setup_tinyllama():
    """Download and cache TinyLlama model."""
    print("\n=== Setting up TinyLlama ===")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"📥 Downloading {model_name}...")
        print("   This may take a few minutes (~1.1GB)...")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Download model with progress
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✓ Model downloaded")
        
        # Quick test
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=5)
        
        del model
        return True
        
    except ImportError:
        print("❌ transformers not installed!")
        print("Please run: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def check_cache_status():
    """Check what models are already cached."""
    print("=== Checking Model Cache ===")
    
    cached_models = []
    
    # Check sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
        st_cache = Path.home() / ".cache" / "torch" / "sentence_transformers"
        if st_cache.exists() and list(st_cache.glob("**/config.json")):
            cached_models.append("Sentence Transformer (all-MiniLM-L6-v2)")
    except:
        pass
    
    # Check transformers
    try:
        from transformers import TRANSFORMERS_CACHE
        cache_dir = Path(TRANSFORMERS_CACHE)
        if list(cache_dir.glob("**/TinyLlama*")):
            cached_models.append("TinyLlama-1.1B-Chat")
    except:
        pass
    
    if cached_models:
        print("\n✓ Found cached models:")
        for model in cached_models:
            print(f"  - {model}")
    else:
        print("\n✗ No models found in cache")
    
    return len(cached_models)


def main():
    """Main setup function."""
    print("InsightSpike-AI Model Setup")
    print("=" * 40)
    
    # Check what's already cached
    cached_count = check_cache_status()
    
    if cached_count >= 2:
        print("\nAll models appear to be cached!")
        response = input("Do you want to re-download anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup skipped.")
            return
    
    print("\nThis script will download and cache:")
    print("1. Sentence Transformer for embeddings (~90MB)")
    print("2. TinyLlama for text generation (~1.1GB)")
    print("\nTotal download: ~1.2GB")
    
    response = input("\nProceed with download? (Y/n): ")
    if response.lower() == 'n':
        print("Setup cancelled.")
        return
    
    # Setup models
    success = True
    
    if not setup_sentence_transformer():
        success = False
    
    if not setup_tinyllama():
        success = False
    
    # Summary
    print("\n" + "=" * 40)
    if success:
        print("✅ All models successfully set up!")
        print("\nModels are cached and ready for use.")
        print("You won't need to download them again.")
    else:
        print("⚠️  Some models failed to download.")
        print("Please check the error messages above.")
    
    # Add to requirements if needed
    print("\n💡 Tip: Add this to your project setup:")
    print("   python scripts/setup_models.py")


if __name__ == "__main__":
    main()