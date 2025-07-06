#!/usr/bin/env python3
"""
Setup TinyLlama Model
=====================

Downloads and caches TinyLlama model for InsightSpike-AI.
This ensures the model is available when needed.
"""

import os
import sys
from pathlib import Path


def setup_tinyllama():
    """Download and cache TinyLlama model."""
    print("=== Setting up TinyLlama for InsightSpike-AI ===\n")
    
    try:
        # Check if transformers is installed
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("❌ transformers not installed!")
            print("Please run: pip install transformers torch")
            return False
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        print(f"📥 Downloading {model_name}...")
        print("This may take a few minutes on first run...\n")
        
        # Download tokenizer
        print("1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded")
        
        # Download model
        print("\n2. Downloading model weights...")
        print("   Model size: ~1.1GB")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Optimize memory usage
        )
        print("✓ Model downloaded")
        
        # Test the model
        print("\n3. Testing model...")
        test_prompt = "Question: What is the capital of France?\nAnswer:"
        
        inputs = tokenizer(test_prompt, return_tensors="pt", max_length=100, truncation=True)
        
        # Generate with minimal tokens for testing
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Model test successful!")
        print(f"   Test response: {response[len(test_prompt):].strip()[:50]}...")
        
        # Clean up test model from memory
        del model
        
        # Get cache info
        from transformers import TRANSFORMERS_CACHE
        cache_dir = Path(TRANSFORMERS_CACHE)
        
        print(f"\n✅ Setup complete!")
        print(f"📁 Model cached at: {cache_dir}")
        
        # Estimate cache size
        model_files = list(cache_dir.glob("**/pytorch_model*.bin"))
        if model_files:
            total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
            print(f"💾 Cache size: ~{total_size:.1f} GB")
        
        print("\n🚀 TinyLlama is ready for use with InsightSpike-AI!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_tinyllama_status():
    """Check if TinyLlama is already cached."""
    try:
        from transformers import TRANSFORMERS_CACHE
        cache_dir = Path(TRANSFORMERS_CACHE)
        
        # Look for TinyLlama files
        pattern = "*TinyLlama*1.1B*"
        tinyllama_files = list(cache_dir.glob(f"**/{pattern}"))
        
        if tinyllama_files:
            print("✓ TinyLlama appears to be already cached")
            return True
        else:
            print("✗ TinyLlama not found in cache")
            return False
            
    except Exception:
        return False


if __name__ == "__main__":
    print("TinyLlama Setup for InsightSpike-AI")
    print("=" * 40)
    
    # Check current status
    if check_tinyllama_status():
        print("\nTinyLlama is already set up!")
        response = input("Do you want to re-download anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup skipped.")
            sys.exit(0)
    
    # Run setup
    success = setup_tinyllama()
    sys.exit(0 if success else 1)