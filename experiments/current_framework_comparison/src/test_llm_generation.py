#!/usr/bin/env python3
"""
Test LLM generation to understand the issue
"""

import json
from pathlib import Path
import sys
import os

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from transformers import pipeline


def test_distilgpt2():
    """Test DistilGPT2 directly"""
    print("🔍 Testing DistilGPT2 Generation")
    
    # Create pipeline
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1  # CPU
    )
    
    # Test 1: Simple prompt
    prompt1 = "Question: What is energy?\nAnswer:"
    print(f"\n📝 Test 1: {prompt1}")
    
    result1 = generator(
        prompt1,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    
    generated = result1[0]["generated_text"]
    print(f"Generated: {generated}")
    print(f"Response only: {generated[len(prompt1):].strip()}")
    
    # Test 2: With formatting (like in the code)
    prompt2 = """<|system|>
You are a helpful AI assistant. Answer the question based on the provided context.

<|user|>
Question: What is energy?

<|assistant|>
"""
    
    print(f"\n📝 Test 2: With special tokens")
    result2 = generator(
        prompt2,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    
    generated2 = result2[0]["generated_text"]
    print(f"Generated: {generated2[:200]}...")
    print(f"Response only: {generated2[len(prompt2):].strip()[:100]}...")
    
    # Test 3: Simple context format
    prompt3 = """Context: Energy is the capacity to do work.

Question: What is energy?

Answer:"""
    
    print(f"\n📝 Test 3: Simple context format")
    result3 = generator(
        prompt3,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    
    generated3 = result3[0]["generated_text"]
    print(f"Generated: {generated3}")
    response3 = generated3[len(prompt3):].strip()
    print(f"Response only: {response3}")
    
    # Test if response makes sense
    if response3 and not response3.startswith("You are") and not response3.startswith("<|"):
        print("\n✅ This format works well!")
    else:
        print("\n❌ Response doesn't look right")
    
    print("\n💡 Conclusion: DistilGPT2 works better with simple prompts without special tokens")


if __name__ == "__main__":
    test_distilgpt2()