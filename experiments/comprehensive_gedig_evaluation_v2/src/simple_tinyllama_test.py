#!/usr/bin/env python3
"""
Simple test to verify TinyLlama works correctly.
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_tinyllama():
    """Test TinyLlama generation directly."""
    
    print("=== Testing TinyLlama ===")
    
    # Load model
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded!")
    
    # Test questions
    questions = [
        "What is 2 + 2?",
        "What color is the sky?",
        "If it rains and the ground is dry, what happens?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # Build prompt
        prompt = f"""<|system|>
You are a helpful AI assistant.
</s>
<|user|>
{question}
</s>
<|assistant|>"""
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        print(f"Response: {response}")


if __name__ == "__main__":
    test_tinyllama()