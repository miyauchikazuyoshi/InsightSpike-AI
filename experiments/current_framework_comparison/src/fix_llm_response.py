#!/usr/bin/env python3
"""
Fix the LLM response issue in InsightSpike
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

from insightspike.config import InsightSpikeConfig
from insightspike.core.layers.layer4_llm_provider import L4LLMProvider


def test_llm_directly():
    """Test LLM provider directly to understand the issue"""
    print("🔍 Testing LLM Provider Directly")
    
    # Create config
    config = InsightSpikeConfig()
    config.core.model_name = "distilgpt2"
    config.core.max_tokens = 50
    
    # Initialize LLM provider
    llm = L4LLMProvider(config)
    
    # Test simple prompt
    simple_prompt = "Question: What is energy?\nAnswer:"
    print(f"\n📝 Testing simple prompt: {simple_prompt}")
    
    response = llm.generate_response(simple_prompt, "")
    print(f"\n📊 Response type: {type(response)}")
    print(f"📊 Response content: {response}")
    
    # Test with context
    context_prompt = """Context: Energy is the capacity to do work.

Question: What is energy?

Answer:"""
    
    print(f"\n📝 Testing context prompt...")
    response2 = llm.generate_response(context_prompt, "")
    print(f"\n📊 Response type: {type(response2)}")
    print(f"📊 Response content: {response2}")
    
    # Let's check what's happening in the generate method
    print("\n🔧 Checking model output format...")
    
    # Try direct generation
    inputs = llm.tokenizer.encode(simple_prompt, return_tensors="pt")
    print(f"Input tokens: {inputs.shape}")
    
    with llm.torch.no_grad():
        outputs = llm.model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=llm.tokenizer.eos_token_id
        )
    
    generated_text = llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🤖 Raw generated text: {generated_text}")
    
    # The issue: DistilGPT2 returns the prompt + response
    # We need to extract just the response part
    if generated_text.startswith(simple_prompt):
        actual_response = generated_text[len(simple_prompt):].strip()
        print(f"\n✅ Extracted response: {actual_response}")
    else:
        print("\n❌ Prompt not found in response")
    
    print("\n💡 Solution: The LLM provider needs to strip the input prompt from the response!")


if __name__ == "__main__":
    test_llm_directly()