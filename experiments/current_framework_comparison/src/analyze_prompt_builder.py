#!/usr/bin/env python3
"""
Analyze Layer4 prompt builder output directly
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
from insightspike.core.layers.layer4_prompt_builder import L4PromptBuilder
from transformers import AutoTokenizer


def analyze_prompt_builder():
    """Analyze what Layer4 prompt builder produces"""
    print("🔍 Analyzing Layer4 Prompt Builder Output")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Create config
    config = InsightSpikeConfig()
    
    # Initialize prompt builder
    builder = L4PromptBuilder(config)
    
    # Create sample context
    context = {
        "retrieved_documents": [
            {
                "text": "Energy is the capacity to do work.",
                "c_value": 0.8,
                "similarity": 0.9
            },
            {
                "text": "Information is the reduction of uncertainty.",
                "c_value": 0.7,
                "similarity": 0.85
            },
            {
                "text": "Entropy measures disorder in a system.",
                "c_value": 0.75,
                "similarity": 0.8
            }
        ],
        "graph_analysis": {
            "metrics": {
                "delta_ged": -0.15,
                "delta_ig": 0.25
            },
            "spike_detected": True,
            "reasoning_quality": 0.85
        },
        "reasoning_quality": 0.85
    }
    
    question = "What is energy?"
    
    print("\n📋 Testing build_prompt method:")
    prompt = builder.build_prompt(context, question)
    
    print(f"\n📊 Prompt Stats:")
    print(f"  - Character count: {len(prompt)}")
    
    tokens = tokenizer.encode(prompt)
    print(f"  - Token count: {len(tokens)}")
    print(f"  - Model limit: 1024 tokens")
    
    if len(tokens) > 1024:
        print("  ⚠️  EXCEEDS MODEL LIMIT!")
        print(f"  - Overflow: {len(tokens) - 1024} tokens")
    elif len(tokens) > 900:
        print("  ⚠️  WARNING: Very close to limit!")
    else:
        print("  ✅ Within model limits")
    
    print("\n📄 Prompt sections:")
    sections = prompt.split("\n\n")
    for i, section in enumerate(sections):
        tokens_in_section = len(tokenizer.encode(section))
        print(f"\nSection {i+1}:")
        print(f"  - First 100 chars: {section[:100]}...")
        print(f"  - Character count: {len(section)}")
        print(f"  - Token count: {tokens_in_section}")
    
    print("\n🔧 Testing build_simple_prompt method:")
    simple_docs = [doc["text"] for doc in context["retrieved_documents"]]
    simple_prompt = builder.build_simple_prompt(simple_docs, question)
    
    simple_tokens = tokenizer.encode(simple_prompt)
    print(f"\n📊 Simple Prompt Stats:")
    print(f"  - Character count: {len(simple_prompt)}")
    print(f"  - Token count: {len(simple_tokens)}")
    
    print("\n💡 Recommendation:")
    if len(tokens) > 1024:
        print("  - The full prompt builder creates prompts that are TOO LONG for DistilGPT2")
        print("  - Should use build_simple_prompt or create a truncated version")
    else:
        print("  - The prompts are within limits, issue might be elsewhere")
    
    # Show the actual simple prompt
    print("\n📝 Simple prompt content:")
    print("=" * 50)
    print(simple_prompt)
    print("=" * 50)


if __name__ == "__main__":
    analyze_prompt_builder()