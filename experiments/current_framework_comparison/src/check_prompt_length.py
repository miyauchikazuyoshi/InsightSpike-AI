#!/usr/bin/env python3
"""
Check prompt length in InsightSpike
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
from insightspike.core.agents.main_agent import MainAgent
from transformers import AutoTokenizer


def check_prompt_length():
    """Check the actual prompt being sent to LLM"""
    print("🔍 Checking InsightSpike Prompt Length")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Create config
    config = InsightSpikeConfig()
    config.core.model_name = "distilgpt2"
    config.core.max_tokens = 50
    
    # Initialize agent
    print("\n📦 Initializing agent...")
    agent = MainAgent(config=config)
    
    if not agent.initialize():
        print("❌ Failed to initialize agent")
        return
    
    # Add episodes
    print("\n📝 Adding episodes...")
    episodes = [
        "Energy is the capacity to do work.",
        "Information is the reduction of uncertainty.",
        "Entropy measures disorder in a system."
    ]
    
    for ep in episodes:
        agent.l2_memory.store_episode(text=ep, c_value=0.5)
    
    # Hook into the LLM to capture prompt
    original_generate_response = agent.l4_llm.generate_response
    captured_prompts = []
    
    def capture_generate_response(context, question):
        # Reconstruct the full prompt
        full_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        captured_prompts.append(full_prompt)
        print(f"\n📋 Captured context (first 500 chars):\n{context[:500]}...")
        print(f"\n❓ Question: {question}")
        print(f"\n📊 Prompt stats:")
        print(f"  - Context character count: {len(context)}")
        print(f"  - Full prompt character count: {len(full_prompt)}")
        tokens = tokenizer.encode(full_prompt)
        print(f"  - Token count: {len(tokens)}")
        print(f"  - Max model context: 1024 tokens (DistilGPT2)")
        
        if len(tokens) > 900:
            print("  ⚠️  WARNING: Prompt is very long! May cause issues.")
        
        # Show token distribution
        print(f"\n🔢 Token IDs (first 20): {tokens[:20]}")
        print(f"🔢 Token IDs (last 20): {tokens[-20:]}")
        
        return original_generate_response(context, question)
    
    agent.l4_llm.generate_response = capture_generate_response
    
    # Process question
    print("\n\n❓ Processing question: 'What is energy?'")
    result = agent.process_question(
        "What is energy?",
        max_cycles=1,
        verbose=False
    )
    
    # Analyze captured prompts
    print("\n\n📊 Prompt Analysis Summary:")
    for i, prompt in enumerate(captured_prompts):
        tokens = tokenizer.encode(prompt)
        print(f"\nPrompt {i+1}:")
        print(f"  - Characters: {len(prompt)}")
        print(f"  - Tokens: {len(tokens)}")
        print(f"  - Exceeds limit: {'YES' if len(tokens) > 1024 else 'NO'}")
        
        # Check what percentage is instructions vs content
        if "Based on the following" in prompt:
            instruction_end = prompt.find("Based on the following")
            instruction_ratio = instruction_end / len(prompt)
            print(f"  - Instruction ratio: {instruction_ratio:.1%}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    check_prompt_length()