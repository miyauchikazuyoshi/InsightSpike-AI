#!/usr/bin/env python3
"""
Patch the LLM provider to work better with DistilGPT2
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


def patch_llm_format_prompt():
    """Patch the LLM provider to use simpler prompts for DistilGPT2"""
    
    # Import after path is set
    from insightspike.core.layers.layer4_llm_provider import L4LLMProvider
    
    # Save original method
    original_format_prompt = L4LLMProvider._format_prompt
    
    # Create new method that doesn't use special tokens for DistilGPT2
    def simple_format_prompt(self, prompt: str) -> str:
        """Format prompt for DistilGPT2 - no special tokens"""
        # For DistilGPT2, just return the prompt as-is
        if hasattr(self, 'model_name') and 'distilgpt2' in str(self.model_name).lower():
            return prompt
        # For other models, use original formatting
        return original_format_prompt(self, prompt)
    
    # Monkey patch the method
    L4LLMProvider._format_prompt = simple_format_prompt
    print("✅ Patched _format_prompt method")


def test_with_patch():
    """Test InsightSpike with the patch"""
    print("🧪 Testing InsightSpike with Patch")
    
    # Apply patch
    patch_llm_format_prompt()
    
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
    
    # Process question
    print("\n❓ Processing question: 'What is energy?'")
    result = agent.process_question(
        "What is energy?",
        max_cycles=1,
        verbose=False
    )
    
    print("\n📊 Result:")
    if isinstance(result, dict):
        response = result.get('response', 'No response')
        print(f"Response: {response[:200]}...")
        print(f"Spike detected: {result.get('spike_detected', False)}")
        print(f"Reasoning quality: {result.get('reasoning_quality', 0.0):.3f}")
        
        # Check if response is reasonable
        if response and not response.startswith("You are") and not response.startswith("<|"):
            print("\n✅ Response looks reasonable!")
        else:
            print("\n⚠️  Response may still have issues")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_with_patch()