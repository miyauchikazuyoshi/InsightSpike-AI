#!/usr/bin/env python3
"""
Quick test of InsightSpike functionality
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


def quick_test():
    """Quick test with minimal setup"""
    print("🧪 Quick InsightSpike Test")
    
    # Create config
    config = InsightSpikeConfig()
    config.core.model_name = "distilgpt2"
    config.core.max_tokens = 50  # Reduce for faster testing
    
    # Initialize agent
    print("📦 Initializing agent...")
    agent = MainAgent(config=config)
    
    if not agent.initialize():
        print("❌ Failed to initialize agent")
        return
    
    print("✅ Agent initialized")
    
    # Add one episode
    print("\n📝 Adding test episode...")
    success = agent.l2_memory.store_episode(
        text="Energy is the capacity to do work.",
        c_value=0.5
    )
    print(f"✅ Episode stored: {success}")
    
    # Process simple question
    print("\n❓ Processing question...")
    result = agent.process_question(
        "What is energy?",
        max_cycles=1,
        verbose=False
    )
    
    print("\n📊 Result structure:")
    print(f"Type: {type(result)}")
    if isinstance(result, dict):
        print("Keys:", list(result.keys()))
        print(f"Response: {result.get('response', 'No response')[:100]}...")
        print(f"Spike detected: {result.get('spike_detected', False)}")
        print(f"Reasoning quality: {result.get('reasoning_quality', 0.0):.3f}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    quick_test()