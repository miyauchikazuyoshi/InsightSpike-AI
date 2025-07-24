#!/usr/bin/env python3
"""Simple test for v2 experiment - runs with minimal questions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.insightspike.experiments.utils import create_experiment_config
from src.insightspike.implementations.agents.main_agent import MainAgent

def main():
    print("\n=== V2 Experiment Simple Test ===\n")
    
    # Test with mock provider
    print("1. Testing with Mock Provider")
    config = create_experiment_config('mock')
    agent = MainAgent(config)
    
    # Add minimal knowledge
    knowledge = [
        "The sky is blue due to Rayleigh scattering",
        "Water freezes at 0 degrees Celsius"
    ]
    
    for k in knowledge:
        result = agent.add_knowledge(k)
        print(f"  Added: {k[:30]}... (success: {result.get('success', False)})")
    
    # Ask one question
    question = "Why is the sky blue?"
    result = agent.process_question(question)
    
    if hasattr(result, 'response'):
        print(f"\nQ: {question}")
        print(f"A: {result.response}")
        print(f"Spike detected: {result.has_spike if hasattr(result, 'has_spike') else 'N/A'}")
    
    print("\n✅ Test completed successfully!")
    
    # Summary
    stats = agent.get_stats()
    print(f"\nAgent Stats:")
    print(f"  Episodes: {stats['memory_stats']['total_episodes']}")
    print(f"  Cycles: {stats['total_cycles']}")

if __name__ == "__main__":
    main()