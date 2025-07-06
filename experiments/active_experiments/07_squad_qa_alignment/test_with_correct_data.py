#!/usr/bin/env python3
"""
Test Q&A with Correct Data Loading
===================================

Fix the technical issues and test again.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config


def test_with_correct_data():
    """Test with correct data loading."""
    
    print("=== Testing with Correct Data Path ===\n")
    
    # Setup - FIXED PATH
    config = get_config()
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "data"  # This is the correct path!
    
    # Update config to use experiment-specific data
    config.paths.data_dir = str(data_dir)
    config.memory.index_file = str(data_dir / "index.faiss")
    config.reasoning.graph_file = str(data_dir / "graph_pyg.pt")
    
    # Check if files exist
    print("Checking data files:")
    episodes_file = data_dir / "episodes.json"
    if episodes_file.exists():
        with open(episodes_file, 'r') as f:
            episodes = json.load(f)
        print(f"✓ Episodes file found: {len(episodes)} episodes")
        
        # Check first episode structure
        if episodes:
            first_ep = episodes[0]
            text = first_ep.get('text', '')
            if '|' in text:
                parts = text.split('|')
                print(f"  Episode 0 has {len(parts)} integrated parts")
            else:
                print(f"  Episode 0 length: {len(text)} chars")
    else:
        print("✗ Episodes file not found!")
        return
    
    print("\nLoading agent...")
    agent = MainAgent(config)
    
    # Use enhanced memory but disable aggressive integration
    agent.l2_memory = L2EnhancedScalableMemory(
        dim=config.embedding.dimension,
        config=config,
        use_scalable_graph=True
    )
    
    # Load state
    if not agent.load_state():
        print("Failed to load agent state!")
        return
    
    # Check what was actually loaded
    actual_episodes = len(agent.l2_memory.episodes)
    print(f"✓ Loaded {actual_episodes} episodes into memory")
    
    if actual_episodes != len(episodes):
        print(f"⚠️  Warning: Mismatch! File has {len(episodes)} but memory has {actual_episodes}")
    
    # Test specific questions
    test_questions = [
        {
            "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
            "answer": "Saint Bernadette Soubirous"
        },
        {
            "question": "What is in front of the Notre Dame Main Building?",
            "answer": "a copper statue of Christ"
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING Q&A\n")
    
    for qa in test_questions:
        print(f"Q: {qa['question']}")
        
        # First, check if answer exists in episodes
        answer_found_in_episodes = False
        for i, ep in enumerate(agent.l2_memory.episodes[:10]):  # Check first 10
            if hasattr(ep, 'text'):
                text = ep.text
            elif isinstance(ep, dict):
                text = ep.get('text', '')
            else:
                continue
                
            if qa['answer'].lower() in text.lower():
                answer_found_in_episodes = True
                print(f"✓ Answer found in episode {i}")
                break
        
        if not answer_found_in_episodes:
            print("✗ Answer not found in loaded episodes")
        
        # Try to get answer
        start_time = time.time()
        try:
            result = agent.process_question(qa['question'], max_cycles=1, verbose=False)
            response = result.get('response', '')
            response_time = time.time() - start_time
            
            print(f"\nResponse: {response[:200]}...")
            
            if qa['answer'].lower() in response.lower():
                print("✅ CORRECT - Answer found!")
            else:
                print("❌ INCORRECT - Answer not in response")
            
            print(f"Time: {response_time:.2f}s")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-"*80)
    
    # Analyze the integration issue
    print("\n=== Integration Analysis ===")
    if episodes:
        total_integrated = 0
        for ep in episodes[:50]:  # Check first 50
            text = ep.get('text', '')
            if '|' in text:
                parts = text.split('|')
                if len(parts) > 1:
                    total_integrated += 1
                    print(f"Episode {ep.get('id', '?')}: {len(parts)} parts integrated")
        
        print(f"\nTotal integrated episodes: {total_integrated}/50")
        
        # Show configuration
        print("\nCurrent integration thresholds:")
        print(f"  Content threshold: {config.memory.episode_integration_content_threshold}")
        print(f"  Similarity threshold: {config.memory.episode_integration_similarity_threshold}")
        
        print("\nRecommendation:")
        print("  - Increase content_threshold to 0.7+ to prevent over-integration")
        print("  - Increase similarity_threshold to 0.95+")
        print("  - Or disable integration for factual Q&A datasets")


if __name__ == "__main__":
    test_with_correct_data()