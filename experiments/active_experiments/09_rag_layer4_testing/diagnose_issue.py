#!/usr/bin/env python3
"""
Diagnose InsightSpike Issue
===========================
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import get_config


def diagnose():
    """Diagnose why InsightSpike returns 0% accuracy."""
    print("=== Diagnosing InsightSpike Issue ===\n")
    
    # Setup
    config = get_config()
    data_dir = Path(config.paths.data_dir)
    
    # Load one Q&A pair
    qa_file = data_dir / "qa_pairs.json"
    if not qa_file.exists():
        print("No Q&A pairs found in data folder")
        return
    
    with open(qa_file, 'r') as f:
        qa_pairs = json.load(f)
    
    if not qa_pairs:
        print("Empty Q&A pairs")
        return
    
    # Initialize agent
    print("1. Initializing agent...")
    agent = MainAgent(config)
    
    # Load state
    if not agent.load_state():
        print("Failed to load state")
        return
    
    print(f"   Loaded {len(agent.l2_memory.episodes)} episodes")
    
    # Test with first Q&A
    qa = qa_pairs[0]
    print(f"\n2. Testing with question: {qa['question']}")
    print(f"   Expected answer: {qa['answer']}")
    
    # Test memory search
    print("\n3. Testing memory search...")
    memory_results = agent._search_memory(qa['question'])
    docs = memory_results.get('documents', [])
    print(f"   Found {len(docs)} documents")
    
    if docs:
        for i, doc in enumerate(docs[:3]):
            print(f"\n   Document {i+1}:")
            print(f"   - Text: {doc['text'][:100]}...")
            print(f"   - Similarity: {doc['similarity']:.3f}")
            print(f"   - Contains answer: {'YES' if qa['answer'].lower() in doc['text'].lower() else 'NO'}")
    
    # Test LLM provider
    print("\n4. Testing LLM provider...")
    print(f"   Provider type: {config.llm.provider}")
    print(f"   Safe mode: {getattr(config.llm, 'safe_mode', False)}")
    
    # Test full process_question
    print("\n5. Testing full process_question...")
    try:
        result = agent.process_question(qa['question'], max_cycles=1, verbose=True)
        
        print(f"\n   Success: {result.get('success', False)}")
        print(f"   Response length: {len(result.get('response', ''))}")
        print(f"   Response preview: {result.get('response', '')[:200]}...")
        
        if result.get('cycle_history'):
            cycle = result['cycle_history'][0]
            print(f"\n   Cycle details:")
            print(f"   - Retrieved docs: {len(cycle.get('retrieved_documents', []))}")
            print(f"   - Reasoning quality: {cycle.get('reasoning_quality', 0):.3f}")
            print(f"   - Spike detected: {cycle.get('spike_detected', False)}")
        
        # Check if answer is in response
        response = result.get('response', '')
        if qa['answer'].lower() in response.lower():
            print(f"\n   ✅ Answer found in response!")
        else:
            print(f"\n   ❌ Answer NOT found in response")
            
            # Check LLM context
            if result.get('cycle_history'):
                retrieved_docs = cycle.get('retrieved_documents', [])
                print(f"\n   LLM Context Analysis:")
                print(f"   - Documents passed to LLM: {len(retrieved_docs)}")
                for i, doc in enumerate(retrieved_docs[:2]):
                    text = doc.get('text', '')
                    print(f"   - Doc {i+1} contains answer: {'YES' if qa['answer'].lower() in text.lower() else 'NO'}")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Check config
    print("\n6. Checking configuration...")
    print(f"   Environment: {getattr(config, 'environment', 'unknown')}")
    print(f"   LLM max tokens: {config.llm.max_tokens}")
    print(f"   LLM temperature: {config.llm.temperature}")
    
    # Check if using clean/safe mode
    if hasattr(config.llm, 'safe_mode') and config.llm.safe_mode:
        print("\n⚠️  WARNING: Safe mode is enabled!")
        print("   This uses CleanLLMProvider which returns generic responses")
        print("   It will NOT use the retrieved context documents")


if __name__ == "__main__":
    diagnose()