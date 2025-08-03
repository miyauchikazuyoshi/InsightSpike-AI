#!/usr/bin/env python3
"""
Test using MainAgent.process_question instead of adaptive_loop
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.processing.embedder import EmbeddingManager

def test_mainagent_approach():
    """Test spike detection using MainAgent instead of adaptive_loop."""
    print("Testing MainAgent approach (without adaptive_loop)...")
    
    # Create MainAgent with test config
    config = {
        "embedder": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu"
        },
        "llm": {
            "provider": "mock",
            "model": "mock-model",
            "prompt_style": "association_extended"
        },
        "vector_storage": {
            "index_type": "flat",
            "dimension": 384
        },
        "graph": {
            "edge_formation": {
                "strategy": "radius",
                "radius": 0.3
            },
            "algorithms": {
                "metrics": {
                    "ged": "pyg",
                    "ig": "pyg"
                }
            },
            "spike_detection": {
                "ged_threshold": -0.5,
                "ig_threshold": 0.2
            }
        }
    }
    
    # Initialize agent
    agent = MainAgent(config=config)
    
    # Load a few knowledge entries
    with open('data/input/knowledge_base/knowledge_500.json', 'r') as f:
        data = json.load(f)
    
    knowledge_entries = data['knowledge_entries'][:5]
    
    print(f"\nAdding {len(knowledge_entries)} knowledge entries...")
    for entry in knowledge_entries:
        agent.add_knowledge(entry['content'])
        print(f"  - Added: {entry['id']}")
    
    # Process a test question
    with open('data/input/questions/questions_100.json', 'r') as f:
        questions_data = json.load(f)
    
    question = questions_data['questions'][0]
    
    print(f"\nProcessing question: {question['question'][:80]}...")
    result = agent.process_question(question['question'])
    
    # Extract results
    print("\nResults:")
    print(f"  - Question ID: {question['id']}")
    
    # Debug: print result structure
    print(f"\nResult type: {type(result)}")
    if hasattr(result, '__dict__'):
        print(f"Result attributes: {list(result.__dict__.keys())}")
    
    print(f"  - Has Spike: {result.has_spike if hasattr(result, 'has_spike') else 'NOT FOUND'}")
    
    # Check graph_analysis for GED/IG values
    if hasattr(result, 'graph_analysis') and result.graph_analysis:
        print(f"\nGraph analysis: {result.graph_analysis}")
        if 'metrics' in result.graph_analysis:
            metrics = result.graph_analysis['metrics']
            print(f"  - GED Value: {metrics.get('delta_ged', 'NOT FOUND')}")
            print(f"  - IG Value: {metrics.get('delta_ig', 'NOT FOUND')}")
        else:
            print("  - No metrics in graph_analysis")
    else:
        print("  - GED/IG values not available in result")
    
    print(f"  - Response: {result.response[:100] if hasattr(result, 'response') else result.get('response', '')[:100]}...")
    
    return result

if __name__ == '__main__':
    test_mainagent_approach()