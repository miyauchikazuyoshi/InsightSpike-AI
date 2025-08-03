#!/usr/bin/env python3
"""
Test 2-hop GED/IG in experiment setup
"""

import json
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from run_experiment import QuestionAnswerExperiment

def test_2hop_experiment():
    """Test 2-hop in experiment configuration."""
    
    # Load config
    with open('experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Current graph configuration:")
    print(f"  Message passing max_hops: {config['insightspike']['graph']['message_passing'].get('max_hops', 'NOT SET')}")
    print(f"  Metrics use_multihop_gedig: {config['insightspike']['graph'].get('metrics', {}).get('use_multihop_gedig', 'NOT SET')}")
    print(f"  Metrics max_hops: {config['insightspike']['graph'].get('metrics', {}).get('max_hops', 'NOT SET')}")
    print(f"  Metrics decay_factor: {config['insightspike']['graph'].get('metrics', {}).get('decay_factor', 'NOT SET')}")
    
    # Create experiment instance
    experiment = QuestionAnswerExperiment('experiment_config.yaml')
    
    # Initialize components
    datastore, adaptive_processor = experiment.initialize_components()
    
    # Check L3 config
    l3_config = adaptive_processor.exploration_loop.l3_graph.config
    print("\nL3GraphReasoner config:")
    print(f"  Config type: {type(l3_config)}")
    if hasattr(l3_config, 'graph'):
        graph_cfg = l3_config.graph
        print(f"  Graph config: {graph_cfg}")
    else:
        print(f"  Direct config: {l3_config}")
    
    # Test with one knowledge entry and one question
    memory_manager = adaptive_processor.exploration_loop.l2_memory
    
    # Load sample data
    knowledge_base = experiment.load_knowledge_base()[:3]
    questions = experiment.load_questions()[:1]
    
    print("\nAdding knowledge...")
    for entry in knowledge_base:
        result = experiment.add_knowledge_with_tracking(datastore, memory_manager, entry)
        print(f"  - Added: {entry['id']}")
    
    print("\nProcessing question...")
    question = questions[0]
    result = experiment.process_question_with_branching(adaptive_processor, question)
    
    print(f"\nResults:")
    print(f"  - Question: {question['question'][:60]}...")
    print(f"  - GED: {result.get('ged_value', 'NOT FOUND')}")
    print(f"  - IG: {result.get('ig_value', 'NOT FOUND')}")
    print(f"  - Has insight: {result['has_insight']}")
    
    # Check if multihop info is available
    if 'metadata' in result and 'hop_results' in result['metadata']:
        print("\nHop results found:")
        for hop, data in result['metadata']['hop_results'].items():
            print(f"  Hop {hop}: {data}")

if __name__ == '__main__':
    test_2hop_experiment()