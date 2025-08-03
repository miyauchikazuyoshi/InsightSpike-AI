#!/usr/bin/env python3
"""
Test script to verify GED/IG values are properly captured in CSV output
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from run_experiment import QuestionAnswerExperiment

def test_ged_ig_capture():
    """Test that GED/IG values are properly captured."""
    print("Testing GED/IG value capture...")
    
    # Temporarily modify config to use mock provider
    import yaml
    with open('experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Save original provider
    original_provider = config['model']['provider']
    config['model']['provider'] = 'mock'
    
    # Write temporary config
    with open('test_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create test instance with mock config
    experiment = QuestionAnswerExperiment('test_config.yaml')
    
    # Initialize components
    datastore, adaptive_processor = experiment.initialize_components()
    memory_manager = adaptive_processor.exploration_loop.l2_memory
    
    # Load a few knowledge entries
    knowledge_base = experiment.load_knowledge_base()[:5]
    
    print(f"\nLoading {len(knowledge_base)} knowledge entries...")
    for entry in knowledge_base:
        result = experiment.add_knowledge_with_tracking(datastore, memory_manager, entry)
        print(f"  - Added: {entry['id']}")
    
    # Process one test question
    questions = experiment.load_questions()[:1]
    question = questions[0]
    
    print(f"\nProcessing question: {question['question'][:80]}...")
    result = experiment.process_question_with_branching(adaptive_processor, question)
    
    # Check for GED/IG values
    print("\nResults:")
    print(f"  - Question ID: {result['question_id']}")
    print(f"  - Has Insight: {result['has_insight']}")
    print(f"  - GED Value: {result.get('ged_value', 'NOT FOUND')}")
    print(f"  - IG Value: {result.get('ig_value', 'NOT FOUND')}")
    print(f"  - Processing Time: {result['processing_time']:.2f}s")
    
    # Check if values are present
    if result.get('ged_value') is not None and result.get('ig_value') is not None:
        print("\n✅ SUCCESS: GED and IG values are being captured!")
    else:
        print("\n❌ FAIL: GED or IG values are missing!")
        
    return result

if __name__ == '__main__':
    test_ged_ig_capture()