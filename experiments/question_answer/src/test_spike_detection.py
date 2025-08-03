#!/usr/bin/env python3
"""
Small-scale test for spike detection (GED and IG values)
Tests with minimal knowledge base to verify spike detection works
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.config import load_config
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.implementations.layers.layer1_error_monitor import ErrorMonitor
from insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager  
from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface
from insightspike.implementations.layers.cached_memory_manager import CachedMemoryManager
from insightspike.adaptive.core.exploration_loop import ExplorationLoop
from insightspike.adaptive.strategies.expanding import ExpandingStrategy
from insightspike.adaptive.core.adaptive_processor import AdaptiveProcessor
from insightspike.adaptive.calculators.adaptive_topk import AdaptiveTopKCalculator
from insightspike.processing.embedder import EmbeddingManager


def create_test_knowledge():
    """Create a small knowledge base designed to trigger spikes"""
    knowledge = [
        # Basic facts
        {
            "id": "test_001",
            "content": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
            "tags": ["physics", "chemistry"],
            "difficulty": "basic"
        },
        {
            "id": "test_002", 
            "content": "Ice expands when it forms, making it less dense than liquid water.",
            "tags": ["physics", "chemistry"],
            "difficulty": "basic"
        },
        {
            "id": "test_003",
            "content": "Most substances contract when they solidify, becoming denser.",
            "tags": ["physics", "chemistry"],
            "difficulty": "basic"
        },
        # Connected facts that should create spike when queried together
        {
            "id": "test_004",
            "content": "Ice floats on water because it is less dense than liquid water.",
            "tags": ["physics", "chemistry"],
            "difficulty": "intermediate"
        },
        {
            "id": "test_005",
            "content": "The unique property of water expanding when frozen is crucial for aquatic life survival in winter.",
            "tags": ["physics", "biology"],
            "difficulty": "intermediate"
        }
    ]
    
    # Test questions designed to trigger different behaviors
    questions = [
        {
            "id": "q1",
            "question": "Why does ice float on water?",
            "expected_spike": True,
            "reason": "Should connect density, expansion, and floating concepts"
        },
        {
            "id": "q2", 
            "question": "What happens to water at 0 degrees?",
            "expected_spike": False,
            "reason": "Direct fact retrieval, no complex connection needed"
        },
        {
            "id": "q3",
            "question": "How does water's freezing behavior help fish survive winter?",
            "expected_spike": True,
            "reason": "Should connect multiple concepts across physics and biology"
        }
    ]
    
    return knowledge, questions


def test_spike_detection():
    """Run small-scale spike detection test"""
    print("=== Small-scale Spike Detection Test ===\n")
    
    # Create test data
    knowledge_base, test_questions = create_test_knowledge()
    
    # Initialize components
    print("1. Initializing components...")
    
    # Clean up any existing test data
    import shutil
    test_data_path = Path("./test_data")
    if test_data_path.exists():
        shutil.rmtree(test_data_path)
    
    # Initialize DataStore
    datastore = DataStoreFactory.create("filesystem", base_path="./test_data")
    
    # Initialize layers
    l1_error = ErrorMonitor()
    embedder = EmbeddingManager()
    l2_memory = CachedMemoryManager(
        datastore=datastore,
        cache_size=100,
        embedder=embedder
    )
    
    # Initialize L3 with spike detection thresholds
    graph_config = {
        'enable_message_passing': True,
        'message_passing': {
            'alpha': 0.3,
            'iterations': 2
        },
        'spike_detection': {
            'ged_threshold': 0,      # Changed: Any negative value (simplification)
            'ig_threshold': 0        # Changed: Any positive value (info gain)
        }
    }
    l3_graph = L3GraphReasoner(config={'graph': graph_config})
    l3_graph.initialize()
    
    # Initialize L4 (MockProvider for testing)
    llm_config = {
        'provider': 'mock',
        'model': 'mock',
        'prompt_style': 'association_extended'
    }
    l4_llm = L4LLMInterface(config={'llm': llm_config})
    l4_llm.initialize()
    
    # Create exploration loop
    exploration_loop = ExplorationLoop(
        l1_monitor=l1_error,
        l2_memory=l2_memory,
        l3_graph=l3_graph
    )
    
    # Create adaptive processor
    strategy = ExpandingStrategy()
    topk_calculator = AdaptiveTopKCalculator()
    
    adaptive_processor = AdaptiveProcessor(
        exploration_loop=exploration_loop,
        strategy=strategy,
        topk_calculator=topk_calculator,
        l4_llm=l4_llm,
        datastore=datastore,
        max_attempts=3
    )
    
    # Phase 1: Load knowledge
    print("\n2. Loading knowledge base...")
    for i, entry in enumerate(knowledge_base):
        episode = {
            'id': f"episode_{entry['id']}",
            'text': entry['content'],
            'vec': embedder.get_embedding(entry['content']),
            'c_value': 0.5,
            'metadata': {
                'tags': entry['tags'],
                'difficulty': entry['difficulty']
            }
        }
        
        # Add to memory
        l2_memory.add_episode(
            text=episode['text'],
            c_value=episode['c_value'],
            metadata=episode['metadata']
        )
        print(f"   Added: {entry['id']} - {entry['content'][:50]}...")
    
    # Episodes are automatically saved through datastore
    
    # Phase 2: Test questions
    print("\n3. Testing questions for spike detection...")
    print("-" * 80)
    
    results = []
    for question in test_questions:
        print(f"\nQuestion {question['id']}: {question['question']}")
        print(f"Expected spike: {question['expected_spike']} ({question['reason']})")
        
        # Process question
        start_time = time.time()
        result = adaptive_processor.process(question['question'], verbose=True)
        processing_time = time.time() - start_time
        
        # Extract metrics
        has_spike = result.get('spike_detected', False)
        metadata = result.get('adaptive_metadata', {})
        final_metrics = metadata.get('final_metrics', {})
        
        # Get GED and IG values
        delta_ged = final_metrics.get('l3_delta_ged', 0)
        delta_ig = final_metrics.get('l3_delta_ig', 0)
        
        print(f"\nResults:")
        print(f"  - Spike detected: {has_spike}")
        print(f"  - ΔGED: {delta_ged:.4f} (threshold: < 0)")
        print(f"  - ΔIG: {delta_ig:.4f} (threshold: > 0)")
        print(f"  - Processing time: {processing_time:.2f}s")
        print(f"  - Attempts: {metadata.get('total_attempts', 0)}")
        
        # Check if result matches expectation
        matches_expectation = has_spike == question['expected_spike']
        print(f"  - Matches expectation: {'✓' if matches_expectation else '✗'}")
        
        results.append({
            'question_id': question['id'],
            'question': question['question'],
            'expected_spike': question['expected_spike'],
            'actual_spike': has_spike,
            'delta_ged': delta_ged,
            'delta_ig': delta_ig,
            'matches': matches_expectation,
            'processing_time': processing_time
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    correct = sum(1 for r in results if r['matches'])
    total = len(results)
    
    print(f"\nCorrect predictions: {correct}/{total} ({correct/total*100:.0f}%)")
    
    print("\nDetailed results:")
    for r in results:
        status = "✓" if r['matches'] else "✗"
        print(f"{status} {r['question_id']}: Expected={r['expected_spike']}, "
              f"Actual={r['actual_spike']}, ΔGED={r['delta_ged']:.3f}, ΔIG={r['delta_ig']:.3f}")
    
    # Save results
    with open("test_spike_results.json", "w") as f:
        json.dump({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                'correct': correct,
                'total': total,
                'accuracy': correct/total
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to test_spike_results.json")
    
    # Cleanup
    if test_data_path.exists():
        shutil.rmtree(test_data_path)
    
    return results


if __name__ == "__main__":
    test_spike_detection()