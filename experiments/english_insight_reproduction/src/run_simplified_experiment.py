#!/usr/bin/env python3
"""
Simplified English Insight Experiment
====================================
Testing with subset of data for faster execution
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Setup
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

# Import quietly
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    from src.insightspike.config import load_config
    from src.insightspike.implementations.agents.main_agent import MainAgent
finally:
    sys.stderr.close()
    sys.stderr = stderr_backup

print("=" * 60)
print("Simplified English Insight Experiment")
print("Using DistilGPT2 (Local Model)")
print("=" * 60)

# Initialize
print("\nInitializing...", flush=True)
config_path = Path(__file__).parent.parent / "config_experiment.yaml"
config = load_config(config_path=str(config_path))

print("Creating MainAgent...", flush=True)
start_time = time.time()

# MainAgent can be created without datastore (will use in-memory)
agent = MainAgent(config)
init_time = time.time() - start_time
print(f"✓ Agent initialized in {init_time:.2f}s", flush=True)

# Simplified knowledge base (10 items instead of 50)
knowledge_items = [
    # Phase 1: Basic concepts
    "Energy is the capacity to do work.",
    "Information is defined as the reduction of uncertainty.",
    
    # Phase 2: Relationships
    "Information and entropy have a deep mathematical relationship.",
    "The second law of thermodynamics and Shannon's information theory share the same mathematical structure.",
    
    # Phase 3: Integration
    "Energy, information, and entropy form the fundamental trinity of the universe.",
    "Life is a dissipative structure that locally decreases entropy.",
    
    # Phase 4: Questions
    "Can the hard problem of consciousness be solved from an information integration perspective?",
    "Is evolution a process for the universe to recognize itself?",
    
    # Phase 5: Insights
    "Energy, information, and consciousness are different aspects of the same reality.",
    "All physical laws reduce to laws of information conservation and transformation."
]

# Add knowledge
print(f"\nInjecting {len(knowledge_items)} knowledge items...", flush=True)
for i, item in enumerate(knowledge_items):
    result = agent.add_knowledge(item)
    print(f"  [{i+1:2d}/{len(knowledge_items)}] Added: {item[:50]}...", flush=True)

# Test questions (3 instead of 6)
test_questions = [
    {
        'id': 1,
        'question': "How are energy and information fundamentally related?",
        'expected_spike': True
    },
    {
        'id': 2,
        'question': "Can consciousness be understood through information theory?",
        'expected_spike': True
    },
    {
        'id': 3,
        'question': "How does life organize information against entropy?",
        'expected_spike': True
    }
]

# Process questions
print(f"\nTesting with {len(test_questions)} questions...", flush=True)
results = []

for q in test_questions:
    print(f"\nQuestion {q['id']}: {q['question']}", flush=True)
    
    start_time = time.time()
    result = agent.process_question(q['question'])
    processing_time = time.time() - start_time
    
    # Handle CycleResult object
    if hasattr(result, 'has_spike'):
        has_spike = result.has_spike
        spike_confidence = getattr(result.spike_info, 'confidence', 0) if hasattr(result, 'spike_info') else 0
        response = getattr(result, 'response', '')
    else:
        # Fallback for dict-like objects
        has_spike = result.get('has_spike', False) if hasattr(result, 'get') else False
        spike_confidence = result.get('spike_info', {}).get('confidence', 0) if hasattr(result, 'get') else 0
        response = result.get('response', '') if hasattr(result, 'get') else ''
    
    test_result = {
        'question_id': q['id'],
        'question': q['question'],
        'has_spike': has_spike,
        'spike_confidence': spike_confidence,
        'response': response,
        'processing_time': processing_time,
        'expected_spike': q['expected_spike'],
        'correct': has_spike == q['expected_spike']
    }
    results.append(test_result)
    
    # Display result
    if has_spike:
        print(f"  ✨ SPIKE DETECTED! (confidence: {spike_confidence:.3f})", flush=True)
    else:
        print(f"  📝 No spike detected", flush=True)
    
    print(f"  Response: {response[:100]}...", flush=True)
    print(f"  Time: {processing_time:.2f}s", flush=True)

# Summary
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

total_questions = len(results)
correct_detections = sum(1 for r in results if r['correct'])
spikes_detected = sum(1 for r in results if r['has_spike'])
avg_processing_time = sum(r['processing_time'] for r in results) / total_questions

print(f"Accuracy: {correct_detections}/{total_questions} ({correct_detections/total_questions:.1%})")
print(f"Spikes Detected: {spikes_detected}/{total_questions}")
print(f"Average Processing Time: {avg_processing_time:.2f}s")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path(__file__).parent.parent / "results" / "outputs"
results_dir.mkdir(parents=True, exist_ok=True)

results_file = results_dir / f"simplified_results_{timestamp}.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        'experiment': 'english_insight_simplified',
        'model': 'distilgpt2',
        'timestamp': timestamp,
        'summary': {
            'accuracy': correct_detections / total_questions,
            'spikes_detected': spikes_detected,
            'total_questions': total_questions,
            'avg_processing_time': avg_processing_time
        },
        'results': results
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {results_file}")
print("\n✅ Experiment completed!")