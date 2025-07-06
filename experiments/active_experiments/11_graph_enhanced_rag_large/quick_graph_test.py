#!/usr/bin/env python3
"""Quick test of graph functionality"""

import os
import sys
import torch
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import get_config
from datasets import Dataset

print("=== Quick Graph Functionality Test ===\n")

# Initialize
config = get_config()
agent = MainAgent(config)
agent.initialize()

# Load a few Q&A pairs
dataset_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets/squad_300")
dataset = Dataset.load_from_disk(str(dataset_path))

qa_pairs = []
for i in range(min(20, len(dataset))):  # Just 20 for quick test
    item = dataset[i]
    context = item.get('context', '')
    question = item.get('question', '')
    answers = item.get('answers', {})
    
    if isinstance(answers, dict) and 'text' in answers:
        answer = answers['text'][0] if answers['text'] else ""
    else:
        answer = str(answers)
    
    if context and question and answer:
        qa_pairs.append({
            'context': context,
            'question': question,
            'answer': answer
        })

print(f"Loaded {len(qa_pairs)} Q&A pairs\n")

# Add episodes and track graph growth
print("Adding episodes and tracking graph growth:")
print(f"{'Step':<6} {'Episodes':<10} {'Graph Nodes':<12} {'Graph Edges':<12}")
print("-" * 42)

for i, qa in enumerate(qa_pairs[:10]):  # Just 10 Q&A pairs
    # Add context
    result1 = agent.add_episode_with_graph_update(qa['context'], c_value=0.8)
    
    # Add Q&A
    qa_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
    result2 = agent.add_episode_with_graph_update(qa_text, c_value=0.6)
    
    # Check graph state
    graph_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/graph_pyg.pt")
    if graph_path.exists():
        data = torch.load(graph_path)
        nodes = data.x.shape[0] if data.x is not None else 0
        edges = data.edge_index.shape[1] if data.edge_index is not None else 0
    else:
        nodes = edges = 0
    
    episodes = result2.get('total_episodes', 0)
    print(f"{i+1:<6} {episodes:<10} {nodes:<12} {edges:<12}")

# Final analysis
print("\n=== Final Graph Analysis ===")
if graph_path.exists():
    data = torch.load(graph_path)
    nodes = data.x.shape[0] if data.x is not None else 0
    edges = data.edge_index.shape[1] if data.edge_index is not None else 0
    density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0
    
    print(f"Total nodes: {nodes}")
    print(f"Total edges: {edges}")
    print(f"Graph density: {density:.3f}")
    print(f"File size: {graph_path.stat().st_size} bytes")
    
    # Show sample connections
    if edges > 0:
        print(f"\nSample edges (first 5):")
        for i in range(min(5, edges)):
            src, dst = data.edge_index[:, i]
            print(f"  {src.item()} → {dst.item()}")
else:
    print("No graph file found!")

# Test retrieval
print("\n=== Testing Retrieval ===")
test_qa = qa_pairs[5]  # Middle one
results = agent.l2_memory.search_episodes(test_qa['question'], k=3)

print(f"Question: {test_qa['question']}")
print(f"Expected answer: {test_qa['answer']}")
print(f"\nTop 3 results:")
for i, result in enumerate(results):
    text_preview = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
    print(f"{i+1}. (score: {result['similarity']:.3f}) {text_preview}")
    if test_qa['answer'].lower() in result['text'].lower():
        print("   ✓ Contains answer!")

print("\n=== Test Complete ===")