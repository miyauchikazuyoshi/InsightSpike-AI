#!/usr/bin/env python3
"""
Final Test - Experiment 9
=========================
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import get_config
from datasets import Dataset
from sentence_transformers import SentenceTransformer


# First, clear data and build fresh
print("=== EXPERIMENT 9: FINAL TEST ===")
print(f"Start: {datetime.now()}")

# Setup
config = get_config()
data_dir = Path(config.paths.data_dir)

# Clear data
print("\nClearing data folder...")
import shutil
if data_dir.exists():
    shutil.rmtree(data_dir)
data_dir.mkdir()

# Load datasets
print("\n=== Loading Datasets ===")
hf_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets")

qa_pairs = []
for dataset_name in ['squad_200', 'squad_300', 'ms_marco_150', 'coqa_80']:
    dataset_path = hf_path / dataset_name
    if dataset_path.exists():
        print(f"Loading {dataset_name}...")
        dataset = Dataset.load_from_disk(str(dataset_path))
        
        for i in range(min(200, len(dataset))):
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
            
            if len(qa_pairs) >= 500:
                break
    
    if len(qa_pairs) >= 500:
        break

print(f"Loaded {len(qa_pairs)} Q&A pairs")

# Build knowledge base
print("\n=== Building Knowledge Base ===")
agent = MainAgent(config)

start_build = time.time()
for i, qa in enumerate(qa_pairs):
    agent.add_episode_with_graph_update(qa['context'], c_value=0.8)
    agent.add_episode_with_graph_update(f"Question: {qa['question']}\nAnswer: {qa['answer']}", c_value=0.6)
    
    if (i + 1) % 100 == 0:
        print(f"Progress: {i+1}/{len(qa_pairs)} - Episodes: {len(agent.l2_memory.episodes)}")

build_time = time.time() - start_build
final_episodes = len(agent.l2_memory.episodes)

print(f"\nBuild complete!")
print(f"  Episodes: {final_episodes}")
print(f"  Integration rate: {(1 - final_episodes/(len(qa_pairs)*2))*100:.1f}%")
print(f"  Build time: {build_time:.1f}s")

# Save
print("\nSaving...")
agent.save_state()

# Save Q&A pairs
with open(data_dir / "qa_pairs.json", 'w') as f:
    json.dump(qa_pairs, f)

# Test performance
print("\n=== Testing Performance ===")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

test_size = 50
test_indices = random.sample(range(len(qa_pairs)), test_size)

# Standard RAG
print(f"\nStandard RAG (n={test_size}):")
rag_correct = 0
rag_times = []

for idx in test_indices:
    qa = qa_pairs[idx]
    start = time.time()
    
    query_vec = embedder.encode(qa['question'])
    D, I = agent.l2_memory.index.search(query_vec.reshape(1, -1).astype(np.float32), k=5)
    
    for ep_idx in I[0]:
        if 0 <= ep_idx < len(agent.l2_memory.episodes):
            ep = agent.l2_memory.episodes[ep_idx]
            ep_text = ep.text if hasattr(ep, 'text') else str(ep)
            
            if qa['answer'].lower() in ep_text.lower():
                rag_correct += 1
                break
    
    rag_times.append(time.time() - start)

rag_accuracy = rag_correct / test_size
rag_avg_time = np.mean(rag_times)
print(f"  Accuracy: {rag_accuracy:.1%} ({rag_correct}/{test_size})")
print(f"  Avg time: {rag_avg_time:.3f}s")

# InsightSpike
print(f"\nInsightSpike (n={test_size}):")
insight_correct = 0
insight_times = []

for idx in test_indices:
    qa = qa_pairs[idx]
    start = time.time()
    
    try:
        result = agent.process_question(qa['question'], max_cycles=1, verbose=False)
        if qa['answer'].lower() in result.get('response', '').lower():
            insight_correct += 1
    except:
        pass
    
    insight_times.append(time.time() - start)

insight_accuracy = insight_correct / test_size
insight_avg_time = np.mean(insight_times)
print(f"  Accuracy: {insight_accuracy:.1%} ({insight_correct}/{test_size})")
print(f"  Avg time: {insight_avg_time:.3f}s")

# Results
print("\n" + "="*60)
print("📊 最終結果")
print("="*60)
print(f"知識ベース: {final_episodes}エピソード（{len(qa_pairs)} Q&Aペアから）")
print(f"統合率: {(1 - final_episodes/(len(qa_pairs)*2))*100:.1f}%")
print("-"*60)
print(f"Standard RAG: {rag_accuracy:.1%} 精度, {rag_avg_time:.3f}秒/質問")
print(f"InsightSpike: {insight_accuracy:.1%} 精度, {insight_avg_time:.3f}秒/質問")
print("-"*60)
accuracy_diff = (insight_accuracy - rag_accuracy) * 100
speed_ratio = rag_avg_time / insight_avg_time if insight_avg_time > 0 else 0
print(f"改善: 精度 {accuracy_diff:+.1f}%, 速度 {speed_ratio:.2f}x")
print("="*60)

# Backup
print("\nBacking up data...")
backup_dir = Path(__file__).parent / "data_final"
if backup_dir.exists():
    shutil.rmtree(backup_dir)
shutil.copytree(data_dir, backup_dir)

# Restore original
print("Restoring original data...")
original_backup = Path(__file__).parent / "data_backup_before"
shutil.rmtree(data_dir)
data_dir.mkdir()
for item in original_backup.iterdir():
    if item.is_file():
        shutil.copy2(item, data_dir / item.name)
    else:
        shutil.copytree(item, data_dir / item.name)

print("\n✅ Experiment 9 Complete!")