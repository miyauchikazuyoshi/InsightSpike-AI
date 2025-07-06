#!/usr/bin/env python3
"""Quick Performance Test"""

import json
import time
import random
from pathlib import Path
import numpy as np

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config
from sentence_transformers import SentenceTransformer

# Setup
config = get_config()
data_dir = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiment_8/data")
config.paths.data_dir = str(data_dir)
config.memory.index_file = str(data_dir / "index.faiss")
config.reasoning.graph_file = str(data_dir / "graph_pyg.pt")

# Load Q&A pairs
with open(data_dir / "qa_pairs.json", 'r') as f:
    qa_pairs = json.load(f)

print(f"Testing with {len(qa_pairs)} Q&A pairs")

# Initialize
agent = MainAgent(config)
agent.l2_memory = L2EnhancedScalableMemory(
    dim=config.embedding.dimension,
    config=config,
    use_scalable_graph=True
)
agent.load_state()

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Test 30 random questions
test_size = 30
test_indices = random.sample(range(len(qa_pairs)), test_size)

# Standard RAG
print("\n=== Standard RAG ===")
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
            if qa['answer'].lower() in (ep.text if hasattr(ep, 'text') else "").lower():
                rag_correct += 1
                break
    
    rag_times.append(time.time() - start)

# InsightSpike
print("\n=== InsightSpike ===")
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

# Results
print(f"\n📊 結果:")
print(f"Standard RAG: {rag_correct}/{test_size} ({rag_correct/test_size*100:.1f}%) - 平均{np.mean(rag_times):.3f}秒")
print(f"InsightSpike: {insight_correct}/{test_size} ({insight_correct/test_size*100:.1f}%) - 平均{np.mean(insight_times):.3f}秒")
print(f"\n改善: 精度{(insight_correct-rag_correct)/test_size*100:+.1f}%, 速度{np.mean(rag_times)/np.mean(insight_times):.2f}x")