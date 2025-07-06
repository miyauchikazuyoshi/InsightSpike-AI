#!/usr/bin/env python3
"""Verify Setup"""

import json
from pathlib import Path

# Check files
data_dir = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiment_8/data")

print("=== Verifying Experiment 8 Setup ===")

print("\n1. Data Files:")
files = {
    "episodes.json": "Episodes",
    "index.faiss": "FAISS index", 
    "graph_pyg.pt": "Graph",
    "qa_pairs.json": "Q&A pairs"
}

all_exist = True
for fname, desc in files.items():
    fpath = data_dir / fname
    if fpath.exists():
        size_mb = fpath.stat().st_size / 1024 / 1024
        print(f"  ✓ {desc}: {size_mb:.2f} MB")
    else:
        print(f"  ✗ {desc}: NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing!")
    exit(1)

print("\n2. Episode Analysis:")
with open(data_dir / "episodes.json", 'r') as f:
    episodes = json.load(f)

print(f"  Total episodes: {len(episodes)}")

# Sample first few episodes
print("\n  First 3 episodes:")
for i in range(min(3, len(episodes))):
    ep = episodes[i]
    text = ep.get('text', '')
    print(f"    Episode {i}: {len(text)} chars")
    if text:
        print(f"      Preview: {text[:100]}...")

print("\n3. Q&A Pairs Analysis:")
with open(data_dir / "qa_pairs.json", 'r') as f:
    qa_pairs = json.load(f)

print(f"  Total Q&A pairs: {len(qa_pairs)}")

# Sample questions
print("\n  Sample questions:")
for i in range(min(3, len(qa_pairs))):
    qa = qa_pairs[i]
    print(f"    Q: {qa['question'][:80]}...")
    print(f"    A: {qa['answer']}")

print("\n✅ Verification complete!")
print("\nExperiment 8 Summary:")
print(f"- Knowledge base: {len(episodes)} episodes from {len(qa_pairs)} Q&A pairs")
print(f"- Integration rate: {(1 - len(episodes) / (len(qa_pairs) * 2)) * 100:.1f}%")
print("\nReady for performance testing!")