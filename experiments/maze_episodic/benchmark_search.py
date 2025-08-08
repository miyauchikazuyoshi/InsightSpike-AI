#!/usr/bin/env python3
"""Benchmark different search methods"""

import numpy as np
import time
from typing import List, Tuple

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Euclidean distance between two vectors"""
    return np.linalg.norm(vec1 - vec2)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors"""
    return np.dot(vec1, vec2)

# Create test data
n_episodes = 10000
print(f"Benchmarking with {n_episodes} episodes...")

# Position data for Manhattan distance
positions = [(np.random.randint(0, 50), np.random.randint(0, 50)) for _ in range(n_episodes)]
query_pos = (25, 25)

# Vector data for Euclidean/Cosine
vectors = [np.random.randn(7) for _ in range(n_episodes)]
# Normalize for cosine similarity
normalized_vectors = [v / np.linalg.norm(v) for v in vectors]
query_vec = np.random.randn(7)
query_vec_norm = query_vec / np.linalg.norm(query_vec)

print("\n1. Manhattan Distance Search (position-based)")
start = time.time()
manhattan_results = []
for i, pos in enumerate(positions):
    dist = manhattan_distance(query_pos, pos)
    if dist < 10:  # Threshold
        manhattan_results.append((i, dist))
manhattan_time = time.time() - start
print(f"   Time: {manhattan_time:.4f}s")
print(f"   Found: {len(manhattan_results)} results")

print("\n2. Euclidean Distance Search (vector-based)")
start = time.time()
euclidean_results = []
for i, vec in enumerate(vectors):
    dist = euclidean_distance(query_vec, vec)
    if dist < 2.0:  # Threshold
        euclidean_results.append((i, dist))
euclidean_time = time.time() - start
print(f"   Time: {euclidean_time:.4f}s")
print(f"   Found: {len(euclidean_results)} results")

print("\n3. Cosine Similarity Search (normalized vectors)")
start = time.time()
cosine_results = []
for i, vec in enumerate(normalized_vectors):
    sim = cosine_similarity(query_vec_norm, vec)
    if sim > 0.7:  # Threshold
        cosine_results.append((i, sim))
cosine_time = time.time() - start
print(f"   Time: {cosine_time:.4f}s")
print(f"   Found: {len(cosine_results)} results")

print("\n4. Combined Search (Manhattan + Cosine)")
start = time.time()
combined_results = []
for i in range(n_episodes):
    # First filter by Manhattan distance
    dist = manhattan_distance(query_pos, positions[i])
    if dist < 10:
        # Then check cosine similarity
        sim = cosine_similarity(query_vec_norm, normalized_vectors[i])
        if sim > 0.7:
            combined_results.append((i, dist, sim))
combined_time = time.time() - start
print(f"   Time: {combined_time:.4f}s")
print(f"   Found: {len(combined_results)} results")

print("\n5. Numpy Vectorized Euclidean")
start = time.time()
# Stack all vectors
vector_array = np.array(vectors)
# Compute all distances at once
distances = np.linalg.norm(vector_array - query_vec, axis=1)
# Find matches
numpy_results = np.where(distances < 2.0)[0]
numpy_time = time.time() - start
print(f"   Time: {numpy_time:.4f}s")
print(f"   Found: {len(numpy_results)} results")

print("\nSpeed comparison:")
print(f"Manhattan: 1.00x (baseline)")
print(f"Euclidean: {euclidean_time/manhattan_time:.2f}x")
print(f"Cosine: {cosine_time/manhattan_time:.2f}x")
print(f"Combined: {combined_time/manhattan_time:.2f}x")
print(f"Numpy vectorized: {numpy_time/manhattan_time:.2f}x")

# Memory usage estimate
print(f"\nMemory usage estimate:")
print(f"Positions: {len(positions) * 2 * 8 / 1024:.1f} KB")
print(f"Vectors (7D): {len(vectors) * 7 * 8 / 1024:.1f} KB")
print(f"Total for 10k episodes: {(len(positions) * 2 * 8 + len(vectors) * 7 * 8) / 1024:.1f} KB")