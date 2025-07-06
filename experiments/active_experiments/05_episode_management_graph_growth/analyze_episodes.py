#!/usr/bin/env python3
"""
Analyze current episode data for integration patterns
"""

import json
import numpy as np
from collections import Counter

# Load episodes
with open("data/episodes.json", 'r') as f:
    episodes = json.load(f)

print(f"=== Episode Analysis ===")
print(f"Total episodes: {len(episodes)}\n")

# Analyze text similarity
print("1. Text Pattern Analysis:")
text_lengths = [len(ep.get('text', '')) for ep in episodes]
print(f"   Average text length: {np.mean(text_lengths):.0f} chars")
print(f"   Min/Max: {min(text_lengths)}/{max(text_lengths)} chars")

# Check for similar content
words_counter = Counter()
for ep in episodes:
    text = ep.get('text', '').lower()
    words = text.split()
    for word in words:
        if len(word) > 4:  # Skip short words
            words_counter[word] += 1

print("\n2. Most Common Words (potential integration candidates):")
for word, count in words_counter.most_common(10):
    if count > 1:
        print(f"   '{word}': appears {count} times")

# Check C-values
c_values = [ep.get('c', 0.5) for ep in episodes]
print(f"\n3. C-value Distribution:")
print(f"   Average: {np.mean(c_values):.3f}")
print(f"   Std Dev: {np.std(c_values):.3f}")
print(f"   Min/Max: {min(c_values):.3f}/{max(c_values):.3f}")

# Estimate integration rate
# If episodes were added without integration, we'd have more episodes
print(f"\n4. Integration Rate Estimation:")
print(f"   Current episodes: {len(episodes)}")
print(f"   If no integration occurred, might have had more episodes")

# Look for patterns in episode indices
if episodes and 'episode_idx' in episodes[0]:
    indices = [ep.get('episode_idx', i) for i, ep in enumerate(episodes)]
    gaps = []
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            gaps.append(i)
    
    if gaps:
        print(f"   Found {len(gaps)} potential integration points")
    else:
        print("   Episodes appear to be sequential (low integration)")
else:
    print("   No episode_idx field to analyze")

# Configuration check
print("\n5. Integration Configuration:")
print("   Default thresholds:")
print("   - Similarity: 0.85")
print("   - Content overlap: 0.70")
print("   - C-value diff: 0.30")
print("\n   These are quite high, explaining low integration rate")