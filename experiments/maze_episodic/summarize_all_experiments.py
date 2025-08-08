#!/usr/bin/env python3
"""Summarize all experiment results"""

import os
import re

experiments = {
    # Pure implementations (no cheats)
    "true_gedig_flow_navigator.py": "Pure geDIG (user definition)",
    "true_pure_gedig_navigator.py": "True pure geDIG",
    "pure_episodic_navigator.py": "Pure episodic (original)",
    "pure_episodic_donut.py": "Pure episodic + donut search + visit count",
    
    # With optimizations/cheats
    "donut_gedig_navigator_simple.py": "Donut geDIG (with bonuses)",
    "optimized_episodic_navigator.py": "Optimized episodic",
    "efficient_topk_navigator.py": "Efficient TopK",
    "donut_gedig_navigator.py": "Donut geDIG (original)",
}

print("="*80)
print("EXPERIMENT SUMMARY")
print("="*80)

for filename, description in experiments.items():
    if not os.path.exists(filename):
        continue
        
    print(f"\n{description} ({filename}):")
    print("-"*60)
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Check for cheats
    cheats = []
    if 'bonus' in content.lower():
        cheats.append("exploration bonus")
    if 'penalty' in content.lower():
        cheats.append("visit penalty")
    if 'visited' in content and 'self.visited' in content:
        cheats.append("visited tracking")
    if 'visual_memory' in content:
        cheats.append("visual memory")
    if 'goal_dist.*<' in content:
        if re.search(r'goal_dist.*<', content):
            cheats.append("goal distance bonus")
    
    print(f"Cheats: {', '.join(cheats) if cheats else 'NONE'}")
    
    # Check for success claims
    success_match = re.search(r'(SUCCESS|Goal reached|✓.*[Gg]oal)', content)
    if success_match:
        print(f"Success claimed: YES")
        
        # Try to find maze sizes
        size_matches = re.findall(r'(\d+)x\1.*(?:success|Success|maze)', content)
        if size_matches:
            print(f"Maze sizes: {', '.join(set(size_matches))}")
    else:
        print(f"Success claimed: NO")

# Check for visualization files
print("\n\nVisualization files (evidence of success):")
print("-"*60)
for file in os.listdir('.'):
    if file.endswith('.png') and ('success' in file or 'gedig' in file):
        print(f"  {file}")