#!/usr/bin/env python3
"""Analyze geDIG computation efficiency"""

from true_pure_gedig_navigator import TruePureGeDIGNavigator
from pure_episodic_navigator import create_complex_maze
import time

# Small maze
maze = create_complex_maze(5, seed=42)
nav = TruePureGeDIGNavigator(maze)

print("geDIG計算の効率性分析")
print("="*40)

# Add some episodes first
positions = [(1,1), (2,1), (3,1), (3,2)]
for i, pos in enumerate(positions[:-1]):
    next_pos = positions[i+1]
    action = 'right' if next_pos[0] > pos[0] else 'down'
    nav.add_episode(pos, action, 'success', False)

print(f"エピソード数: {len(nav.episodes)}")
print(f"グラフノード数: {nav.episode_graph.number_of_nodes()}")
print(f"グラフエッジ数: {nav.episode_graph.number_of_edges()}")

# Time single geDIG calculation
print("\n単一のgeDIG計算時間:")
start = time.time()
gedig = nav.evaluate_action_gedig((2, 1), 'right')
elapsed = time.time() - start
print(f"時間: {elapsed*1000:.2f}ms")
print(f"geDIG値: {gedig:.3f}")

# Test with more episodes
print("\nエピソード数を増やしてテスト:")
for n in [10, 20, 30]:
    # Add dummy episodes
    for i in range(len(nav.episodes), n):
        x = (i % 5) + 1
        y = (i // 5) + 1
        nav.add_episode((x, y), 'right', 'success', False)
    
    start = time.time()
    gedig = nav.evaluate_action_gedig((2, 1), 'right')
    elapsed = time.time() - start
    
    print(f"エピソード数 {len(nav.episodes)}: {elapsed*1000:.2f}ms")

print("\n問題:")
print("- グラフのコピーとサブグラフ抽出が重い")
print("- 各行動でグラフ全体をコピーしている")
print("- NetworkXの操作が遅い")

print("\n解決策:")
print("- ローカルな変化のみを計算")
print("- グラフのコピーを避ける")
print("- 簡略化されたGED/IG計算")