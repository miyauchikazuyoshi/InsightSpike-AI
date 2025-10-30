#!/usr/bin/env python3
"""Debug why edges are not created."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.graph_manager import GraphManager
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


def debug_gedig_values():
    """実際のgeDIG値を確認"""
    
    print("実際のgeDIG値を調査")
    print("=" * 60)
    
    evaluator = GeDIGEvaluator()
    mgr = GraphManager(evaluator)
    
    # シンプルなテスト
    episodes = []
    for i in range(5):
        ep = Episode(
            episode_id=i,
            position=(i, 0),
            direction='N',
            vector=np.random.randn(128),
            is_wall=False,
            timestamp=float(i)
        )
        episodes.append(ep)
        mgr.add_episode_node(ep)
    
    # 手動でgeDIG値を計算
    print("エピソード間のgeDIG値:")
    print("-" * 40)
    
    for i in range(1, len(episodes)):
        current = episodes[i]
        previous = episodes[i-1]
        
        # グラフのコピーを作成
        g_before = mgr.get_graph_snapshot()
        g_after = g_before.copy()
        g_after.add_edge(current.episode_id, previous.episode_id)
        
        # geDIG計算
        gedig_value = evaluator.calculate(g_before, g_after)
        
        print(f"  Episode {previous.episode_id} -> {current.episode_id}: {gedig_value:.4f}")
        
        # 実際にエッジを追加してグラフを更新
        if gedig_value <= -0.15:  # 閾値チェック
            mgr.graph.add_edge(current.episode_id, previous.episode_id)
            print(f"    → エッジ追加！")
    
    print()
    print(f"最終的なエッジ数: {mgr.graph.number_of_edges()}")
    
    # 異なる閾値でテスト
    print("\n閾値ごとのエッジ作成:")
    print("-" * 40)
    
    thresholds = [0.5, 0.0, -0.1, -0.2, -0.3, -0.5]
    
    for threshold in thresholds:
        test_mgr = GraphManager(evaluator)
        for ep in episodes:
            test_mgr.add_episode_node(ep)
        
        test_mgr._wire_with_gedig(episodes, threshold=threshold)
        
        edges = test_mgr.graph.number_of_edges()
        print(f"  閾値 {threshold:>5.1f}: {edges} エッジ")
    
    print("\n注意: geDIG値が閾値より大きい場合が多いため、エッジが作られない可能性がある")


if __name__ == '__main__':
    debug_gedig_values()