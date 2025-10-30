#!/usr/bin/env python3
"""迅速な閾値テスト"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.graph_manager import GraphManager
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


def test_gedig_values():
    """実際のgeDIG値の範囲を確認"""
    
    print("NoCopy版GraphManagerのgeDIG値確認")
    print("=" * 60)
    
    evaluator = GeDIGEvaluator()
    mgr = GraphManager(evaluator)
    
    # テストエピソード
    episodes = []
    for i in range(10):
        ep = Episode(
            episode_id=i,
            position=(i % 5, i // 5),
            direction='N',
            vector=np.random.randn(128),
            is_wall=False,
            timestamp=float(i)
        )
        episodes.append(ep)
        mgr.add_episode_node(ep)
    
    # 異なる閾値でテスト
    thresholds = [-0.02, -0.03, -0.04, -0.045, -0.05, -0.06]
    
    print("閾値ごとのエッジ作成:")
    print("-" * 40)
    
    for threshold in thresholds:
        # リセット
        test_mgr = GraphManager(evaluator)
        for ep in episodes:
            test_mgr.add_episode_node(ep)
        
        # geDIG配線
        test_mgr._wire_with_gedig(episodes, threshold=threshold)
        
        edges = test_mgr.graph.number_of_edges()
        
        # geDIG値の統計
        if test_mgr.edge_logs:
            values = [log['gedig'] for log in test_mgr.edge_logs]
            min_val = min(values)
            max_val = max(values)
            avg_val = np.mean(values)
            print(f"  閾値 {threshold:>6.3f}: {edges:>2} edges, "
                  f"geDIG range [{min_val:.4f}, {max_val:.4f}], avg={avg_val:.4f}")
        else:
            print(f"  閾値 {threshold:>6.3f}: {edges:>2} edges (no edges created)")
    
    print("\n推奨:")
    print("  迷路ナビゲーション用: geDIG閾値 = -0.045")
    print("  バックトラック用: バックトラック閾値 = -0.2")


if __name__ == '__main__':
    test_gedig_values()