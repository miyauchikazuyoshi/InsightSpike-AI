#!/usr/bin/env python3
"""Test refined geDIG implementation."""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.graph_manager import GraphManager
from core.graph_manager_optimized import OptimizedGraphManager
from core.graph_manager_refined import RefinedGraphManager
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


def compare_all_implementations():
    """全実装の比較"""
    
    print("=" * 70)
    print("geDIG実装の比較: Original vs Optimized vs Refined")
    print("=" * 70)
    
    # テスト用エピソード作成
    episodes = []
    for i in range(30):
        ep = Episode(
            episode_id=i,
            position=(i % 10, i // 10),
            direction='N',
            vector=np.random.randn(128),  # ベクトルも含める
            is_wall=False,
            timestamp=float(i)
        )
        episodes.append(ep)
    
    print(f"テストデータ: {len(episodes)}エピソード")
    print()
    
    evaluator = GeDIGEvaluator()
    
    # 1. オリジナル実装（本物のgeDIG、遅い）
    print("1. Original GraphManager (本物のgeDIG計算)")
    print("-" * 50)
    original_mgr = GraphManager(evaluator)
    for ep in episodes:
        original_mgr.add_episode_node(ep)
    
    start = time.perf_counter()
    original_mgr._wire_with_gedig(episodes, threshold=-0.15)
    original_time = time.perf_counter() - start
    
    print(f"  時間: {original_time*1000:.2f} ms")
    print(f"  エッジ数: {original_mgr.graph.number_of_edges()}")
    
    # エッジログから実際のgeDIG値を取得
    if original_mgr.edge_creation_log:
        gedig_values = []
        for log in original_mgr.edge_creation_log:
            if 'gedig' in log['strategy']:
                # "gedig (value=-0.123)" から値を抽出
                import re
                match = re.search(r'value=([-\d.]+)', log['strategy'])
                if match:
                    gedig_values.append(float(match.group(1)))
        
        if gedig_values:
            print(f"  geDIG値: min={min(gedig_values):.3f}, max={max(gedig_values):.3f}, mean={np.mean(gedig_values):.3f}")
    print()
    
    # 2. 最適化実装（簡単なヒューリスティック、速い）
    print("2. Optimized GraphManager (簡略化ヒューリスティック)")
    print("-" * 50)
    optimized_mgr = OptimizedGraphManager(evaluator)
    for ep in episodes:
        optimized_mgr.add_episode_node(ep)
    
    start = time.perf_counter()
    optimized_mgr._wire_with_gedig_optimized(episodes, threshold=-0.15)
    optimized_time = time.perf_counter() - start
    
    print(f"  時間: {optimized_time*1000:.2f} ms")
    print(f"  エッジ数: {optimized_mgr.graph.number_of_edges()}")
    
    if optimized_mgr.edge_logs:
        gedig_values = [log['gedig'] for log in optimized_mgr.edge_logs]
        print(f"  geDIG値: min={min(gedig_values):.3f}, max={max(gedig_values):.3f}, mean={np.mean(gedig_values):.3f}")
    print()
    
    # 3. 洗練実装（ハイブリッド、バランス）
    print("3. Refined GraphManager (ハイブリッド・洗練版)")
    print("-" * 50)
    refined_mgr = RefinedGraphManager(evaluator)
    for ep in episodes:
        refined_mgr.add_episode_node(ep)
    
    start = time.perf_counter()
    refined_mgr._wire_with_refined_gedig(episodes, threshold=-0.15, use_real_gedig_sampling=True)
    refined_time = time.perf_counter() - start
    
    print(f"  時間: {refined_time*1000:.2f} ms")
    print(f"  エッジ数: {refined_mgr.graph.number_of_edges()}")
    
    if refined_mgr.edge_logs:
        gedig_values = [log['gedig'] for log in refined_mgr.edge_logs]
        print(f"  geDIG値: min={min(gedig_values):.3f}, max={max(gedig_values):.3f}, mean={np.mean(gedig_values):.3f}")
    
    print()
    print("=" * 70)
    print("性能比較")
    print("=" * 70)
    print(f"Optimized vs Original: {original_time/optimized_time:.1f}x 高速")
    print(f"Refined vs Original: {original_time/refined_time:.1f}x 高速")
    print(f"Refined vs Optimized: {optimized_time/refined_time:.1f}x")
    print()
    print("結論:")
    print("- Optimized: 最速だが精度が低い（単純なヒューリスティック）")
    print("- Refined: バランスが良い（本物のgeDIGサンプリング + 洗練された近似）")
    print("- Original: 最も正確だが遅い（全て本物のgeDIG計算）")


if __name__ == '__main__':
    compare_all_implementations()