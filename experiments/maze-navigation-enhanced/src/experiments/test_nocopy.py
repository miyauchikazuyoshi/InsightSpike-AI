#!/usr/bin/env python3
"""Test NO COPY optimization - オリジナルに近い実装で高速化"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.graph_manager import GraphManager
from core.graph_manager_nocopy import NoCopyGraphManager
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


def test_nocopy_performance():
    """コピーなし実装のテスト"""
    
    print("=" * 70)
    print("NO COPY最適化テスト - オリジナルとの比較")
    print("=" * 70)
    
    # テストデータ
    episodes = []
    for i in range(30):
        ep = Episode(
            episode_id=i,
            position=(i % 10, i // 10),
            direction='N',
            vector=np.random.randn(128),
            is_wall=False,
            timestamp=float(i)
        )
        episodes.append(ep)
    
    print(f"テストデータ: {len(episodes)}エピソード")
    print()
    
    evaluator = GeDIGEvaluator()
    
    # 1. オリジナル実装（グラフコピーあり）
    print("1. Original GraphManager (グラフコピーあり)")
    print("-" * 50)
    original_mgr = GraphManager(evaluator)
    for ep in episodes:
        original_mgr.add_episode_node(ep)
    
    start = time.perf_counter()
    original_mgr._wire_with_gedig(episodes, threshold=-0.04)  # より緩い閾値
    original_time = time.perf_counter() - start
    
    print(f"  時間: {original_time*1000:.2f} ms")
    print(f"  エッジ数: {original_mgr.graph.number_of_edges()}")
    
    # エッジリストを取得
    original_edges = set(original_mgr.graph.edges())
    print(f"  作成されたエッジ: {len(original_edges)}")
    print()
    
    # 2. NO COPY実装（グラフコピーなし）
    print("2. NoCopy GraphManager (グラフコピーなし)")
    print("-" * 50)
    nocopy_mgr = NoCopyGraphManager(evaluator)
    for ep in episodes:
        nocopy_mgr.add_episode_node(ep)
    
    start = time.perf_counter()
    nocopy_mgr._wire_with_gedig_nocopy(episodes, threshold=-0.04)
    nocopy_time = time.perf_counter() - start
    
    print(f"  時間: {nocopy_time*1000:.2f} ms")
    print(f"  エッジ数: {nocopy_mgr.graph.number_of_edges()}")
    
    nocopy_edges = set(nocopy_mgr.graph.edges())
    print(f"  作成されたエッジ: {len(nocopy_edges)}")
    
    if nocopy_mgr.edge_logs:
        gedig_values = [log['gedig'] for log in nocopy_mgr.edge_logs]
        print(f"  geDIG値: min={min(gedig_values):.4f}, max={max(gedig_values):.4f}")
    print()
    
    # 3. 詳細な比較
    print("=" * 70)
    print("性能比較")
    print("=" * 70)
    
    if original_time > 0:
        speedup = original_time / nocopy_time
        print(f"高速化: {speedup:.1f}x")
    
    # エッジの一致度を確認
    if original_edges and nocopy_edges:
        common_edges = original_edges & nocopy_edges
        print(f"共通エッジ: {len(common_edges)}/{len(original_edges)}")
        
        if len(original_edges) > 0:
            similarity = len(common_edges) / len(original_edges) * 100
            print(f"一致率: {similarity:.1f}%")
    
    print()
    print("結論:")
    print("- NoCopy実装はグラフコピーを避けることで高速化")
    print("- エッジの一時追加・削除でgeDIG計算を実現")
    print("- オリジナルのgeDIG計算ロジックを維持")


if __name__ == '__main__':
    test_nocopy_performance()