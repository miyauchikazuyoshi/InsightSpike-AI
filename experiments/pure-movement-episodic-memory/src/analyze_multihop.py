#!/usr/bin/env python3
"""
マルチホップ探索の効果を分析
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent import PureMemoryAgent


def analyze_multihop():
    """マルチホップの使用状況を詳細分析"""
    
    print("="*60)
    print("MULTI-HOP ANALYSIS")
    print("="*60)
    
    # 7×7迷路でテスト
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(7, 7), seed=42)
    
    print("Maze (7×7):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '#' for x in row]))
    
    # エージェント作成（深度5まで）
    agent = PureMemoryAgent(
        maze=maze,
        datastore_path="../results/multihop_analysis",
        config={
            'max_depth': 5,  # 5段階まで
            'search_k': 20
        }
    )
    
    print(f"\nStart: {agent.position}, Goal: {agent.goal}")
    print("-" * 40)
    
    # デバッグ用：各ステップでの洞察を記録
    step_insights = []
    
    # 100ステップ実行
    for step in range(100):
        if agent.is_goal_reached():
            print(f"\n✅ SUCCESS in {step} steps!")
            break
        
        # 洞察生成前の状態
        pos_before = agent.position
        
        # クエリ生成
        if agent.position not in agent.visit_counts:
            agent.visit_counts[agent.position] = 0
        agent.visit_counts[agent.position] += 1
        agent._add_visual_observations()
        
        query = agent._create_pure_query()
        
        # 各深度での洞察を記録
        indices, scores = agent.index.search(query, k=agent.search_k, mode='hybrid')
        
        if len(indices) > 0:
            depth_insights = []
            for depth in range(1, agent.max_depth + 1):
                insight = agent._pure_message_passing(indices.tolist(), depth)
                depth_insights.append({
                    'depth': depth,
                    'insight': insight,
                    'direction_value': insight[2],
                    'confidence': insight[3],
                    'norm': np.linalg.norm(insight)
                })
            
            step_insights.append({
                'step': step,
                'position': pos_before,
                'depth_insights': depth_insights
            })
        
        # 行動実行
        action = agent.get_action()
        agent.execute_action(action)
        
        if step % 20 == 0:
            stats = agent.get_statistics()
            print(f"Step {step}: dist={stats['distance_to_goal']}, "
                  f"episodes={stats['total_episodes']}")
    
    # マルチホップ使用統計
    stats = agent.get_statistics()
    print("\n" + "="*40)
    print("MULTI-HOP USAGE STATISTICS")
    print("="*40)
    
    total_usage = sum(stats['depth_usage'].values())
    if total_usage > 0:
        for depth, count in sorted(stats['depth_usage'].items()):
            percentage = count / total_usage * 100
            print(f"{depth}-hop: {count:4d} times ({percentage:5.1f}%)")
    
    # 深度ごとの洞察の質を分析
    print("\n" + "="*40)
    print("INSIGHT QUALITY BY DEPTH")
    print("="*40)
    
    if step_insights:
        depth_norms = {i: [] for i in range(1, 6)}
        depth_confidences = {i: [] for i in range(1, 6)}
        
        for step_data in step_insights[:20]:  # 最初の20ステップ
            for depth_insight in step_data['depth_insights']:
                d = depth_insight['depth']
                depth_norms[d].append(depth_insight['norm'])
                depth_confidences[d].append(depth_insight['confidence'])
        
        print("\nAverage insight norm by depth:")
        for depth in range(1, 6):
            if depth_norms[depth]:
                avg_norm = np.mean(depth_norms[depth])
                avg_conf = np.mean(depth_confidences[depth])
                print(f"  {depth}-hop: norm={avg_norm:.3f}, confidence={avg_conf:.3f}")
    
    # エピソード間の接続を確認
    print("\n" + "="*40)
    print("GRAPH CONNECTIVITY")
    print("="*40)
    
    graph = agent.index.graph
    if graph.number_of_nodes() > 0:
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        print(f"Density: {graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes()-1) / 2):.3f}")
        
        # 次数分布
        degrees = [graph.degree(n) for n in graph.nodes()]
        if degrees:
            print(f"Avg degree: {np.mean(degrees):.2f}")
            print(f"Max degree: {max(degrees)}")
            print(f"Min degree: {min(degrees)}")
    else:
        print("No graph connections yet")
    
    return stats


if __name__ == "__main__":
    stats = analyze_multihop()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    # マルチホップが機能しているかチェック
    if stats['depth_usage']:
        usage_values = list(stats['depth_usage'].values())
        if max(usage_values) == usage_values[0]:  # 1-hopが最も使われている
            print("⚠️  Multi-hop is called but 1-hop dominates")
            print("   Deeper propagation may not be effective")
        elif max(usage_values) == usage_values[-1]:  # 最深が最も使われている
            print("✅ Deep multi-hop is actively used")
            print("   Complex reasoning is happening")
        else:
            print("📊 Mixed multi-hop usage")
            print("   Adaptive depth selection is working")
    else:
        print("❌ No multi-hop data available")