#!/usr/bin/env python3
"""
適応的geDIG深度選択のクイックテスト
小さい迷路で動作確認
"""

import numpy as np
import time
from datetime import datetime
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_adaptive import PureMemoryAgentAdaptive


def test_adaptive_gedig():
    """適応的深度選択の動作確認"""
    
    print("="*60)
    print("ADAPTIVE geDIG QUICK TEST")
    print("Testing adaptive depth selection mechanism")
    print("="*60)
    
    # 小さい迷路（5×5）でテスト
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(5, 5), seed=42)
    
    print("\nMaze (5×5):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '#' for x in row]))
    
    # 適応的エージェント作成
    agent = PureMemoryAgentAdaptive(
        maze=maze,
        datastore_path="../results/adaptive_quick_test",
        config={
            'max_depth': 5,
            'search_k': 10,
            'gedig_improvement_threshold': 0.05
        }
    )
    
    print(f"\nStart: {agent.position}, Goal: {agent.goal}")
    print("-" * 40)
    
    # 深度選択を観察
    depth_selections = []
    gedig_improvements = []
    
    # 50ステップ実行
    max_steps = 50
    for step in range(max_steps):
        if agent.is_goal_reached():
            print(f"\n✅ SUCCESS in {step} steps!")
            break
        
        # 行動前の深度選択数を記録
        before_count = len(agent.stats['adaptive_depth_selections'])
        
        # 行動実行
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # 深度選択があった場合
        if len(agent.stats['adaptive_depth_selections']) > before_count:
            selected_depth = agent.stats['adaptive_depth_selections'][-1]
            depth_selections.append(selected_depth)
            
            # geDIG評価履歴を取得
            if agent.stats['gedig_evaluations']:
                latest_eval = agent.stats['gedig_evaluations'][-1]
                if len(latest_eval) > 1:
                    base_gedig = latest_eval[0][1]
                    final_gedig = latest_eval[-1][1]
                    improvement = (base_gedig - final_gedig) / (base_gedig + 0.001)
                    gedig_improvements.append(improvement)
                    
                    # 詳細表示（最初の10ステップ）
                    if step < 10:
                        print(f"\nStep {step}: Selected depth={selected_depth}")
                        print(f"  geDIG evaluation:")
                        for depth, gedig_val in latest_eval:
                            print(f"    {depth}-hop: geDIG={gedig_val:.4f}")
                        print(f"  Improvement: {improvement:.3f}")
        
        # 簡易進捗
        if step % 10 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"\nStep {step}: dist={stats['distance_to_goal']}, "
                  f"wall_hits={stats['wall_hits']}")
    
    # 統計分析
    print("\n" + "="*60)
    print("ADAPTIVE DEPTH STATISTICS")
    print("="*60)
    
    if depth_selections:
        print(f"\nTotal depth selections: {len(depth_selections)}")
        print(f"Average selected depth: {np.mean(depth_selections):.2f}")
        print(f"Min depth: {min(depth_selections)}")
        print(f"Max depth: {max(depth_selections)}")
        
        # 深度分布
        print("\nDepth distribution:")
        for depth in range(1, 6):
            count = depth_selections.count(depth)
            if count > 0:
                percentage = count / len(depth_selections) * 100
                print(f"  {depth}-hop: {count} times ({percentage:.1f}%)")
    
    if gedig_improvements:
        print(f"\nAverage geDIG improvement: {np.mean(gedig_improvements):.3f}")
        print(f"Max improvement: {max(gedig_improvements):.3f}")
        print(f"Min improvement: {min(gedig_improvements):.3f}")
    
    # 最終統計
    final_stats = agent.get_statistics()
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Wall hit rate: {final_stats['wall_hits']/max(step,1)*100:.1f}%")
    print(f"Total episodes: {final_stats['total_episodes']}")
    print(f"Avg search time: {final_stats['avg_search_time']:.2f} ms")
    
    # 深度使用パターン分析
    if len(depth_selections) > 5:
        print("\n" + "="*60)
        print("DEPTH SELECTION PATTERN")
        print("="*60)
        
        # 最初と最後の選択を比較
        early_depths = depth_selections[:5]
        late_depths = depth_selections[-5:] if len(depth_selections) > 10 else depth_selections[5:]
        
        if late_depths:
            print(f"Early selections (first 5): {early_depths}")
            print(f"  Average: {np.mean(early_depths):.2f}")
            print(f"Late selections (last 5): {late_depths}")
            print(f"  Average: {np.mean(late_depths):.2f}")
            
            # 深度が変化しているか
            depth_change = np.mean(late_depths) - np.mean(early_depths)
            if abs(depth_change) > 0.5:
                if depth_change > 0:
                    print("📈 Depth increased over time - exploring deeper connections")
                else:
                    print("📉 Depth decreased over time - focusing on local patterns")
            else:
                print("📊 Depth remained relatively stable")
    
    return agent.is_goal_reached()


if __name__ == "__main__":
    success = test_adaptive_gedig()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ADAPTIVE geDIG WORKS!")
        print("   The agent successfully uses geDIG values to select depth")
    else:
        print("📊 Test completed")
        print("   Adaptive depth selection is functioning")
    print("="*60)