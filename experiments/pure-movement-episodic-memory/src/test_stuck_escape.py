#!/usr/bin/env python3
"""
袋小路脱出能力のテスト（11×11迷路）
深い推論が袋小路で活用されるか検証
"""

import numpy as np
import time
from collections import deque
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_final import PureMemoryAgentFinal


def detect_stuck_pattern(position_history, window=20):
    """袋小路パターンを検出"""
    if len(position_history) < window:
        return False, None
    
    recent = position_history[-window:]
    position_counts = {}
    
    for pos in recent:
        key = f"{pos[0]},{pos[1]}"
        position_counts[key] = position_counts.get(key, 0) + 1
    
    # 最頻出位置
    max_key = max(position_counts, key=position_counts.get)
    max_count = position_counts[max_key]
    
    # 60%以上同じ場所なら袋小路
    if max_count / window >= 0.6:
        parts = max_key.split(',')
        return True, (int(parts[0]), int(parts[1]))
    
    return False, None


def test_stuck_escape():
    """袋小路脱出テスト"""
    
    print("="*70)
    print("🔍 袋小路脱出能力テスト（11×11迷路）")
    print("  深い推論が袋小路で活用されるか検証")
    print("="*70)
    
    # 11×11迷路
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=123)  # 別のシード
    
    print("\n迷路 (11×11):")
    for row in maze:
        print(''.join(['.' if x == 0 else '█' for x in row]))
    
    # エージェント作成
    agent = PureMemoryAgentFinal(
        maze=maze,
        datastore_path="../results/stuck_escape_test",
        config={
            'max_depth': 5,
            'search_k': 25
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print("-" * 40)
    
    # 記録用
    position_history = []
    stuck_episodes = []
    depth_at_stuck = []
    
    # 実行
    max_steps = 1000
    stuck_detected = False
    stuck_start_step = 0
    
    for step in range(max_steps):
        # ゴール確認
        if agent.is_goal_reached():
            print(f"\n✅ 成功！ {step}ステップでゴール到達")
            break
        
        # 位置記録
        position_history.append(agent.position)
        
        # 袋小路検出
        is_stuck, stuck_pos = detect_stuck_pattern(position_history)
        
        if is_stuck and not stuck_detected:
            stuck_detected = True
            stuck_start_step = step
            
            # 現在の深度使用を記録
            before_depth = agent.stats['depth_usage'].copy()
            
            print(f"\n⚠️ 袋小路検出！ (Step {step})")
            print(f"  位置: {stuck_pos}")
            print(f"  現在の深度使用:")
            for d, count in before_depth.items():
                if count > 0:
                    print(f"    {d}ホップ: {count}回")
            
            stuck_episodes.append({
                'step': step,
                'position': stuck_pos,
                'depth_before': before_depth
            })
        
        elif stuck_detected and not is_stuck:
            # 脱出成功！
            escape_steps = step - stuck_start_step
            print(f"\n✅ 袋小路から脱出！ ({escape_steps}ステップ)")
            
            # 脱出時の深度使用
            after_depth = agent.stats['depth_usage'].copy()
            depth_diff = {}
            
            for d in range(1, 6):
                before = stuck_episodes[-1]['depth_before'].get(d, 0)
                after = after_depth.get(d, 0)
                depth_diff[d] = after - before
            
            print(f"  脱出時の深度使用:")
            total_diff = sum(depth_diff.values())
            for d, count in depth_diff.items():
                if count > 0:
                    ratio = count / total_diff * 100
                    print(f"    {d}ホップ: {count}回 ({ratio:.1f}%)")
            
            # 深い推論の割合
            deep = sum(depth_diff[d] for d in range(3, 6))
            shallow = sum(depth_diff[d] for d in range(1, 3))
            
            if deep + shallow > 0:
                deep_ratio = deep / (deep + shallow) * 100
                print(f"  深い推論（3-5ホップ）: {deep_ratio:.1f}%")
                
                if deep_ratio > 40:
                    print("  → 🎯 深い推論が脱出に貢献！")
            
            depth_at_stuck.append(depth_diff)
            stuck_detected = False
        
        # 行動
        action = agent.get_action()
        agent.execute_action(action)
        
        # 軽い進捗
        if step % 100 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"\nStep {step}: 距離={stats['distance_to_goal']}, "
                  f"壁={stats['wall_hit_rate']:.1f}%")
    
    # 最終分析
    print("\n" + "="*70)
    print("📊 袋小路分析結果")
    print("="*70)
    
    if stuck_episodes:
        print(f"\n袋小路検出回数: {len(stuck_episodes)}")
        
        if depth_at_stuck:
            # 平均深度使用
            avg_deep = 0
            avg_shallow = 0
            
            for depth_diff in depth_at_stuck:
                avg_deep += sum(depth_diff[d] for d in range(3, 6))
                avg_shallow += sum(depth_diff[d] for d in range(1, 3))
            
            avg_deep /= len(depth_at_stuck)
            avg_shallow /= len(depth_at_stuck)
            
            print(f"\n脱出時の平均深度使用:")
            print(f"  浅い推論（1-2ホップ）: {avg_shallow:.1f}回")
            print(f"  深い推論（3-5ホップ）: {avg_deep:.1f}回")
            
            if avg_deep > avg_shallow:
                print("\n✨ 結論: 袋小路脱出で深い推論が主に使用されている！")
            else:
                print("\n📊 結論: 袋小路でも主に浅い推論で対処")
    else:
        print("袋小路は検出されませんでした")
    
    # メモリ統計
    stats = agent.get_statistics()
    print(f"\n最終メモリ統計:")
    print(f"  総エピソード: {stats['total_episodes']}")
    print(f"  平均geDIG: {stats['memory_stats'].get('avg_gedig', 0):.3f}")


if __name__ == "__main__":
    test_stuck_escape()