#!/usr/bin/env python3
"""
ゴールビーコンエピソードのテスト
ゴール到達時に位置情報のみを持つビーコンを生成
"""

import numpy as np
import time
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_with_goal_beacon import PureMemoryAgentWithGoalBeacon


def test_goal_beacon():
    """ゴールビーコンの効果をテスト（2回の試行）"""
    
    print("="*70)
    print("🎯 ゴールビーコン実験")
    print("  1回目: ビーコンなしで探索")
    print("  2回目: ビーコンありで探索（1回目でゴール到達時に生成）")
    print("="*70)
    
    # 7×7迷路
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(7, 7), seed=42)
    
    print("\n迷路 (7×7):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '█' for x in row]))
    
    # エージェント作成（永続的なメモリ）
    agent = PureMemoryAgentWithGoalBeacon(
        maze=maze,
        datastore_path="../results/goal_beacon_test",
        config={
            'max_depth': 3,
            'search_k': 15,
            'gedig_improvement_threshold': 0.05
        }
    )
    
    print(f"\nスタート: {agent.position}, ゴール: {agent.goal}")
    
    # ============================================================
    # 1回目の試行（ビーコンなし）
    # ============================================================
    print("\n" + "="*70)
    print("📍 1回目の試行（ビーコンなし）")
    print("="*70)
    
    first_trial_steps = 0
    max_steps = 100
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            first_trial_steps = step
            print(f"\n✅ 1回目成功！ {step}ステップでゴール到達")
            print(f"  → ゴールビーコンが生成されました")
            break
        
        action = agent.get_action()
        agent.execute_action(action)
        
        if step % 20 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"  ステップ {step}: 距離={stats['distance_to_goal']}")
    else:
        print(f"\n❌ 1回目失敗（{max_steps}ステップ）")
        first_trial_steps = max_steps
    
    stats1 = agent.get_statistics()
    print(f"\n1回目の統計:")
    print(f"  壁衝突率: {stats1['wall_hits']/max(first_trial_steps,1)*100:.1f}%")
    print(f"  ビーコン生成: {stats1['goal_beacon_created']}")
    print(f"  ビーコン活性化: {stats1['goal_beacon_activations']}回")
    
    # ============================================================
    # エージェントをリセット（メモリは保持）
    # ============================================================
    print("\n" + "-"*70)
    print("🔄 エージェントをスタート位置にリセット（メモリは保持）")
    print("-"*70)
    
    # 位置だけリセット、メモリは保持
    agent.position = agent._find_start()
    agent.stats['path'] = [agent.position]
    agent.stats['wall_hits'] = 0
    
    # ============================================================
    # 2回目の試行（ビーコンあり）
    # ============================================================
    print("\n" + "="*70)
    print("🎯 2回目の試行（ビーコンあり）")
    print("="*70)
    
    beacon_activations_before = stats1['goal_beacon_activations']
    beacon_ranks = []
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            second_trial_steps = step
            print(f"\n✅ 2回目成功！ {step}ステップでゴール到達")
            break
        
        # 検索前のビーコン活性化数
        before_activations = agent.stats['goal_beacon_activations']
        
        action = agent.get_action()
        agent.execute_action(action)
        
        # ビーコンが活性化されたか
        if agent.stats['goal_beacon_activations'] > before_activations:
            current_rank = agent.stats['beacon_search_ranks'][-1] if agent.stats['beacon_search_ranks'] else -1
            beacon_ranks.append(current_rank)
            if step < 20:  # 最初の20ステップ
                print(f"  ステップ {step}: ビーコン活性化！（検索順位: {current_rank}位）")
        
        if step % 20 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"  ステップ {step}: 距離={stats['distance_to_goal']}")
    else:
        print(f"\n❌ 2回目失敗（{max_steps}ステップ）")
        second_trial_steps = max_steps
    
    stats2 = agent.get_statistics()
    
    # ============================================================
    # 比較分析
    # ============================================================
    print("\n" + "="*70)
    print("📊 比較分析")
    print("="*70)
    
    print(f"\n🏁 ステップ数:")
    print(f"  1回目（ビーコンなし）: {first_trial_steps}ステップ")
    print(f"  2回目（ビーコンあり）: {second_trial_steps}ステップ")
    
    if second_trial_steps < first_trial_steps:
        improvement = (first_trial_steps - second_trial_steps) / first_trial_steps * 100
        print(f"  → {improvement:.1f}% 改善！")
    
    print(f"\n🎯 ビーコン活性化:")
    total_activations = stats2['goal_beacon_activations'] - beacon_activations_before
    print(f"  2回目での活性化回数: {total_activations}回")
    
    if beacon_ranks:
        print(f"  平均検索順位: {np.mean(beacon_ranks):.1f}位")
        print(f"  最高順位: {min(beacon_ranks)}位")
    
    print(f"\n💡 ビーコンの効果:")
    if stats2['goal_beacon_created']:
        print("  ✅ ビーコンが生成され、メモリに保存された")
        
        if total_activations > 0:
            print("  ✅ ビーコンが検索で発見され、活用された")
            
            if second_trial_steps < first_trial_steps:
                print("  ✅ ビーコンにより経路探索が改善した")
            else:
                print("  📊 ビーコンは活用されたが、顕著な改善は見られない")
        else:
            print("  ⚠️ ビーコンが検索で発見されなかった")
    
    # ビーコンエピソードの内容を確認
    print(f"\n📝 ビーコンエピソードの構造:")
    beacon_vec = agent._create_goal_beacon_episode()
    print(f"  位置: [{beacon_vec[0]:.2f}, {beacon_vec[1]:.2f}] （ゴール位置）")
    print(f"  方向: {beacon_vec[2]:.2f} （中立）")
    print(f"  成功: {beacon_vec[3]:.2f} （中立）")
    print(f"  壁/通路: {beacon_vec[4]:.2f} （中立）")
    print(f"  訪問: {beacon_vec[5]:.2f} （未訪問）")
    print(f"  ゴール: {beacon_vec[6]:.2f} （ビーコン！）")
    
    return second_trial_steps < first_trial_steps


if __name__ == "__main__":
    improved = test_goal_beacon()
    
    print("\n" + "="*70)
    if improved:
        print("🎉 ゴールビーコンが効果的！")
        print("   位置情報のみのビーコンが経路探索を改善")
    else:
        print("📊 実験完了")
        print("   さらなる最適化が必要かも")
    print("="*70)