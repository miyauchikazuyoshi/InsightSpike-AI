#!/usr/bin/env python3
"""
深度使用パターンの簡易分析
経験が増えるにつれて深い推論が使われるか確認
"""

import numpy as np
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_final import PureMemoryAgentFinal


def test_depth_progression():
    """深度使用の進行を分析"""
    
    print("="*60)
    print("📊 深度使用パターン分析")
    print("  経験蓄積と深い推論の関係")
    print("="*60)
    
    # 9×9迷路（処理を軽く）
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(9, 9), seed=42)
    
    print("\n迷路 (9×9):")
    for row in maze:
        print(''.join(['.' if x == 0 else '█' for x in row]))
    
    # エージェント作成
    agent = PureMemoryAgentFinal(
        maze=maze,
        datastore_path="../results/depth_analysis",
        config={
            'max_depth': 4,
            'search_k': 15
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print("-" * 40)
    
    # 深度使用を段階的に記録
    depth_snapshots = []
    
    # 200ステップ実行
    for step in range(200):
        if agent.is_goal_reached():
            print(f"\n✅ ゴール到達！ ({step}ステップ)")
            break
        
        # 行動
        action = agent.get_action()
        agent.execute_action(action)
        
        # 50ステップごとにスナップショット
        if step % 50 == 49:
            stats = agent.get_statistics()
            depth_usage = stats['depth_usage'].copy()
            total = sum(depth_usage.values())
            
            snapshot = {
                'step': step + 1,
                'episodes': stats['total_episodes'],
                'depth_usage': depth_usage,
                'total': total
            }
            depth_snapshots.append(snapshot)
            
            print(f"\n📸 Step {step+1} スナップショット:")
            print(f"  エピソード数: {stats['total_episodes']}")
            print(f"  深度使用:")
            
            for d in range(1, 5):
                count = depth_usage.get(d, 0)
                if total > 0:
                    ratio = count / total * 100
                    bar = '█' * int(ratio / 10)
                    print(f"    {d}ホップ: {ratio:5.1f}% {bar}")
    
    # 分析
    print("\n" + "="*60)
    print("📈 深度使用の変化")
    print("="*60)
    
    if len(depth_snapshots) >= 2:
        # 初期と後期の比較
        early = depth_snapshots[0]
        late = depth_snapshots[-1]
        
        print(f"\n初期（{early['step']}ステップ時）:")
        early_deep = sum(early['depth_usage'].get(d, 0) for d in range(3, 5))
        early_shallow = sum(early['depth_usage'].get(d, 0) for d in range(1, 3))
        
        if early['total'] > 0:
            print(f"  浅い推論（1-2）: {early_shallow/early['total']*100:.1f}%")
            print(f"  深い推論（3-4）: {early_deep/early['total']*100:.1f}%")
        
        print(f"\n後期（{late['step']}ステップ時）:")
        late_deep = sum(late['depth_usage'].get(d, 0) for d in range(3, 5))
        late_shallow = sum(late['depth_usage'].get(d, 0) for d in range(1, 3))
        
        if late['total'] > 0:
            print(f"  浅い推論（1-2）: {late_shallow/late['total']*100:.1f}%")
            print(f"  深い推論（3-4）: {late_deep/late['total']*100:.1f}%")
        
        # 変化の評価
        if late['total'] > 0 and early['total'] > 0:
            deep_change = (late_deep/late['total']) - (early_deep/early['total'])
            
            print(f"\n変化:")
            if deep_change > 0.1:
                print(f"  ✅ 深い推論が {deep_change*100:.1f}% 増加")
                print("  → 経験の蓄積とともに深い推論を活用！")
            elif deep_change < -0.1:
                print(f"  📉 深い推論が {-deep_change*100:.1f}% 減少")
                print("  → タスクが簡単になり浅い推論で十分")
            else:
                print(f"  📊 深度使用は安定")
    
    # 最終統計
    final_stats = agent.get_statistics()
    print(f"\n最終統計:")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    print(f"  総エピソード: {final_stats['total_episodes']}")
    
    mem_stats = final_stats['memory_stats']
    if 'avg_gedig' in mem_stats:
        print(f"  平均geDIG: {mem_stats['avg_gedig']:.3f}")
        
        if mem_stats['avg_gedig'] < 0:
            print("  → 情報利得が編集距離を上回る（良好な結合）")


if __name__ == "__main__":
    test_depth_progression()