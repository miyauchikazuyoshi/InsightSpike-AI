#!/usr/bin/env python3
"""
11×11迷路での適応的深度選択クイックテスト（300ステップ限定）
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


def quick_11x11_test():
    """11×11迷路で適応的深度選択をクイックテスト"""
    
    print("="*70)
    print("🎯 11×11迷路 適応的geDIG深度選択 クイックテスト")
    print("  ※低いgeDIG値 = より良いエッジ品質")
    print("="*70)
    
    # 11×11迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=42)
    
    print("\n🗺️ 迷路 (11×11):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '█' for x in row]))
    
    # 適応的エージェント作成（軽い設定）
    agent = PureMemoryAgentAdaptive(
        maze=maze,
        datastore_path="../results/11x11_quick_test",
        config={
            'max_depth': 4,      # 最大4ホップ
            'search_k': 15,      # 検索数を減らす
            'gedig_improvement_threshold': 0.05
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print(f"⚙️ 最大300ステップ、最大深度4ホップ")
    print("-" * 40)
    
    # 深度選択を追跡
    depth_selections = []
    gedig_improvements = []
    
    # 300ステップ実行
    max_steps = 300
    start_time = time.time()
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            success = True
            break
        
        # 深度選択前の数を記録
        before_count = len(agent.stats['adaptive_depth_selections'])
        
        # 行動実行
        action = agent.get_action()
        agent.execute_action(action)
        
        # 新しい深度選択があった場合
        if len(agent.stats['adaptive_depth_selections']) > before_count:
            selected_depth = agent.stats['adaptive_depth_selections'][-1]
            depth_selections.append(selected_depth)
            
            # 最初の20ステップは詳細表示
            if step < 20 and agent.stats['gedig_evaluations']:
                latest_eval = agent.stats['gedig_evaluations'][-1]
                if len(latest_eval) > 0:
                    print(f"\nステップ {step}: 深度{selected_depth}を選択")
                    for depth, gedig_val in latest_eval[:3]:  # 最初の3深度
                        print(f"  {depth}ホップ: geDIG={gedig_val:.4f}")
                    
                    # 改善率計算
                    if len(latest_eval) > 1:
                        base = latest_eval[0][1]
                        selected_idx = min(selected_depth - 1, len(latest_eval) - 1)
                        selected_gedig = latest_eval[selected_idx][1]
                        improvement = (base - selected_gedig) / (base + 0.001)
                        gedig_improvements.append(improvement)
                        print(f"  → 改善率: {improvement:.3f}")
        
        # 進捗報告
        if step % 50 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"\n📊 ステップ {step}: 距離={stats['distance_to_goal']}, "
                  f"壁衝突率={stats['wall_hits']/step*100:.1f}%")
    else:
        success = False
    
    elapsed = time.time() - start_time
    final_stats = agent.get_statistics()
    
    # ============================================================
    # 結果分析
    # ============================================================
    print("\n" + "="*70)
    if success:
        print(f"✅ 成功！ {step}ステップでゴール到達")
    else:
        print(f"❌ 300ステップで未到達（最終距離: {final_stats['distance_to_goal']}）")
    
    print(f"\n📊 統計:")
    print(f"  壁衝突率: {final_stats['wall_hits']/max(step,1)*100:.1f}%")
    print(f"  総エピソード数: {final_stats['total_episodes']}")
    print(f"  実行時間: {elapsed:.2f}秒")
    print(f"  平均検索時間: {final_stats['avg_search_time']:.2f}ms")
    
    # 深度選択パターン
    if depth_selections:
        print(f"\n🔍 深度選択パターン（{len(depth_selections)}回の選択）:")
        print(f"  平均深度: {np.mean(depth_selections):.2f}")
        
        # 深度分布
        for depth in range(1, 5):
            count = depth_selections.count(depth)
            if count > 0:
                percentage = count / len(depth_selections) * 100
                bar = '█' * int(percentage / 3)
                print(f"  {depth}ホップ: {count:3d}回 ({percentage:5.1f}%) {bar}")
        
        # 学習による変化
        if len(depth_selections) > 20:
            early = depth_selections[:10]
            late = depth_selections[-10:]
            print(f"\n  序盤（最初10回）: 平均 {np.mean(early):.2f}")
            print(f"  終盤（最後10回）: 平均 {np.mean(late):.2f}")
            
            change = np.mean(late) - np.mean(early)
            if change > 0.3:
                print("  → 📈 深い探索を学習（複雑なパターン発見）")
            elif change < -0.3:
                print("  → 📉 浅い探索に収束（局所的パターンで十分）")
            else:
                print("  → 📊 安定した深度選択")
    
    # geDIG改善分析
    if gedig_improvements:
        print(f"\n💡 geDIG改善分析:")
        print(f"  平均改善率: {np.mean(gedig_improvements):.3f}")
        print(f"  最大改善: {max(gedig_improvements):.3f}")
        
        positive_improvements = [x for x in gedig_improvements if x > 0]
        if positive_improvements:
            print(f"  改善があった割合: {len(positive_improvements)/len(gedig_improvements)*100:.1f}%")
    
    # geDIG値の説明
    print("\n📝 geDIG値の解釈:")
    print("  ・低いgeDIG値 = エッジ品質が高い（情報価値が高い）")
    print("  ・高いgeDIG値 = エッジ品質が低い（冗長性が高い）")
    print("  ・深い探索でgeDIG値が下がる = より良い洞察を発見")
    
    return success


if __name__ == "__main__":
    success = quick_11x11_test()
    
    print("\n" + "="*70)
    if success:
        print("🎉 適応的geDIG深度選択が11×11迷路で成功！")
    else:
        print("📊 300ステップでは未完了、より長い学習が必要")
    print("="*70)