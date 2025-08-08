#!/usr/bin/env python3
"""
ゴールフラグの効果を分析
ゴール=1.0が目的関数の代替として機能しているか検証
"""

import numpy as np
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_adaptive import PureMemoryAgentAdaptive
from pure_memory_agent_goal_oriented import PureMemoryAgentGoalOriented


def analyze_goal_flag_effect():
    """ゴールフラグの効果を詳細分析"""
    
    print("="*70)
    print("🔬 ゴールフラグ効果分析")
    print("  仮説：ゴール=1.0は目的関数の代替として機能")
    print("="*70)
    
    # 7×7迷路
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(7, 7), seed=42)
    
    print("\n迷路 (7×7):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '█' for x in row]))
    
    # ゴール指向エージェント
    agent = PureMemoryAgentGoalOriented(
        maze=maze,
        datastore_path="../results/goal_effect_analysis",
        config={
            'max_depth': 3,
            'search_k': 10,
            'gedig_improvement_threshold': 0.05
        }
    )
    
    print(f"\nスタート: {agent.position}, ゴール: {agent.goal}")
    print("-" * 40)
    
    # 検索結果を詳細に記録
    search_results = []
    goal_episode_ranks = []
    
    # 50ステップ実行
    for step in range(50):
        if agent.is_goal_reached():
            print(f"\n✅ ゴール到達！ {step}ステップ")
            break
        
        # 視覚観測追加
        agent._add_visual_observations()
        
        # クエリ生成
        query = agent._create_goal_oriented_query()
        
        # 検索実行
        indices, scores = agent.index.search(query, k=agent.search_k, mode='hybrid')
        
        if len(indices) > 0:
            # ゴールエピソードの順位を記録
            goal_ranks = []
            for rank, idx in enumerate(indices[:10]):
                if idx < len(agent.index.metadata):
                    episode = agent.index.metadata[idx]
                    vec = episode['vec']
                    
                    # ゴールフラグをチェック
                    if vec[6] > 0.5:  # ゴールエピソード
                        goal_ranks.append(rank + 1)
                        if step < 10:  # 最初の10ステップは詳細表示
                            print(f"  Step {step}: ゴールエピソードが{rank+1}位で発見！")
            
            if goal_ranks:
                goal_episode_ranks.append(min(goal_ranks))  # 最高順位を記録
            else:
                goal_episode_ranks.append(-1)  # ゴールエピソードなし
            
            # 上位エピソードの分析（最初の5ステップ）
            if step < 5:
                print(f"\nStep {step} - 検索結果上位5件:")
                for rank, idx in enumerate(indices[:5]):
                    if idx < len(agent.index.metadata):
                        episode = agent.index.metadata[idx]
                        vec = episode['vec']
                        meta = episode
                        
                        pos = (int(vec[0] * agent.height), int(vec[1] * agent.width))
                        success = "成功" if vec[3] > 0.5 else "失敗"
                        goal = "🎯" if vec[6] > 0.5 else ""
                        
                        print(f"  {rank+1}位: {pos} {success} {goal} (スコア: {scores[rank]:.3f})")
        
        # 行動実行
        action = agent.get_action()
        agent.execute_action(action)
    
    # ============================================================
    # 分析結果
    # ============================================================
    print("\n" + "="*70)
    print("📊 分析結果")
    print("="*70)
    
    # ゴールエピソードの検索順位
    valid_ranks = [r for r in goal_episode_ranks if r > 0]
    
    if valid_ranks:
        print(f"\n🎯 ゴールエピソードの検索順位:")
        print(f"  平均順位: {np.mean(valid_ranks):.1f}位")
        print(f"  最高順位: {min(valid_ranks)}位")
        print(f"  最低順位: {max(valid_ranks)}位")
        print(f"  上位3位以内率: {sum(1 for r in valid_ranks if r <= 3) / len(valid_ranks) * 100:.1f}%")
        
        # 時系列変化
        if len(valid_ranks) > 5:
            early = valid_ranks[:3]
            late = valid_ranks[-3:]
            print(f"\n  序盤の平均順位: {np.mean(early):.1f}位")
            print(f"  終盤の平均順位: {np.mean(late):.1f}位")
            
            if np.mean(late) < np.mean(early):
                print("  → 📈 学習とともにゴールエピソードの順位が上昇！")
    else:
        print("\n⚠️ ゴールエピソードがまだ生成されていません")
    
    # ゴールフラグの影響度
    print("\n💡 ゴールフラグの影響分析:")
    
    # クエリベクトルの各次元の寄与を推定
    query_example = agent._create_goal_oriented_query()
    print(f"\nクエリベクトル:")
    print(f"  位置: [{query_example[0]:.2f}, {query_example[1]:.2f}]")
    print(f"  方向: {query_example[2]:.2f}")
    print(f"  成功: {query_example[3]:.2f}")
    print(f"  通路: {query_example[4]:.2f}")
    print(f"  訪問: {query_example[5]:.2f}")
    print(f"  ゴール: {query_example[6]:.2f} ← これが1.0！")
    
    # 統計情報
    stats = agent.get_statistics()
    print(f"\n最終統計:")
    print(f"  壁衝突率: {stats['wall_hits']/max(step,1)*100:.1f}%")
    print(f"  総エピソード: {stats['total_episodes']}")
    
    # 結論
    print("\n" + "="*70)
    print("🔬 結論")
    print("="*70)
    
    if valid_ranks and np.mean(valid_ranks) < 5:
        print("✅ 仮説は支持される！")
        print("   ゴール=1.0によってゴールエピソードが優先的に検索され、")
        print("   目的関数の代替として機能している")
    else:
        print("📊 効果は限定的")
        print("   ゴールエピソードの生成後に効果が現れる可能性")
    
    print("\n📝 メカニズム:")
    print("  1. ゴール=1.0のクエリ → ゴールエピソードと高い類似度")
    print("  2. 検索で上位にランク → メッセージパッシングで強い影響")
    print("  3. ゴール方向への行動を誘導")
    print("\n  これは「ゴールの存在を知っている」という事前知識ですが、")
    print("  経路自体は純粋に経験から学習しています。")
    
    return agent.is_goal_reached()


if __name__ == "__main__":
    success = analyze_goal_flag_effect()
    
    print("\n" + "="*70)
    if success:
        print("🎯 分析完了：ゴール指向クエリの効果を確認")
    else:
        print("📊 分析完了：さらなる検証が必要")
    print("="*70)