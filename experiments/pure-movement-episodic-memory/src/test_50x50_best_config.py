#!/usr/bin/env python3
"""
50×50迷路での最善設定テスト
ゴール指向クエリ + geDIG適応的深度選択
"""

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_goal_oriented import PureMemoryAgentGoalOriented


def test_50x50_maze():
    """50×50迷路での大規模テスト"""
    
    print("="*80)
    print("🏔️ 50×50 大規模迷路チャレンジ")
    print("  設定: ゴール指向クエリ（訪問=0、ゴール=1.0）")
    print("       geDIG適応的深度選択（最大5ホップ）")
    print("="*80)
    
    # 結果保存ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(f"../results/50x50_challenge_{timestamp}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 50×50迷路生成（実際は51×51になる）
    print("\n🏗️ 50×50迷路を生成中...")
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(51, 51), seed=42)
    
    # 迷路を保存
    np.save(base_path / "maze.npy", maze)
    
    # 迷路の概要表示（全体は大きすぎるので一部のみ）
    print("\n🗺️ 迷路の一部（左上10×20）:")
    for i in range(10):
        row_str = ' '.join(['.' if maze[i][j] == 0 else '█' for j in range(20)])
        print(row_str + " ...")
    print("... (続く)")
    
    # エージェント作成（最善の設定）
    agent = PureMemoryAgentGoalOriented(
        maze=maze,
        datastore_path=str(base_path / "datastore"),
        config={
            'max_depth': 5,           # 最大5ホップ
            'search_k': 50,           # 大規模迷路用に増加
            'gedig_improvement_threshold': 0.05  # 5%改善で採用
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print(f"📏 直線距離: {abs(agent.position[0] - agent.goal[0]) + abs(agent.position[1] - agent.goal[1])}")
    
    # 実験パラメータ
    max_steps = 10000  # 50×50×4 = 10000
    checkpoint_interval = 500
    
    print(f"\n⚙️ 設定:")
    print(f"  最大ステップ数: {max_steps}")
    print(f"  チェックポイント: {checkpoint_interval}ステップごと")
    print(f"  検索数(k): 50")
    print(f"  最大深度: 5ホップ")
    print("-" * 40)
    
    # 実行
    start_time = time.time()
    checkpoints = []
    
    for step in range(max_steps):
        # ゴール到達チェック
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            stats = agent.get_statistics()
            
            print(f"\n🎉 成功！ {step}ステップでゴール到達")
            print(f"  実行時間: {elapsed:.2f}秒")
            print(f"  壁衝突率: {stats['wall_hits']/step*100:.1f}%")
            print(f"  総エピソード数: {stats['total_episodes']}")
            
            # 成功時の結果保存
            result = {
                'success': True,
                'steps': step,
                'time': elapsed,
                'wall_hits': stats['wall_hits'],
                'wall_hit_rate': stats['wall_hits'] / step,
                'total_episodes': stats['total_episodes'],
                'path_length': stats['path_length'],
                'depth_usage': stats['depth_usage'],
                'query_types': stats.get('query_types', {}),
                'checkpoints': checkpoints
            }
            
            with open(base_path / "result.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            # 深度使用統計
            print(f"\n📊 深度使用統計:")
            total_depth_usage = sum(stats['depth_usage'].values())
            for depth, count in sorted(stats['depth_usage'].items()):
                if count > 0:
                    percentage = count / total_depth_usage * 100
                    bar = '█' * int(percentage / 5)
                    print(f"  {depth}ホップ: {count:4d}回 ({percentage:5.1f}%) {bar}")
            
            # クエリタイプ統計
            qt = stats.get('query_types', {})
            if qt:
                total_queries = sum(qt.values())
                print(f"\n🔍 クエリタイプ:")
                print(f"  ゴール指向: {qt.get('goal_oriented', 0)} ({qt.get('goal_oriented', 0)/total_queries*100:.1f}%)")
                print(f"  探索: {qt.get('exploration', 0)} ({qt.get('exploration', 0)/total_queries*100:.1f}%)")
            
            return True
        
        # 行動実行
        action = agent.get_action()
        agent.execute_action(action)
        
        # チェックポイント
        if step % checkpoint_interval == 0 and step > 0:
            stats = agent.get_statistics()
            checkpoint_data = {
                'step': step,
                'distance': stats['distance_to_goal'],
                'wall_hits': stats['wall_hits'],
                'wall_hit_rate': stats['wall_hits'] / step,
                'episodes': stats['total_episodes'],
                'time': time.time() - start_time
            }
            checkpoints.append(checkpoint_data)
            
            print(f"\n📍 チェックポイント {step}:")
            print(f"  現在位置: {stats['position']}")
            print(f"  ゴールまでの距離: {stats['distance_to_goal']}")
            print(f"  壁衝突率: {checkpoint_data['wall_hit_rate']:.1%}")
            print(f"  総エピソード数: {stats['total_episodes']}")
            print(f"  経過時間: {checkpoint_data['time']:.1f}秒")
            
            # 進捗評価
            if len(checkpoints) >= 2:
                prev_dist = checkpoints[-2]['distance']
                curr_dist = checkpoints[-1]['distance']
                if curr_dist < prev_dist:
                    print(f"  → 📈 前進中！（距離が{prev_dist - curr_dist}減少）")
                elif curr_dist == prev_dist:
                    print(f"  → 📊 停滞中...")
                else:
                    print(f"  → 📉 後退？（距離が{curr_dist - prev_dist}増加）")
    
    # タイムアウト
    elapsed = time.time() - start_time
    stats = agent.get_statistics()
    
    print(f"\n⏱️ タイムアウト（{max_steps}ステップ）")
    print(f"  最終距離: {stats['distance_to_goal']}")
    print(f"  壁衝突率: {stats['wall_hits']/max_steps*100:.1f}%")
    print(f"  総エピソード数: {stats['total_episodes']}")
    print(f"  実行時間: {elapsed:.2f}秒")
    
    # 失敗時の結果保存
    result = {
        'success': False,
        'steps': max_steps,
        'time': elapsed,
        'wall_hits': stats['wall_hits'],
        'wall_hit_rate': stats['wall_hits'] / max_steps,
        'total_episodes': stats['total_episodes'],
        'final_distance': stats['distance_to_goal'],
        'path_length': stats['path_length'],
        'depth_usage': stats['depth_usage'],
        'query_types': stats.get('query_types', {}),
        'checkpoints': checkpoints
    }
    
    with open(base_path / "result.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    # パスの一部を保存（メモリ節約のため最初と最後の100ステップ）
    path_sample = {
        'first_100': [list(p) for p in agent.stats['path'][:100]],
        'last_100': [list(p) for p in agent.stats['path'][-100:]]
    }
    with open(base_path / "path_sample.json", 'w') as f:
        json.dump(path_sample, f, indent=2)
    
    print(f"\n📁 結果を保存: {base_path}")
    
    return False


def analyze_progress(checkpoints):
    """進捗を分析"""
    if not checkpoints:
        return
    
    print("\n" + "="*80)
    print("📈 進捗分析")
    print("="*80)
    
    # 距離の推移
    distances = [c['distance'] for c in checkpoints]
    min_dist = min(distances)
    max_dist = max(distances)
    
    print(f"\n距離の推移:")
    print(f"  初期: {distances[0]}")
    print(f"  最小: {min_dist}（ステップ{checkpoints[distances.index(min_dist)]['step']}）")
    print(f"  最終: {distances[-1]}")
    
    # 改善率
    improvement = (distances[0] - distances[-1]) / distances[0] * 100
    if improvement > 0:
        print(f"  改善率: {improvement:.1f}%")
    else:
        print(f"  悪化: {-improvement:.1f}%")
    
    # 壁衝突率の推移
    wall_hit_rates = [c['wall_hit_rate'] for c in checkpoints]
    print(f"\n壁衝突率の推移:")
    print(f"  初期: {wall_hit_rates[0]:.1%}")
    print(f"  最終: {wall_hit_rates[-1]:.1%}")
    
    if wall_hit_rates[-1] < wall_hit_rates[0]:
        print(f"  → 学習により壁回避が改善")


if __name__ == "__main__":
    success = test_50x50_maze()
    
    print("\n" + "="*80)
    if success:
        print("🏆 50×50迷路攻略成功！")
        print("   純粋記憶ベースで大規模迷路を解決")
        print("   geDIG適応的深度選択が効果的に機能")
    else:
        print("📊 50×50迷路は未攻略")
        print("   より長い学習時間が必要")
        print("   または追加の最適化が必要")
    print("="*80)