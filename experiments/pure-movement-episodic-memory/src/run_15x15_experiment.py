#!/usr/bin/env python3
"""
15×15迷路での実験（計算時間を考慮）
"""

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import os
from typing import Dict

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent import PureMemoryAgent


def run_15x15_experiment():
    """15×15迷路で実験"""
    
    print("="*60)
    print("PURE MEMORY EXPERIMENT - 15×15 Maze")
    print("No bonuses, no penalties - just pure memory")
    print("="*60)
    
    # 実験ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(f"../results/experiment_15x15_{timestamp}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(15, 15), seed=42)
    
    # 迷路を保存
    np.save(base_path / "maze.npy", maze)
    
    print(f"Maze size: 15×15")
    print(f"Max steps: 2250 (15×15×10)")
    
    # エージェント作成
    agent = PureMemoryAgent(
        maze=maze,
        datastore_path=str(base_path / "datastore"),
        config={
            'max_depth': 5,
            'search_k': 20  # 15×15用に調整
        }
    )
    
    print(f"Start: {agent.position}, Goal: {agent.goal}")
    print("-" * 40)
    
    # 実験実行
    start_time = time.time()
    max_steps = 2250
    
    for step in range(max_steps):
        # ゴール到達チェック
        if agent.is_goal_reached():
            success = True
            break
        
        # 行動決定と実行
        action = agent.get_action()
        agent.execute_action(action)
        
        # 進捗報告
        if step % 100 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"Step {step}: pos={stats['position']}, "
                  f"dist={stats['distance_to_goal']}, "
                  f"wall_hits={stats['wall_hits']} "
                  f"({stats['wall_hits']/step*100:.1f}%), "
                  f"episodes={stats['total_episodes']}")
    else:
        success = False
    
    # 実験終了
    total_time = time.time() - start_time
    final_stats = agent.get_statistics()
    
    # 結果作成
    result = {
        'success': success,
        'maze_size': (15, 15),
        'seed': 42,
        'steps': step,
        'total_time': total_time,
        'total_episodes': final_stats['total_episodes'],
        'wall_hits': final_stats['wall_hits'],
        'wall_hit_rate': final_stats['wall_hits'] / max(step, 1),
        'path_length': final_stats['path_length'],
        'distance_to_goal': final_stats['distance_to_goal'],
        'avg_search_time': final_stats['avg_search_time'],
        'depth_usage': final_stats['depth_usage']
    }
    
    # 結果表示
    print("\n" + "="*60)
    if success:
        print(f"✅ SUCCESS in {step} steps!")
    else:
        print(f"❌ Failed after {max_steps} steps")
        print(f"   Final distance to goal: {result['distance_to_goal']}")
    
    print(f"Wall hit rate: {result['wall_hit_rate']:.2%}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Path length: {result['path_length']}")
    print(f"Total time: {result['total_time']:.2f} seconds")
    print(f"Avg search time: {result['avg_search_time']:.2f} ms")
    
    # 深度使用統計
    print("\nDepth usage:")
    for depth, count in final_stats['depth_usage'].items():
        print(f"  {depth}-hop: {count} times")
    
    # 結果を保存
    with open(base_path / "result.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    # パスを保存
    path_data = {
        'path': [list(p) for p in agent.stats['path']],
        'visit_counts': {f"{k[0]},{k[1]}": v 
                        for k, v in agent.visit_counts.items()}
    }
    with open(base_path / "path.json", 'w') as f:
        json.dump(path_data, f, indent=2)
    
    print(f"\n📁 Results saved to: {base_path}")
    
    return result


if __name__ == "__main__":
    result = run_15x15_experiment()
    
    # 成功判定
    print("\n" + "="*60)
    if result['success']:
        print("🎉 EXPERIMENT SUCCESS!")
        print("   Pure memory-based navigation works!")
        if result['wall_hit_rate'] < 0.25:
            print("   ⭐ Excellent wall avoidance!")
    else:
        print("📊 EXPERIMENT COMPLETE")
        print("   Further optimization may be needed")
    print("="*60)