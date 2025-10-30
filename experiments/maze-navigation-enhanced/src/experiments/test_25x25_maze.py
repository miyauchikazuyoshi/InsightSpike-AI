#!/usr/bin/env python3
"""
25×25大規模迷路での動作テスト
"""

import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import warnings

# Suppress noisy Pydantic v2 transition warnings (optional)
warnings.filterwarnings("ignore", message="Support for class-based `config` is deprecated")
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator
from maze_layouts import (
    create_large_maze,
    create_complex_maze,
    create_perfect_maze,
    LARGE_DEFAULT_START,
    LARGE_DEFAULT_GOAL,
    COMPLEX_DEFAULT_START,
    COMPLEX_DEFAULT_GOAL,
    PERFECT_DEFAULT_START,
    PERFECT_DEFAULT_GOAL,
)


def visualize_result(maze, path, start_pos, goal_pos, title="25×25 Maze Navigation"):
    """結果を可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左: 迷路全体とパス
    h, w = maze.shape
    
    # 迷路描画
    for y in range(h):
        for x in range(w):
            if maze[y, x] == 1:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray')
                ax1.add_patch(rect)
    
    # パス描画
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax1.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Path')
        
        # スタートとゴール
        ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
        ax1.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
    
    ax1.set_xlim(-0.5, w-0.5)
    ax1.set_ylim(-0.5, h-0.5)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title(f'{title} - Full Maze')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右: 訪問ヒートマップ
    visit_map = np.zeros_like(maze, dtype=float)
    for pos in path:
        visit_map[pos[1], pos[0]] += 1
    
    # 壁を-1に設定
    visit_map[maze == 1] = -1
    
    # ヒートマップ表示
    im = ax2.imshow(visit_map, cmap='hot', interpolation='nearest')
    ax2.plot(start_pos[0], start_pos[1], 'go', markersize=10)
    ax2.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10)
    ax2.set_title('Visit Frequency Heatmap')
    plt.colorbar(im, ax=ax2, label='Visit Count')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="25x25 Maze Navigation Benchmark (Simple Mode metrics)")
    parser.add_argument("--complex", action="store_true", help="複雑な迷路バリアントを実行")
    parser.add_argument("--perfect", action="store_true", help="パーフェクト迷路バリアントを実行")
    parser.add_argument("--max_steps", type=int, default=2000, help="各設定の最大ステップ")
    args = parser.parse_args()
    print("="*60)
    print("25×25 LARGE MAZE NAVIGATION TEST")
    print("="*60)
    
    # 迷路作成
    base_maze = create_large_maze()
    complex_maze = create_complex_maze() if args.complex else None
    perfect_maze = create_perfect_maze() if args.perfect else None

    # 実行対象 (名前, 迷路, start, goal)
    scenarios = [
        ("Large", base_maze, LARGE_DEFAULT_START, LARGE_DEFAULT_GOAL)
    ]
    if complex_maze is not None:
        scenarios.append(("Complex", complex_maze, COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL))
    if perfect_maze is not None:
        scenarios.append(("Perfect", perfect_maze, PERFECT_DEFAULT_START, PERFECT_DEFAULT_GOAL))
    
    for scenario_name, maze, start_pos, goal_pos in scenarios:
        print("\n" + "="*60)
        print(f"SCENARIO: {scenario_name}")
        print("="*60)
        print(f"Maze size: {maze.shape}")
        print(f"Start: {start_pos}, Goal: {goal_pos}")
        passages = np.sum(maze == 0)
        total = maze.size
        print(f"Passages: {passages}/{total} ({passages*100/total:.1f}%)")
        
        # 複数の設定でテスト
        configs = [
        {
            'name': 'Default',
            'temperature': 0.1,
            'gedig_threshold': 0.3,
            'backtrack_threshold': -0.2,
            'wiring_strategy': 'simple'
        },
        {
            'name': 'Aggressive Exploration',
            'temperature': 0.3,  # より探索的
            'gedig_threshold': 0.2,
            'backtrack_threshold': -0.1,  # 早めにバックトラック
            'wiring_strategy': 'simple'
        },
        {
            'name': 'Conservative',
            'temperature': 0.05,  # より活用的
            'gedig_threshold': 0.4,
            'backtrack_threshold': -0.3,  # 遅めにバックトラック
            'wiring_strategy': 'simple'
        }
        ]

        # デフォルト重み
        weights = np.array([
            1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.1, 0.0
        ])

        results = []

        for config in configs:
            print(f"\n" + "-"*60)
            print(f"Testing: {config['name']}")
            print("-"*60)

            # ナビゲーター作成
            navigator = MazeNavigator(
                maze=maze,
                start_pos=start_pos,
                goal_pos=goal_pos,
                weights=weights,
                temperature=config['temperature'],
                gedig_threshold=config['gedig_threshold'],
                backtrack_threshold=config['backtrack_threshold'],
                wiring_strategy=config['wiring_strategy'],
                simple_mode=True,
                backtrack_debounce=True
            )

            # 実行
            success = navigator.run(max_steps=args.max_steps)

            # 統計取得
            stats = navigator.get_statistics()

            result = {
                'config': config,
                'success': success,
                'steps': navigator.step_count,
                'path': navigator.path.copy(),
                'stats': stats
            }
            results.append(result)

            # 結果表示
            print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"  Steps: {navigator.step_count}")
            print(f"  Path length: {len(navigator.path)}")
            print(f"  Unique positions: {stats['unique_positions']}")
            print(f"  Episodes: {stats['episode_stats']['total_episodes']}")
            print(f"  Graph edges: {stats['graph_stats']['num_edges']}")
            sm = stats.get('simple_mode')
            if sm:
                print(f"  SimpleMode queries_per_step: {sm.get('queries_per_step'):.3f}")
                print(f"  SimpleMode backtrack_trigger_rate: {sm.get('backtrack_trigger_rate'):.3f}")
        
        if stats['branch_stats']['total_branch_points'] > 0:
            print(f"  Branch points: {stats['branch_stats']['total_branch_points']}")
            print(f"  Completed branches: {stats['branch_stats']['completed_branches']}")
    
    # 比較分析
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    print("\n設定別パフォーマンス:")
    print("Config | Success | Steps | Path Len | Unique Pos")
    print("-"*55)
    
    for r in results:
        success_mark = "✓" if r['success'] else "✗"
        print(f"{r['config']['name']:20} | {success_mark:^7} | {r['steps']:5d} | "
              f"{len(r['path']):8d} | {r['stats']['unique_positions']:10d}")
    
    # 最良の結果を可視化
        successful_results = [r for r in results if r['success']]
        if successful_results:
            best = min(successful_results, key=lambda x: x['steps'])
            print(f"\nBest configuration: {best['config']['name']}")
            print(f"  Completed in {best['steps']} steps")
            # 可視化
            fig = visualize_result(
                maze,
                best['path'],
                start_pos,
                goal_pos,
                title=f"{scenario_name} - Best {best['config']['name']}"
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'../../results/25x25_maze/{scenario_name.lower()}_{timestamp}.png'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {output_path}")
            plt.close(fig)
        else:
            print("\nNo successful navigation found!")
    
    # 結果をテキストで保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f'../../results/25x25_maze/{scenario_name.lower()}_results_{timestamp}.txt'
        with open(result_file, 'w') as f:
            f.write(f"25×25 Maze Navigation Test Results ({scenario_name})\n")
            f.write("="*60 + "\n\n")
            for r in results:
                sm = r['stats'].get('simple_mode')
                f.write(f"Configuration: {r['config']['name']}\n")
                f.write(f"  Success: {r['success']}\n")
                f.write(f"  Steps: {r['steps']}\n")
                f.write(f"  Path length: {len(r['path'])}\n")
                f.write(f"  Unique positions: {r['stats']['unique_positions']}\n")
                if sm:
                    f.write("  SimpleMode Metrics:\n")
                    f.write(f"    query_generated: {sm.get('query_generated')}\n")
                    f.write(f"    queries_per_step: {sm.get('queries_per_step')}\n")
                    f.write(f"    backtrack_trigger_rate: {sm.get('backtrack_trigger_rate')}\n")
                f.write("  Parameters:\n")
                for key, value in r['config'].items():
                    if key != 'name':
                        f.write(f"    {key}: {value}\n")
                f.write("-"*40 + "\n")
        print(f"Results saved: {result_file}")


if __name__ == "__main__":
    main()