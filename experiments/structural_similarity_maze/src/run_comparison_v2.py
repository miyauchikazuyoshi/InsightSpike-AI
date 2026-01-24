"""
構造類似度比較実験 v2

異なる迷路タイプ（行き止まり多い）で比較
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from navigator import MazeNavigator, NavigationResult
from maze_generators import (
    generate_perfect_maze,
    generate_dead_end_maze,
    generate_branching_maze,
    count_dead_ends,
)
import json
from datetime import datetime


@dataclass
class ExperimentResult:
    """実験結果"""
    maze_type: str
    size: int
    condition: str
    beta: float
    success_rate: float
    avg_steps: float
    std_steps: float
    avg_dead_ends_encountered: float
    avg_backtracks: float
    num_trials: int
    dead_ends_in_maze: float  # 迷路内の行き止まり数


def run_comparison_v2():
    """行き止まりが多い迷路での比較実験"""

    sizes = [15, 21, 25]
    seeds = list(range(20))  # 20シード
    beta_values = [0.0, 0.3, 0.5]  # 0.0 = baseline

    maze_types = {
        'perfect': generate_perfect_maze,
        'dead_end': lambda s, seed: generate_dead_end_maze(s, seed, dead_end_ratio=0.4),
        'branching': lambda s, seed: generate_branching_maze(s, seed, branch_probability=0.3),
    }

    all_results = []

    for maze_type, generator in maze_types.items():
        print(f"\n{'='*60}")
        print(f"Maze Type: {maze_type}")
        print(f"{'='*60}")

        for size in sizes:
            print(f"\n--- Size: {size}x{size} ---")

            max_steps = size * 15  # 行き止まり多いので余裕を持たせる

            results_by_beta = {beta: [] for beta in beta_values}
            dead_ends_counts = []

            for seed in seeds:
                maze = generator(size, seed)
                start = (1, 1)
                goal = (size - 2, size - 2)
                dead_ends_counts.append(count_dead_ends(maze))

                for beta in beta_values:
                    use_ss = beta > 0
                    navigator = MazeNavigator(
                        use_structural_similarity=use_ss,
                        beta=beta,
                        max_steps=max_steps,
                    )
                    result = navigator.navigate(maze, start, goal)
                    results_by_beta[beta].append(result)

            avg_dead_ends_in_maze = np.mean(dead_ends_counts)

            # 集計
            for beta in beta_values:
                results = results_by_beta[beta]
                successes = [r for r in results if r.success]
                success_rate = len(successes) / len(results)

                if successes:
                    steps = [r.steps for r in successes]
                    dead_enc = [r.dead_end_encounters for r in successes]
                    backtracks = [r.backtrack_count for r in successes]
                else:
                    steps = [r.steps for r in results]
                    dead_enc = [r.dead_end_encounters for r in results]
                    backtracks = [r.backtrack_count for r in results]

                condition = "Baseline" if beta == 0 else f"SS(β={beta})"

                exp_result = ExperimentResult(
                    maze_type=maze_type,
                    size=size,
                    condition=condition,
                    beta=beta,
                    success_rate=success_rate,
                    avg_steps=np.mean(steps),
                    std_steps=np.std(steps),
                    avg_dead_ends_encountered=np.mean(dead_enc),
                    avg_backtracks=np.mean(backtracks),
                    num_trials=len(results),
                    dead_ends_in_maze=avg_dead_ends_in_maze,
                )
                all_results.append(exp_result)

            # 結果表示
            baseline = next(r for r in all_results if r.maze_type == maze_type and r.size == size and r.beta == 0)
            print(f"  Dead-ends in maze: {avg_dead_ends_in_maze:.1f}")
            print(f"  Baseline: success={baseline.success_rate:.1%}, steps={baseline.avg_steps:.1f}±{baseline.std_steps:.1f}, "
                  f"backtracks={baseline.avg_backtracks:.1f}")

            for beta in [b for b in beta_values if b > 0]:
                ss_result = next(r for r in all_results if r.maze_type == maze_type and r.size == size and r.beta == beta)
                if baseline.avg_steps > 0:
                    improvement = (baseline.avg_steps - ss_result.avg_steps) / baseline.avg_steps * 100
                else:
                    improvement = 0
                print(f"  SS(β={beta}): success={ss_result.success_rate:.1%}, steps={ss_result.avg_steps:.1f}±{ss_result.std_steps:.1f}, "
                      f"backtracks={ss_result.avg_backtracks:.1f}, improvement={improvement:+.1f}%")

    # サマリーテーブル
    print("\n" + "="*100)
    print("SUMMARY: Steps Improvement by Maze Type and Size")
    print("="*100)
    print(f"{'Maze Type':<12} {'Size':<8} {'Dead-ends':<12} {'Baseline Steps':<16} {'SS(β=0.3)':<16} {'Improvement':<12}")
    print("-"*100)

    for maze_type in maze_types.keys():
        for size in sizes:
            baseline = next(r for r in all_results if r.maze_type == maze_type and r.size == size and r.beta == 0)
            ss_result = next(r for r in all_results if r.maze_type == maze_type and r.size == size and r.beta == 0.3)

            if baseline.avg_steps > 0:
                improvement = (baseline.avg_steps - ss_result.avg_steps) / baseline.avg_steps * 100
            else:
                improvement = 0

            print(f"{maze_type:<12} {size:<8} {baseline.dead_ends_in_maze:<12.1f} "
                  f"{baseline.avg_steps:.1f}±{baseline.std_steps:.1f}  "
                  f"{ss_result.avg_steps:.1f}±{ss_result.std_steps:.1f}  "
                  f"{improvement:+.1f}%")

    # 結果保存
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    output_path = os.path.join(
        output_dir, "results",
        f"comparison_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    data = {
        'timestamp': datetime.now().isoformat(),
        'results': [
            {
                'maze_type': r.maze_type,
                'size': r.size,
                'condition': r.condition,
                'beta': r.beta,
                'success_rate': r.success_rate,
                'avg_steps': r.avg_steps,
                'std_steps': r.std_steps,
                'avg_dead_ends_encountered': r.avg_dead_ends_encountered,
                'avg_backtracks': r.avg_backtracks,
                'num_trials': r.num_trials,
                'dead_ends_in_maze': r.dead_ends_in_maze,
            }
            for r in all_results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    run_comparison_v2()
