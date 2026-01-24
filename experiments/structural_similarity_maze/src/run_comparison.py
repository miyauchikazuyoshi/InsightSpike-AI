"""
構造類似度の有無による比較実験

複数サイズ、複数シードで比較
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from navigator import MazeNavigator, generate_maze, NavigationResult
import json
from datetime import datetime


@dataclass
class ExperimentConfig:
    """実験設定"""
    sizes: List[int]
    seeds: List[int]
    beta_values: List[float]
    max_steps_multiplier: int = 10  # size * multiplier


@dataclass
class AggregatedResult:
    """集計結果"""
    size: int
    condition: str
    beta: float
    success_rate: float
    avg_steps: float
    avg_visited: float
    avg_dead_ends: float
    avg_backtracks: float
    std_steps: float
    num_trials: int


def run_single_experiment(
    maze: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    use_ss: bool,
    beta: float,
    max_steps: int,
) -> NavigationResult:
    """単一実験を実行"""
    navigator = MazeNavigator(
        use_structural_similarity=use_ss,
        beta=beta,
        max_steps=max_steps,
    )
    return navigator.navigate(maze, start, goal)


def run_comparison_experiment(config: ExperimentConfig) -> List[AggregatedResult]:
    """比較実験を実行"""
    results = []

    for size in config.sizes:
        print(f"\n{'='*50}")
        print(f"Size: {size}x{size}")
        print(f"{'='*50}")

        max_steps = size * config.max_steps_multiplier

        # 各条件の結果を蓄積
        baseline_results = []
        ss_results = {beta: [] for beta in config.beta_values}

        for seed in config.seeds:
            maze = generate_maze(size, seed=seed)
            start = (1, 1)
            goal = (size - 2, size - 2)

            # Baseline (no SS)
            result_base = run_single_experiment(
                maze, start, goal,
                use_ss=False, beta=0.0, max_steps=max_steps
            )
            baseline_results.append(result_base)

            # With SS (各beta値)
            for beta in config.beta_values:
                result_ss = run_single_experiment(
                    maze, start, goal,
                    use_ss=True, beta=beta, max_steps=max_steps
                )
                ss_results[beta].append(result_ss)

        # Baseline集計
        base_agg = aggregate_results(baseline_results, size, "Baseline", 0.0)
        results.append(base_agg)
        print(f"\nBaseline: success={base_agg.success_rate:.1%}, "
              f"steps={base_agg.avg_steps:.1f}±{base_agg.std_steps:.1f}")

        # SS条件集計
        for beta in config.beta_values:
            ss_agg = aggregate_results(ss_results[beta], size, f"SS(β={beta})", beta)
            results.append(ss_agg)

            # 改善率計算
            if base_agg.avg_steps > 0:
                improvement = (base_agg.avg_steps - ss_agg.avg_steps) / base_agg.avg_steps * 100
            else:
                improvement = 0

            print(f"SS(β={beta}): success={ss_agg.success_rate:.1%}, "
                  f"steps={ss_agg.avg_steps:.1f}±{ss_agg.std_steps:.1f}, "
                  f"improvement={improvement:+.1f}%")

    return results


def aggregate_results(
    results: List[NavigationResult],
    size: int,
    condition: str,
    beta: float,
) -> AggregatedResult:
    """結果を集計"""
    successes = [r for r in results if r.success]
    success_rate = len(successes) / len(results) if results else 0

    if successes:
        steps = [r.steps for r in successes]
        visited = [r.visited_count for r in successes]
        dead_ends = [r.dead_end_encounters for r in successes]
        backtracks = [r.backtrack_count for r in successes]
    else:
        steps = [r.steps for r in results]
        visited = [r.visited_count for r in results]
        dead_ends = [r.dead_end_encounters for r in results]
        backtracks = [r.backtrack_count for r in results]

    return AggregatedResult(
        size=size,
        condition=condition,
        beta=beta,
        success_rate=success_rate,
        avg_steps=np.mean(steps) if steps else 0,
        avg_visited=np.mean(visited) if visited else 0,
        avg_dead_ends=np.mean(dead_ends) if dead_ends else 0,
        avg_backtracks=np.mean(backtracks) if backtracks else 0,
        std_steps=np.std(steps) if steps else 0,
        num_trials=len(results),
    )


def save_results(results: List[AggregatedResult], output_path: str):
    """結果を保存"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'results': [
            {
                'size': r.size,
                'condition': r.condition,
                'beta': r.beta,
                'success_rate': r.success_rate,
                'avg_steps': r.avg_steps,
                'std_steps': r.std_steps,
                'avg_visited': r.avg_visited,
                'avg_dead_ends': r.avg_dead_ends,
                'avg_backtracks': r.avg_backtracks,
                'num_trials': r.num_trials,
            }
            for r in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def print_summary_table(results: List[AggregatedResult]):
    """サマリーテーブルを表示"""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Size':<8} {'Condition':<15} {'Success':<10} {'Steps':<15} {'Improvement':<12}")
    print("-"*80)

    # サイズごとにグループ化
    sizes = sorted(set(r.size for r in results))
    for size in sizes:
        size_results = [r for r in results if r.size == size]
        baseline = next((r for r in size_results if r.condition == "Baseline"), None)

        for r in size_results:
            if baseline and r.condition != "Baseline" and baseline.avg_steps > 0:
                improvement = (baseline.avg_steps - r.avg_steps) / baseline.avg_steps * 100
                imp_str = f"{improvement:+.1f}%"
            else:
                imp_str = "-"

            print(f"{r.size:<8} {r.condition:<15} {r.success_rate:<10.1%} "
                  f"{r.avg_steps:.1f}±{r.std_steps:.1f}  {imp_str:<12}")

        print("-"*80)


if __name__ == "__main__":
    # 実験設定
    config = ExperimentConfig(
        sizes=[11, 15, 21, 25],  # 迷路サイズ
        seeds=list(range(10)),   # 10シード
        beta_values=[0.1, 0.3, 0.5],  # 構造類似度の重み
    )

    print("="*60)
    print("構造類似度比較実験")
    print("="*60)
    print(f"Sizes: {config.sizes}")
    print(f"Seeds: {len(config.seeds)} trials per condition")
    print(f"Beta values: {config.beta_values}")

    # 実験実行
    results = run_comparison_experiment(config)

    # サマリー表示
    print_summary_table(results)

    # 結果保存
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    output_path = os.path.join(
        output_dir, "results",
        f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_results(results, output_path)
