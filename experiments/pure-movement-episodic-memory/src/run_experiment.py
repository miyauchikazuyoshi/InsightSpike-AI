#!/usr/bin/env python3
"""
純粋な移動エピソード記憶実験の実行スクリプト
"""

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Optional

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent import PureMemoryAgent


class PureMemoryExperiment:
    """純粋な記憶ベース実験"""
    
    def __init__(self, experiment_name: str = None):
        """
        Args:
            experiment_name: 実験名（タイムスタンプが自動付与される）
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"pure_memory_{self.timestamp}"
        
        # 実験ディレクトリ作成
        self.base_path = Path(f"../results/{self.experiment_name}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # DataStoreパス
        self.datastore_path = str(self.base_path / "datastore")
        
        # 結果保存用
        self.results = []
    
    def run_single_maze(self, 
                       maze_size: Tuple[int, int],
                       seed: int,
                       max_steps: int,
                       config: Optional[Dict] = None) -> Dict:
        """単一の迷路で実験を実行"""
        
        print(f"\n{'='*60}")
        print(f"Maze Size: {maze_size[0]}×{maze_size[1]}, Seed: {seed}")
        print(f"{'='*60}")
        
        # 迷路生成
        generator = ProperMazeGenerator()
        maze = generator.generate_dfs_maze(size=maze_size, seed=seed)
        
        # エージェント作成
        agent_config = config or {
            'max_depth': 5,
            'search_k': 30
        }
        
        agent = PureMemoryAgent(
            maze=maze,
            datastore_path=f"{self.datastore_path}/maze_{maze_size[0]}x{maze_size[1]}_seed{seed}",
            config=agent_config
        )
        
        print(f"Start: {agent.position}, Goal: {agent.goal}")
        print(f"Max steps: {max_steps}")
        print("-" * 40)
        
        # 実験実行
        start_time = time.time()
        steps = 0
        
        for step in range(max_steps):
            steps = step
            
            # ゴール到達チェック
            if agent.is_goal_reached():
                break
            
            # 行動決定と実行
            action = agent.get_action()
            agent.execute_action(action)
            
            # 進捗報告
            if step % 1000 == 0 and step > 0:
                stats = agent.get_statistics()
                print(f"Step {step}: pos={stats['position']}, "
                      f"dist={stats['distance_to_goal']}, "
                      f"wall_hits={stats['wall_hits']} "
                      f"({stats['wall_hits']/step*100:.1f}%)")
        
        # 実験終了
        total_time = time.time() - start_time
        final_stats = agent.get_statistics()
        
        # 結果作成
        result = {
            'success': agent.is_goal_reached(),
            'maze_size': maze_size,
            'seed': seed,
            'steps': steps,
            'total_time': total_time,
            'total_episodes': final_stats['total_episodes'],
            'wall_hits': final_stats['wall_hits'],
            'wall_hit_rate': final_stats['wall_hits'] / max(steps, 1),
            'path_length': final_stats['path_length'],
            'distance_to_goal': final_stats['distance_to_goal'],
            'avg_search_time': final_stats['avg_search_time'],
            'depth_usage': final_stats['depth_usage'],
            'config': agent_config
        }
        
        # 結果表示
        if result['success']:
            print(f"\n✅ SUCCESS in {steps} steps!")
        else:
            print(f"\n❌ Failed after {max_steps} steps")
            print(f"   Final distance to goal: {result['distance_to_goal']}")
        
        print(f"Wall hit rate: {result['wall_hit_rate']:.2%}")
        print(f"Total episodes: {result['total_episodes']}")
        print(f"Path length: {result['path_length']}")
        
        # パスとvisit_countsを保存
        self._save_detailed_result(
            agent, maze, result, 
            f"maze_{maze_size[0]}x{maze_size[1]}_seed{seed}"
        )
        
        return result
    
    def _save_detailed_result(self, agent, maze, result, name):
        """詳細な結果を保存"""
        # 結果をJSON形式で保存
        result_path = self.base_path / f"{name}_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # パスを保存
        path_data = {
            'path': [list(p) for p in agent.stats['path']],
            'visit_counts': {f"{k[0]},{k[1]}": v 
                           for k, v in agent.visit_counts.items()}
        }
        path_path = self.base_path / f"{name}_path.json"
        with open(path_path, 'w') as f:
            json.dump(path_data, f, indent=2)
        
        # 迷路を保存
        maze_path = self.base_path / f"{name}_maze.npy"
        np.save(maze_path, maze)
    
    def run_experiment_suite(self):
        """実験スイートを実行"""
        print("\n" + "="*60)
        print("PURE MOVEMENT EPISODIC MEMORY EXPERIMENT")
        print("No bonuses, no penalties - just pure memory")
        print("="*60)
        
        # 実験設定
        experiments = [
            # 小規模（学習確認）
            {'size': (15, 15), 'seeds': [42, 123, 456], 'max_steps': 2250},
            # 中規模（性能評価）
            {'size': (25, 25), 'seeds': [42, 123], 'max_steps': 6250},
            # 大規模（本実験）
            {'size': (51, 51), 'seeds': [42], 'max_steps': 26010},
        ]
        
        all_results = []
        
        for exp in experiments:
            size = exp['size']
            print(f"\n{'='*60}")
            print(f"Testing {size[0]}×{size[1]} mazes")
            print(f"{'='*60}")
            
            size_results = []
            
            for seed in exp['seeds']:
                result = self.run_single_maze(
                    maze_size=size,
                    seed=seed,
                    max_steps=exp['max_steps']
                )
                size_results.append(result)
                all_results.append(result)
            
            # サイズごとの統計
            self._print_size_statistics(size, size_results)
        
        # 全体統計
        self._print_overall_statistics(all_results)
        
        # 結果を保存
        self._save_experiment_summary(all_results)
        
        return all_results
    
    def _print_size_statistics(self, size, results):
        """サイズごとの統計を表示"""
        successes = [r for r in results if r['success']]
        
        print(f"\n{size[0]}×{size[1]} Statistics:")
        print(f"  Success rate: {len(successes)}/{len(results)} "
              f"({len(successes)/len(results)*100:.1f}%)")
        
        if successes:
            avg_steps = np.mean([r['steps'] for r in successes])
            avg_wall_hit = np.mean([r['wall_hit_rate'] for r in successes])
            avg_episodes = np.mean([r['total_episodes'] for r in successes])
            
            print(f"  Avg steps (success): {avg_steps:.0f}")
            print(f"  Avg wall hit rate: {avg_wall_hit:.2%}")
            print(f"  Avg episodes: {avg_episodes:.0f}")
    
    def _print_overall_statistics(self, results):
        """全体統計を表示"""
        print("\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        
        by_size = {}
        for r in results:
            size_key = f"{r['maze_size'][0]}x{r['maze_size'][1]}"
            if size_key not in by_size:
                by_size[size_key] = []
            by_size[size_key].append(r)
        
        for size_key, size_results in by_size.items():
            successes = [r for r in size_results if r['success']]
            success_rate = len(successes) / len(size_results)
            
            print(f"\n{size_key}:")
            print(f"  Success rate: {success_rate:.1%}")
            
            if successes:
                print(f"  Avg steps: {np.mean([r['steps'] for r in successes]):.0f}")
                print(f"  Avg wall hit rate: {np.mean([r['wall_hit_rate'] for r in successes]):.2%}")
    
    def _save_experiment_summary(self, results):
        """実験サマリーを保存"""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'total_runs': len(results),
            'results': results,
            'statistics': {
                'overall_success_rate': sum(r['success'] for r in results) / len(results),
                'avg_wall_hit_rate': np.mean([r['wall_hit_rate'] for r in results]),
                'avg_episodes': np.mean([r['total_episodes'] for r in results])
            }
        }
        
        summary_path = self.base_path / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Results saved to: {self.base_path}")


def main():
    """メイン実行関数"""
    # 実験実行
    experiment = PureMemoryExperiment()
    results = experiment.run_experiment_suite()
    
    # 成功判定
    success_rate = sum(r['success'] for r in results) / len(results)
    
    print("\n" + "="*60)
    if success_rate >= 0.7:
        print("🎉 EXPERIMENT SUCCESS!")
        print(f"   Success rate: {success_rate:.1%}")
        print("   Pure memory-based navigation works!")
    else:
        print("📊 EXPERIMENT COMPLETE")
        print(f"   Success rate: {success_rate:.1%}")
        print("   Further optimization may be needed")
    print("="*60)


if __name__ == "__main__":
    main()