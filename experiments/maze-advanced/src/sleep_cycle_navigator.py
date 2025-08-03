#!/usr/bin/env python3
"""
Sleep Cycle Navigator
=====================

睡眠サイクルによるエピソードグラフの最適化
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
import json
from datetime import datetime
import time
import random
import gc

# パスを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from test_visual_memory_maze import VisualMemoryNavigator, Episode7D, generate_complex_maze

try:
    from insightspike.environments.maze import SimpleMaze
except ImportError:
    from src.insightspike.environments.maze import SimpleMaze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeNode:
    """グラフのノードとしてのエピソード"""
    episode: Episode7D
    connections: Set[int]  # 接続されているエピソードのインデックス
    access_count: int = 0  # アクセス回数
    last_access: int = 0   # 最後にアクセスされたステップ
    
    def connection_density(self) -> float:
        """接続密度を計算"""
        return len(self.connections)


class SleepCycleNavigator(VisualMemoryNavigator):
    """睡眠サイクルを持つナビゲーター"""
    
    def __init__(self, maze_size: int = 30):
        super().__init__(maze_size)
        
        # エピソードグラフ
        self.episode_nodes: List[EpisodeNode] = []
        self.sleep_interval = 500  # 500ステップごとに睡眠
        self.density_threshold = 20  # 接続数がこれ以上なら過密
        self.min_connections = 1     # 最小接続数
        
        # 睡眠統計
        self.sleep_history = []
        
    def add_episode(self, episode: Episode7D) -> int:
        """エピソードを追加してインデックスを返す"""
        node = EpisodeNode(episode=episode, connections=set())
        self.episode_nodes.append(node)
        self.episodes.append(episode)
        return len(self.episode_nodes) - 1
    
    def connect_episodes(self, idx1: int, idx2: int):
        """2つのエピソードを接続"""
        if idx1 != idx2 and idx1 < len(self.episode_nodes) and idx2 < len(self.episode_nodes):
            self.episode_nodes[idx1].connections.add(idx2)
            self.episode_nodes[idx2].connections.add(idx1)
    
    def _search_episodes(self, queries: List[Episode7D]) -> List[Tuple[Episode7D, float]]:
        """検索（アクセス記録付き）"""
        results = super()._search_episodes(queries)
        
        # アクセスされたエピソードを記録
        for episode, score in results[:10]:  # トップ10のみ
            for i, node in enumerate(self.episode_nodes):
                if node.episode == episode:
                    node.access_count += 1
                    node.last_access = self.step_count
                    
                    # 関連エピソードとの接続を強化
                    for other_ep, other_score in results[:5]:
                        if other_ep != episode:
                            for j, other_node in enumerate(self.episode_nodes):
                                if other_node.episode == other_ep:
                                    self.connect_episodes(i, j)
        
        return results
    
    def sleep_cycle(self):
        """睡眠サイクル：グラフの整理"""
        print(f"\n💤 Sleep cycle at step {self.step_count}")
        
        initial_episodes = len(self.episode_nodes)
        initial_connections = sum(len(node.connections) for node in self.episode_nodes)
        
        # 1. 過密ノードのエッジ削除
        pruned_edges = self._prune_dense_connections()
        
        # 2. 遊離ノードの検出と削除
        removed_nodes = self._remove_isolated_nodes()
        
        # 3. 統計を記録
        final_episodes = len(self.episode_nodes)
        final_connections = sum(len(node.connections) for node in self.episode_nodes)
        
        self.sleep_history.append({
            'step': self.step_count,
            'initial_episodes': initial_episodes,
            'final_episodes': final_episodes,
            'removed_episodes': initial_episodes - final_episodes,
            'initial_connections': initial_connections,
            'final_connections': final_connections,
            'pruned_edges': pruned_edges,
            'removed_nodes': removed_nodes
        })
        
        print(f"   Pruned {pruned_edges} edges from dense nodes")
        print(f"   Removed {removed_nodes} isolated nodes")
        print(f"   Episodes: {initial_episodes} → {final_episodes}")
        print(f"   Connections: {initial_connections} → {final_connections}")
        
        # メモリ解放
        gc.collect()
    
    def _prune_dense_connections(self) -> int:
        """過密な接続を削除"""
        pruned = 0
        
        for i, node in enumerate(self.episode_nodes):
            if len(node.connections) > self.density_threshold:
                # 接続の重要度を評価
                connection_scores = []
                
                for conn_idx in node.connections:
                    conn_node = self.episode_nodes[conn_idx]
                    
                    # スコア計算（最近アクセスされた、アクセス頻度が高い、ゴール関連を優先）
                    score = 0.0
                    
                    # 最近のアクセス
                    recency = self.step_count - conn_node.last_access
                    score += 1.0 / (1.0 + recency * 0.001)
                    
                    # アクセス頻度
                    score += conn_node.access_count * 0.1
                    
                    # ゴール関連
                    if conn_node.episode.goal_or_not:
                        score += 10.0
                    
                    connection_scores.append((conn_idx, score))
                
                # スコアでソートして、下位の接続を削除
                connection_scores.sort(key=lambda x: x[1], reverse=True)
                keep_connections = self.density_threshold // 2  # 半分まで削減
                
                new_connections = set()
                for conn_idx, _ in connection_scores[:keep_connections]:
                    new_connections.add(conn_idx)
                
                # 削除される接続の相手側も更新
                for conn_idx in node.connections - new_connections:
                    if conn_idx < len(self.episode_nodes):
                        self.episode_nodes[conn_idx].connections.discard(i)
                        pruned += 1
                
                node.connections = new_connections
        
        return pruned
    
    def _remove_isolated_nodes(self) -> int:
        """遊離ノードを削除"""
        # 接続が少ないノードを特定
        nodes_to_remove = []
        
        for i, node in enumerate(self.episode_nodes):
            # ゴールノードは保護
            if node.episode.goal_or_not:
                continue
            
            # 接続が少なく、最近アクセスされていないノード
            if (len(node.connections) < self.min_connections and 
                self.step_count - node.last_access > 1000 and
                node.access_count < 2):
                nodes_to_remove.append(i)
        
        # 削除（後ろから処理してインデックスのズレを防ぐ）
        for i in sorted(nodes_to_remove, reverse=True):
            # このノードへの参照を削除
            removed_node = self.episode_nodes[i]
            for conn_idx in removed_node.connections:
                if conn_idx < len(self.episode_nodes):
                    self.episode_nodes[conn_idx].connections.discard(i)
            
            # ノードを削除
            del self.episode_nodes[i]
            del self.episodes[i]
            
            # 残りのノードの接続インデックスを更新
            for node in self.episode_nodes:
                node.connections = {
                    conn if conn < i else conn - 1
                    for conn in node.connections
                    if conn != i
                }
        
        return len(nodes_to_remove)
    
    def solve_maze(self, max_steps: int = 3000) -> Dict:
        """睡眠サイクル付きで迷路を解く"""
        self.setup_maze()
        
        # 初期エピソードをノードとして追加
        goal_idx = self.add_episode(self.episodes.pop())  # ゴールエピソード
        
        # 迷路情報
        maze_array = self.maze_env.grid
        total_cells = self.maze_size * self.maze_size
        wall_cells = np.sum(maze_array == 1)
        path_cells = total_cells - wall_cells
        
        print(f"\n=== Sleep Cycle Navigator ===")
        print(f"Maze size: {self.maze_size}x{self.maze_size}")
        print(f"Sleep interval: every {self.sleep_interval} steps")
        print(f"Density threshold: {self.density_threshold} connections\n")
        
        path_history_sparse = [self.position]
        all_positions = [self.position]
        save_interval = 10
        start_time = time.time()
        
        while self.step_count < max_steps:
            # 睡眠サイクル
            if self.step_count > 0 and self.step_count % self.sleep_interval == 0:
                self.sleep_cycle()
            
            # 進捗表示
            if self.step_count % 100 == 0:
                unique_count = len(self.unique_positions)
                distance_to_goal = abs(self.position[0] - self.maze_env.goal_pos[0]) + \
                                 abs(self.position[1] - self.maze_env.goal_pos[1])
                
                print(f"Step {self.step_count}: "
                      f"Pos {self.position}, "
                      f"Unique: {unique_count}, "
                      f"Episodes: {len(self.episodes)}, "
                      f"Goal dist: {distance_to_goal}")
            
            # 行動決定と実行
            action = self.decide_action()
            result = self.execute_action(action)
            
            all_positions.append(self.position)
            
            if self.step_count % save_interval == 0:
                path_history_sparse.append(self.position)
            
            # ゴール判定
            if self.position == self.maze_env.goal_pos:
                total_time = time.time() - start_time
                print(f"\n🎉 Goal reached in {self.step_count} steps!")
                print(f"Time: {total_time:.2f} seconds")
                print(f"Total sleep cycles: {len(self.sleep_history)}")
                path_history_sparse.append(self.position)
                break
        
        # 結果を保存
        self._save_results_with_sleep(path_history_sparse, all_positions, maze_array)
        self._visualize_sleep_effects(path_history_sparse, all_positions, maze_array)
        
        return {
            'success': self.position == self.maze_env.goal_pos,
            'steps': self.step_count,
            'unique_positions': len(self.unique_positions),
            'total_episodes': len(self.episodes),
            'sleep_cycles': len(self.sleep_history),
            'path_cells': path_cells,
            'efficiency': len(self.unique_positions) / self.step_count * 100
        }
    
    def _save_results_with_sleep(self, path_history_sparse, all_positions, maze_array):
        """睡眠統計を含む結果を保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        summary = {
            'timestamp': timestamp,
            'maze_size': self.maze_size,
            'algorithm': 'sleep_cycle_navigator',
            'success': self.position == self.maze_env.goal_pos,
            'total_steps': self.step_count,
            'unique_positions': len(self.unique_positions),
            'total_episodes': len(self.episodes),
            'sleep_cycles': len(self.sleep_history),
            'sleep_history': self.sleep_history,
            'efficiency': len(self.unique_positions) / self.step_count * 100
        }
        
        filename = f"results/sleep_cycle_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def _visualize_sleep_effects(self, path_history_sparse, all_positions, maze_array):
        """睡眠効果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 1. 迷路と経路
        ax1 = axes[0, 0]
        ax1.imshow(maze_array, cmap='binary', alpha=0.8)
        
        if path_history_sparse:
            path_array = np.array(path_history_sparse)
            ax1.plot(path_array[:, 0], path_array[:, 1], 
                   'b-', linewidth=1.5, alpha=0.7, label='Path')
        
        ax1.plot(0, 0, 'go', markersize=10, label='Start')
        gx, gy = self.maze_env.goal_pos
        ax1.plot(gx, gy, 'r*', markersize=15, label='Goal')
        ax1.set_title(f'Path with Sleep Cycles (Steps: {self.step_count})')
        ax1.legend()
        
        # 2. エピソード数の推移
        ax2 = axes[0, 1]
        
        if self.sleep_history:
            steps = [0] + [h['step'] for h in self.sleep_history]
            episodes = [100] + [h['final_episodes'] for h in self.sleep_history]  # 初期値は推定
            
            ax2.plot(steps, episodes, 'b-o', linewidth=2)
            ax2.fill_between(steps, episodes, alpha=0.3)
            
            # 睡眠ポイントをマーク
            for h in self.sleep_history:
                ax2.axvline(x=h['step'], color='red', alpha=0.3, linestyle='--')
        
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Number of Episodes')
        ax2.set_title('Episode Count with Sleep Cycles')
        ax2.grid(True, alpha=0.3)
        
        # 3. 接続密度の分布
        ax3 = axes[1, 0]
        
        if self.episode_nodes:
            densities = [len(node.connections) for node in self.episode_nodes]
            ax3.hist(densities, bins=20, alpha=0.7, color='green')
            ax3.axvline(x=self.density_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.density_threshold})')
            ax3.set_xlabel('Connection Count')
            ax3.set_ylabel('Number of Episodes')
            ax3.set_title('Connection Density Distribution (Final)')
            ax3.legend()
        
        # 4. 睡眠効果のサマリー
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if self.sleep_history:
            total_removed = sum(h['removed_episodes'] for h in self.sleep_history)
            total_pruned = sum(h['pruned_edges'] for h in self.sleep_history)
            
            summary_text = f"""Sleep Cycle Summary
            
Total Sleep Cycles: {len(self.sleep_history)}
Total Episodes Removed: {total_removed}
Total Edges Pruned: {total_pruned}

Average per Sleep:
- Episodes Removed: {total_removed/len(self.sleep_history):.1f}
- Edges Pruned: {total_pruned/len(self.sleep_history):.1f}

Final State:
- Episodes: {len(self.episodes)}
- Avg Connections: {np.mean([len(n.connections) for n in self.episode_nodes]):.1f}
"""
            ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/sleep_cycle_effects_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sleep effects visualization saved to: {filename}")


def visualize_graph_before_after_sleep(navigator: SleepCycleNavigator):
    """睡眠前後のグラフ構造を可視化"""
    # 簡易的なグラフ可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 睡眠前のサンプル（最初の100エピソード）
    sample_size = min(100, len(navigator.episode_nodes))
    
    # ノードの位置（エピソードの座標を使用）
    positions1 = []
    for i in range(sample_size):
        node = navigator.episode_nodes[i]
        if node.episode.x is not None and node.episode.y is not None:
            positions1.append((node.episode.x, node.episode.y))
        else:
            positions1.append((random.randint(0, 30), random.randint(0, 30)))
    
    # 接続を描画
    for i in range(sample_size):
        node = navigator.episode_nodes[i]
        x1, y1 = positions1[i]
        
        for conn_idx in node.connections:
            if conn_idx < sample_size:
                x2, y2 = positions1[conn_idx]
                ax1.plot([x1, x2], [y1, y2], 'b-', alpha=0.1, linewidth=0.5)
    
    # ノードを描画
    for i, (x, y) in enumerate(positions1):
        node = navigator.episode_nodes[i]
        color = 'red' if node.episode.goal_or_not else 'blue'
        size = min(len(node.connections) * 5, 100)
        ax1.scatter(x, y, c=color, s=size, alpha=0.7)
    
    ax1.set_title('Episode Graph (Sample)')
    ax1.set_xlim(-1, 31)
    ax1.set_ylim(-1, 31)
    
    # 統計情報
    ax2.axis('off')
    stats_text = f"""Graph Statistics

Total Episodes: {len(navigator.episode_nodes)}
Total Connections: {sum(len(n.connections) for n in navigator.episode_nodes)}

Connection Distribution:
- Max: {max(len(n.connections) for n in navigator.episode_nodes)}
- Avg: {np.mean([len(n.connections) for n in navigator.episode_nodes]):.1f}
- Min: {min(len(n.connections) for n in navigator.episode_nodes)}

Sleep Cycles: {len(navigator.sleep_history)}
"""
    ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/episode_graph_structure_{timestamp}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"Graph structure saved to: {filename}")


def main():
    """メイン実行"""
    print("="*60)
    print("Sleep Cycle Navigator Test")
    print("="*60)
    
    # 複数のシードでテスト
    results = []
    
    for seed in [42, 123, 456]:
        print(f"\n--- Testing seed {seed} ---")
        
        # シード設定
        random.seed(seed)
        np.random.seed(seed)
        
        # 睡眠サイクル付きナビゲーター
        navigator = SleepCycleNavigator(maze_size=30)
        result = navigator.solve_maze(max_steps=3000)
        
        results.append({
            'seed': seed,
            'success': result['success'],
            'steps': result['steps'],
            'episodes': result['total_episodes'],
            'sleep_cycles': result['sleep_cycles'],
            'efficiency': result['efficiency']
        })
        
        # グラフ構造を可視化（最初のシードのみ）
        if seed == 42:
            visualize_graph_before_after_sleep(navigator)
    
    # 結果サマリー
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Seed':<10} {'Success':<10} {'Steps':<10} {'Episodes':<12} {'Sleep':<10} {'Efficiency':<12}")
    print("-"*80)
    
    for r in results:
        success_str = "✓ Yes" if r['success'] else "✗ No"
        print(f"{r['seed']:<10} {success_str:<10} {r['steps']:<10} "
              f"{r['episodes']:<12} {r['sleep_cycles']:<10} {r['efficiency']:<12.1f}%")
    
    print("="*60)
    
    # 成功率
    success_count = sum(1 for r in results if r['success'])
    print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
    
    print("\n💤 Sleep Cycle Benefits:")
    print("- Prevents memory explosion")
    print("- Removes redundant connections")
    print("- Maintains important pathways (goal, recent)")
    print("- Improves search efficiency")


if __name__ == "__main__":
    main()