#!/usr/bin/env python3
"""
SLAM-based Maze Navigation Comparison
=====================================

様々なSLAMアルゴリズムと視覚記憶手法の比較
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from collections import deque, defaultdict
import random
from datetime import datetime
from abc import ABC, abstractmethod

# パスを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from test_visual_memory_maze import generate_complex_maze, VisualMemoryNavigator

try:
    from insightspike.environments.maze import SimpleMaze
except ImportError:
    from src.insightspike.environments.maze import SimpleMaze


class BaseSLAMNavigator(ABC):
    """SLAMナビゲーターの基底クラス"""
    
    def __init__(self, maze_size: int = 30):
        self.maze_size = maze_size
        self.position = (0, 0)
        self.step_count = 0
        self.visited_positions = set()
        self.map = np.ones((maze_size, maze_size)) * -1  # -1: unknown, 0: path, 1: wall
        
    @abstractmethod
    def decide_action(self) -> str:
        """行動決定（各アルゴリズムで実装）"""
        pass
    
    def get_visual_info(self, x: int, y: int) -> Optional[int]:
        """視覚情報を取得（実際の迷路から）"""
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.maze_env.grid[y, x]
        return 1  # 範囲外は壁
    
    def update_map(self):
        """現在位置から見える範囲の地図を更新"""
        x, y = self.position
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                self.map[ny, nx] = self.get_visual_info(nx, ny)
    
    def execute_action(self, action: str) -> bool:
        """行動を実行"""
        x, y = self.position
        dx, dy = 0, 0
        
        if action == 'up':
            dy = -1
        elif action == 'down':
            dy = 1
        elif action == 'left':
            dx = -1
        elif action == 'right':
            dx = 1
        
        new_x, new_y = x + dx, y + dy
        
        if 0 <= new_x < self.maze_size and 0 <= new_y < self.maze_size:
            if self.maze_env.grid[new_y, new_x] == 0:  # 通路
                self.position = (new_x, new_y)
                self.visited_positions.add(self.position)
                self.step_count += 1
                return True
        
        self.step_count += 1
        return False


class FrontierExplorer(BaseSLAMNavigator):
    """フロンティア探索法（Frontier-based Exploration）"""
    
    def __init__(self, maze_size: int = 30):
        super().__init__(maze_size)
        self.name = "Frontier Explorer"
        
    def find_frontiers(self) -> List[Tuple[int, int]]:
        """フロンティア（既知と未知の境界）を見つける"""
        frontiers = []
        
        for y in range(self.maze_size):
            for x in range(self.maze_size):
                if self.map[y, x] == 0:  # 既知の通路
                    # 隣接セルをチェック
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                            if self.map[ny, nx] == -1:  # 未知
                                frontiers.append((x, y))
                                break
        
        return frontiers
    
    def find_path_to_target(self, target: Tuple[int, int]) -> Optional[str]:
        """BFSで目標への経路を探索"""
        from collections import deque
        
        queue = deque([(self.position, [])])
        visited = {self.position}
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == target:
                return path[0] if path else None
            
            for action, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)), 
                                     ('left', (-1, 0)), ('right', (1, 0))]:
                nx, ny = x + dx, y + dy
                
                if (nx, ny) not in visited and 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                    if self.map[ny, nx] == 0:  # 既知の通路
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [action]))
        
        return None
    
    def decide_action(self) -> str:
        """最も近いフロンティアへ向かう"""
        self.update_map()
        
        # ゴールが既知なら優先
        goal = (self.maze_size - 1, self.maze_size - 1)
        if self.map[goal[1], goal[0]] == 0:
            action = self.find_path_to_target(goal)
            if action:
                return action
        
        # フロンティアを探す
        frontiers = self.find_frontiers()
        if not frontiers:
            # ランダムウォーク
            return random.choice(['up', 'down', 'left', 'right'])
        
        # 最も近いフロンティアを選択
        min_dist = float('inf')
        best_frontier = None
        
        for frontier in frontiers:
            dist = abs(frontier[0] - self.position[0]) + abs(frontier[1] - self.position[1])
            if dist < min_dist:
                min_dist = dist
                best_frontier = frontier
        
        if best_frontier:
            action = self.find_path_to_target(best_frontier)
            if action:
                return action
        
        return random.choice(['up', 'down', 'left', 'right'])


class WallFollower(BaseSLAMNavigator):
    """壁沿い探索法（Wall Following）"""
    
    def __init__(self, maze_size: int = 30):
        super().__init__(maze_size)
        self.name = "Wall Follower"
        self.direction = 'right'  # 現在の向き
        self.following_wall = 'right'  # 右手法
        
    def get_relative_directions(self) -> Dict[str, str]:
        """現在の向きに対する相対方向を取得"""
        directions = {
            'up': {'right': 'right', 'left': 'left', 'forward': 'up', 'back': 'down'},
            'down': {'right': 'left', 'left': 'right', 'forward': 'down', 'back': 'up'},
            'left': {'right': 'up', 'left': 'down', 'forward': 'left', 'back': 'right'},
            'right': {'right': 'down', 'left': 'up', 'forward': 'right', 'back': 'left'}
        }
        return directions[self.direction]
    
    def check_wall(self, direction: str) -> bool:
        """指定方向に壁があるかチェック"""
        x, y = self.position
        dx, dy = 0, 0
        
        if direction == 'up':
            dy = -1
        elif direction == 'down':
            dy = 1
        elif direction == 'left':
            dx = -1
        elif direction == 'right':
            dx = 1
        
        nx, ny = x + dx, y + dy
        
        if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
            return self.maze_env.grid[ny, nx] == 1
        return True  # 範囲外は壁
    
    def decide_action(self) -> str:
        """右手法で壁に沿って進む"""
        self.update_map()
        
        relative = self.get_relative_directions()
        
        # 1. 右に壁がなければ右に曲がる
        if not self.check_wall(relative['right']):
            self.direction = relative['right']
            return self.direction
        
        # 2. 前に進める場合は前進
        if not self.check_wall(relative['forward']):
            return self.direction
        
        # 3. 左に曲がる
        if not self.check_wall(relative['left']):
            self.direction = relative['left']
            return self.direction
        
        # 4. 後ろに戻る
        self.direction = relative['back']
        return self.direction


class PotentialFieldNavigator(BaseSLAMNavigator):
    """ポテンシャル場法（Potential Field Method）"""
    
    def __init__(self, maze_size: int = 30):
        super().__init__(maze_size)
        self.name = "Potential Field"
        self.goal = (maze_size - 1, maze_size - 1)
        
    def calculate_potential(self, x: int, y: int) -> float:
        """位置のポテンシャルを計算"""
        # ゴールへの引力
        goal_distance = np.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)
        attraction = -goal_distance * 10
        
        # 壁からの斥力
        repulsion = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                    if self.map[ny, nx] == 1:  # 壁
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist > 0:
                            repulsion += 50 / dist
        
        # 訪問済み位置への斥力
        for vx, vy in self.visited_positions:
            dist = np.sqrt((x - vx)**2 + (y - vy)**2)
            if dist < 3 and dist > 0:
                repulsion += 20 / dist
        
        return attraction + repulsion
    
    def decide_action(self) -> str:
        """最も低いポテンシャルの方向へ移動"""
        self.update_map()
        
        x, y = self.position
        best_action = None
        best_potential = float('inf')
        
        for action, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)), 
                                 ('left', (-1, 0)), ('right', (1, 0))]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                if self.map[ny, nx] != 1:  # 壁でない
                    potential = self.calculate_potential(nx, ny)
                    
                    if potential < best_potential:
                        best_potential = potential
                        best_action = action
        
        return best_action if best_action else random.choice(['up', 'down', 'left', 'right'])


def run_algorithm_comparison(maze_size: int = 30, max_steps: int = 3000):
    """各アルゴリズムを実行して比較"""
    
    # 迷路を生成
    random.seed(42)
    np.random.seed(42)
    maze_array = generate_complex_maze(maze_size, maze_size)
    maze_env = SimpleMaze((maze_size, maze_size))
    maze_env.grid = maze_array
    
    # アルゴリズムを準備
    algorithms = [
        FrontierExplorer(maze_size),
        WallFollower(maze_size),
        PotentialFieldNavigator(maze_size),
        VisualMemoryNavigator(maze_size)  # 我々の手法
    ]
    
    results = []
    
    for algo in algorithms:
        print(f"\n--- Running {algo.name if hasattr(algo, 'name') else 'Visual Memory'} ---")
        
        # 迷路環境を設定
        algo.maze_env = maze_env
        algo.position = (0, 0)
        algo.step_count = 0
        
        if isinstance(algo, VisualMemoryNavigator):
            # Visual Memory特別処理
            result = algo.solve_maze(max_steps)
            results.append({
                'name': 'Visual Memory (7D)',
                'success': result['success'],
                'steps': result['steps'],
                'unique_positions': result['unique_positions'],
                'efficiency': result['efficiency']
            })
        else:
            # SLAM系アルゴリズム
            algo.visited_positions = {(0, 0)}
            path_history = [(0, 0)]
            
            for step in range(max_steps):
                action = algo.decide_action()
                success = algo.execute_action(action)
                path_history.append(algo.position)
                
                if algo.position == (maze_size - 1, maze_size - 1):
                    print(f"Goal reached in {algo.step_count} steps!")
                    break
            
            results.append({
                'name': algo.name,
                'success': algo.position == (maze_size - 1, maze_size - 1),
                'steps': algo.step_count,
                'unique_positions': len(algo.visited_positions),
                'efficiency': len(algo.visited_positions) / algo.step_count * 100 if algo.step_count > 0 else 0,
                'path_history': path_history[::10]  # 間引き
            })
    
    return results, maze_array


def visualize_comparison_results(results: List[Dict], maze_array: np.ndarray):
    """比較結果を可視化"""
    
    n_algos = len(results)
    fig, axes = plt.subplots(2, n_algos, figsize=(5*n_algos, 10))
    
    # カラーパレット
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (result, color) in enumerate(zip(results, colors)):
        # 上段：メトリクス比較
        ax1 = axes[0, i] if n_algos > 1 else axes[0]
        
        metrics = ['Steps', 'Unique\nPositions', 'Efficiency\n(%)']
        values = [
            result['steps'],
            result['unique_positions'],
            result['efficiency']
        ]
        
        bars = ax1.bar(metrics, values, color=color, alpha=0.7)
        
        # 成功/失敗を色で表示
        title_color = 'green' if result['success'] else 'red'
        ax1.set_title(f"{result['name']}\n{'SUCCESS' if result['success'] else 'FAILED'}", 
                     color=title_color, fontweight='bold')
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}' if value > 1 else f'{value:.1f}',
                    ha='center', va='bottom')
        
        ax1.set_ylim(0, max(3500, max(values) * 1.2))
        
        # 下段：性能指標
        ax2 = axes[1, i] if n_algos > 1 else axes[1]
        
        # レーダーチャート風の評価
        categories = ['Speed\n(1/steps)', 'Coverage\n(%)', 'Efficiency\n(%)']
        
        # 正規化した値（0-100）
        speed_score = min(100, 3000 / result['steps'] * 100) if result['steps'] > 0 else 0
        coverage_score = min(100, result['unique_positions'] / (maze_array.shape[0] * maze_array.shape[1]) * 200)
        efficiency_score = min(100, result['efficiency'] * 2)
        
        scores = [speed_score, coverage_score, efficiency_score]
        
        # 棒グラフで表示
        bars2 = ax2.bar(categories, scores, color=color, alpha=0.7)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Score')
        
        # スコアを表示
        for bar, score in zip(bars2, scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.0f}',
                    ha='center', va='bottom')
        
        # 総合スコア
        total_score = np.mean(scores)
        ax2.text(0.5, 0.95, f'Total: {total_score:.0f}', 
                transform=ax2.transAxes, ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    plt.suptitle('SLAM Algorithm Comparison on Maze Navigation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/slam_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison visualization saved to: {filename}")
    return filename


def create_summary_table(results: List[Dict]):
    """結果のサマリーテーブルを作成"""
    
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<20} {'Success':<10} {'Steps':<10} {'Unique Pos':<12} {'Efficiency':<12}")
    print("-"*80)
    
    for result in results:
        success_str = "✓ Yes" if result['success'] else "✗ No"
        print(f"{result['name']:<20} {success_str:<10} {result['steps']:<10} "
              f"{result['unique_positions']:<12} {result['efficiency']:<12.1f}%")
    
    print("="*80)
    
    # 勝者を決定
    successful_algos = [r for r in results if r['success']]
    if successful_algos:
        winner = min(successful_algos, key=lambda x: x['steps'])
        print(f"\n🏆 Winner: {winner['name']} (solved in {winner['steps']} steps)")
    else:
        print("\n❌ No algorithm successfully solved the maze")


def main():
    """メイン実行"""
    print("="*60)
    print("SLAM Algorithm Comparison")
    print("Frontier vs Wall-Following vs Potential Field vs Visual Memory")
    print("="*60)
    
    # 比較実験を実行
    results, maze_array = run_algorithm_comparison(maze_size=30, max_steps=3000)
    
    # 結果を可視化
    visualize_comparison_results(results, maze_array)
    
    # サマリーテーブル
    create_summary_table(results)
    
    # 詳細分析
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    print("\n1. Frontier Explorer:")
    print("   - 未知領域の境界を優先的に探索")
    print("   - 効率的だが、実装が複雑")
    
    print("\n2. Wall Follower:")
    print("   - シンプルで確実だが、効率が悪い")
    print("   - 単純連結迷路では必ず解ける")
    
    print("\n3. Potential Field:")
    print("   - ゴールへの引力と壁からの斥力")
    print("   - 局所最適に陥りやすい")
    
    print("\n4. Visual Memory (7D):")
    print("   - エピソード記憶と類似度計算")
    print("   - 人間的な認知プロセス")
    
    print("="*60)


if __name__ == "__main__":
    main()