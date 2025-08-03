#!/usr/bin/env python3
"""修正版Frontier-based Exploration - 公正な比較"""

import sys
from pathlib import Path
import numpy as np
from typing import Set, Tuple, List, Optional
from collections import deque
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze


class ImprovedFrontierExplorer:
    """改良版Frontier探索 - より現実的な実装"""
    
    def __init__(self):
        self.explored = set()
        self.obstacles = set()
        self.position = None
        self.goal_found = None
        self.maze_bounds = None
        
    def update_bounds(self, position: Tuple[int, int]):
        """迷路の境界を推定"""
        if self.maze_bounds is None:
            self.maze_bounds = [position[0], position[0], position[1], position[1]]
        else:
            self.maze_bounds[0] = min(self.maze_bounds[0], position[0])
            self.maze_bounds[1] = max(self.maze_bounds[1], position[0])
            self.maze_bounds[2] = min(self.maze_bounds[2], position[1])
            self.maze_bounds[3] = max(self.maze_bounds[3], position[1])
    
    def get_unexplored_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """未探索の隣接セルを取得"""
        neighbors = []
        x, y = position
        
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            # 境界チェック
            if self.maze_bounds:
                if (neighbor[0] < self.maze_bounds[0] - 5 or 
                    neighbor[0] > self.maze_bounds[1] + 5 or
                    neighbor[1] < self.maze_bounds[2] - 5 or 
                    neighbor[1] > self.maze_bounds[3] + 5):
                    continue
            
            if neighbor not in self.explored and neighbor not in self.obstacles:
                neighbors.append(neighbor)
                
        return neighbors
    
    def find_nearest_unexplored(self) -> Optional[Tuple[int, int]]:
        """最も近い未探索セルを見つける（BFS）"""
        if not self.position:
            return None
            
        queue = deque([self.position])
        visited = {self.position}
        
        while queue:
            current = queue.popleft()
            
            # 未探索の隣接セルをチェック
            unexplored = self.get_unexplored_neighbors(current)
            if unexplored:
                # 最も近いものを返す
                return min(unexplored, 
                          key=lambda p: abs(p[0] - self.position[0]) + abs(p[1] - self.position[1]))
            
            # 探索を広げる
            x, y = current
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                if neighbor not in visited and neighbor not in self.obstacles:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return None
    
    def decide_action(self, obs, maze) -> int:
        """次の行動を決定"""
        self.position = obs.position
        self.explored.add(self.position)
        self.update_bounds(self.position)
        
        # ゴール発見
        if obs.is_goal:
            self.goal_found = self.position
            return 0  # 停止
        
        # 視覚情報から障害物を記録
        x, y = self.position
        for action in range(4):
            dx, dy = maze.ACTIONS[action]
            next_pos = (x + dx, y + dy)
            
            if action not in obs.possible_moves:
                self.obstacles.add(next_pos)
            else:
                # 移動可能な方向で未探索なら優先
                if next_pos not in self.explored:
                    return action
        
        # 全て探索済みの場合、最も近い未探索地点を目指す
        target = self.find_nearest_unexplored()
        if target:
            # 目標への方向を計算
            dx = target[0] - x
            dy = target[1] - y
            
            # 優先順位を決めて試行
            if abs(dx) > abs(dy):
                # 横方向優先
                if dx > 0 and 1 in obs.possible_moves:
                    return 1
                elif dx < 0 and 3 in obs.possible_moves:
                    return 3
            else:
                # 縦方向優先
                if dy > 0 and 2 in obs.possible_moves:
                    return 2
                elif dy < 0 and 0 in obs.possible_moves:
                    return 0
        
        # それでも動けない場合はランダム
        if obs.possible_moves:
            return np.random.choice(obs.possible_moves)
        return 0


def run_fair_comparison():
    """より公正な比較実験"""
    from insightspike.maze_experimental.navigators.experience_memory_navigator import ExperienceMemoryNavigator
    from insightspike.maze_experimental.maze_config import MazeNavigatorConfig
    
    print("公正な比較：Frontier探索 vs geDIG")
    print("=" * 60)
    print("同じ条件下での探索アルゴリズム比較")
    print("- 地図を作りながら探索")
    print("- 隣接セルのみ観測可能")
    print("- 動的な記憶構築")
    print("=" * 60)
    
    config = MazeNavigatorConfig(
        ged_weight=1.0,
        ig_weight=2.0,
        temperature=1.0,
        exploration_epsilon=0.0
    )
    
    # テスト設定
    maze_sizes = [(10, 10), (15, 15), (20, 20)]
    n_trials = 10
    
    for size in maze_sizes:
        print(f"\n{size[0]}x{size[1]} 迷路での比較（{n_trials}試行）:")
        print("-" * 40)
        
        results = {'frontier': [], 'gedig': []}
        
        for trial in range(n_trials):
            np.random.seed(trial)
            maze = SimpleMaze(size=size, maze_type='dfs')
            
            # Frontier探索
            explorer = ImprovedFrontierExplorer()
            obs = maze.reset()
            steps = 0
            
            for _ in range(size[0] * size[1] * 5):  # 十分なステップ数
                action = explorer.decide_action(obs, maze)
                obs, reward, done, info = maze.step(action)
                steps += 1
                if done and maze.agent_pos == maze.goal_pos:
                    break
                    
            results['frontier'].append({
                'steps': steps,
                'success': maze.agent_pos == maze.goal_pos,
                'explored': len(explorer.explored)
            })
            
            # geDIG
            navigator = ExperienceMemoryNavigator(config)
            obs = maze.reset()
            steps = 0
            
            for _ in range(size[0] * size[1] * 5):
                action = navigator.decide_action(obs, maze)
                obs, reward, done, info = maze.step(action)
                steps += 1
                if done and maze.agent_pos == maze.goal_pos:
                    break
                    
            results['gedig'].append({
                'steps': steps,
                'success': maze.agent_pos == maze.goal_pos,
                'explored': len(navigator.memory_nodes)
            })
        
        # 結果表示
        for method in ['frontier', 'gedig']:
            successes = [r for r in results[method] if r['success']]
            success_rate = len(successes) / n_trials
            
            if successes:
                avg_steps = np.mean([r['steps'] for r in successes])
                avg_explored = np.mean([r['explored'] for r in successes])
            else:
                avg_steps = float('inf')
                avg_explored = 0
                
            print(f"\n{method.upper()}:")
            print(f"  成功率: {success_rate:.0%}")
            print(f"  平均ステップ数: {avg_steps:.1f}")
            print(f"  平均探索セル数: {avg_explored:.1f}")
    
    print("\n" + "=" * 60)
    print("結論：")
    print("これが本当の比較！両方とも「地図を作りながら探索」している。")
    print("geDIGの特徴：情報利得に基づく効率的な探索戦略")
    print("=" * 60)


if __name__ == "__main__":
    run_fair_comparison()