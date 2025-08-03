#!/usr/bin/env python3
"""Frontier-based Explorationの実装 - 本当の競合アルゴリズム"""

import sys
from pathlib import Path
import numpy as np
from typing import Set, Tuple, List, Optional
from collections import deque
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze


class FrontierExplorer:
    """Frontier-based Exploration - ロボティクスで標準的な探索手法"""
    
    def __init__(self):
        self.explored = set()  # 探索済みセル
        self.obstacles = set()  # 発見した障害物
        self.frontiers = set()  # フロンティア（既知と未知の境界）
        self.position = None
        self.goal_found = None
        
    def get_frontiers(self) -> Set[Tuple[int, int]]:
        """既知領域と未知領域の境界（フロンティア）を取得"""
        frontiers = set()
        
        for pos in self.explored:
            x, y = pos
            # 4方向の隣接セルをチェック
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                # 未探索かつ障害物でない隣接セルがフロンティア
                if neighbor not in self.explored and neighbor not in self.obstacles:
                    frontiers.add(neighbor)
                    
        return frontiers
    
    def find_nearest_frontier(self, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """最も近いフロンティアを見つける"""
        frontiers = self.get_frontiers()
        if not frontiers:
            return None
            
        min_dist = float('inf')
        nearest = None
        
        for frontier in frontiers:
            dist = abs(frontier[0] - position[0]) + abs(frontier[1] - position[1])
            if dist < min_dist:
                min_dist = dist
                nearest = frontier
                
        return nearest
    
    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[int]:
        """A*アルゴリズムで経路計画"""
        # 簡易版：BFSで実装
        if start == goal:
            return []
            
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            pos, path = queue.popleft()
            
            # 4方向を探索
            for action, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                next_pos = (pos[0] + dx, pos[1] + dy)
                
                if next_pos == goal:
                    return path + [action]
                    
                if next_pos not in visited and next_pos not in self.obstacles:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [action]))
                    
        return []  # 経路が見つからない
    
    def decide_action(self, obs, maze) -> int:
        """観測に基づいて次の行動を決定"""
        self.position = obs.position
        self.explored.add(self.position)
        
        # ゴールが見つかったら記録
        if obs.is_goal:
            self.goal_found = self.position
            
        # 視覚情報から障害物を記録
        x, y = self.position
        for action in range(4):
            if action not in obs.possible_moves:
                dx, dy = maze.ACTIONS[action]
                wall_pos = (x + dx, y + dy)
                self.obstacles.add(wall_pos)
        
        # ゴールが既知なら直接向かう
        if self.goal_found:
            path = self.plan_path(self.position, self.goal_found)
            if path:
                return path[0]
        
        # 最も近いフロンティアを探索
        nearest_frontier = self.find_nearest_frontier(self.position)
        if nearest_frontier:
            path = self.plan_path(self.position, nearest_frontier)
            if path:
                return path[0]
        
        # フロンティアがない場合はランダム
        if obs.possible_moves:
            return np.random.choice(obs.possible_moves)
        return 0


def compare_with_gedig():
    """geDIGとFrontier探索の公正な比較"""
    from insightspike.maze_experimental.navigators.experience_memory_navigator import ExperienceMemoryNavigator
    from insightspike.maze_experimental.maze_config import MazeNavigatorConfig
    
    print("Frontier-based Exploration vs geDIG")
    print("=" * 60)
    
    # 設定
    config = MazeNavigatorConfig(
        ged_weight=1.0,
        ig_weight=2.0,
        temperature=1.0,
        exploration_epsilon=0.0
    )
    
    # 複数の迷路でテスト
    n_trials = 20
    results = {'frontier': [], 'gedig': []}
    
    for trial in range(n_trials):
        np.random.seed(trial)
        maze = SimpleMaze(size=(15, 15), maze_type='dfs')
        
        # Frontier探索
        frontier_explorer = FrontierExplorer()
        obs = maze.reset()
        steps = 0
        start_time = time.time()
        
        for _ in range(1000):
            action = frontier_explorer.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            steps += 1
            if done and maze.agent_pos == maze.goal_pos:
                break
                
        frontier_time = time.time() - start_time
        results['frontier'].append({
            'steps': steps,
            'time': frontier_time,
            'success': maze.agent_pos == maze.goal_pos
        })
        
        # geDIG
        gedig_navigator = ExperienceMemoryNavigator(config)
        obs = maze.reset()
        steps = 0
        start_time = time.time()
        
        for _ in range(1000):
            action = gedig_navigator.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            steps += 1
            if done and maze.agent_pos == maze.goal_pos:
                break
                
        gedig_time = time.time() - start_time
        results['gedig'].append({
            'steps': steps,
            'time': gedig_time,
            'success': maze.agent_pos == maze.goal_pos
        })
    
    # 結果の集計
    print("\n結果（15x15迷路、20試行）：")
    print("-" * 40)
    
    for method in ['frontier', 'gedig']:
        successes = [r for r in results[method] if r['success']]
        if successes:
            avg_steps = np.mean([r['steps'] for r in successes])
            avg_time = np.mean([r['time'] for r in successes])
            success_rate = len(successes) / n_trials
        else:
            avg_steps = float('inf')
            avg_time = float('inf')
            success_rate = 0
            
        print(f"\n{method.upper()}:")
        print(f"  成功率: {success_rate:.1%}")
        print(f"  平均ステップ数: {avg_steps:.1f}")
        print(f"  平均計算時間: {avg_time*1000:.1f}ms")
    
    # 統計的比較
    if all(results['frontier']) and all(results['gedig']):
        frontier_steps = [r['steps'] for r in results['frontier'] if r['success']]
        gedig_steps = [r['steps'] for r in results['gedig'] if r['success']]
        
        if frontier_steps and gedig_steps:
            improvement = (np.mean(frontier_steps) - np.mean(gedig_steps)) / np.mean(frontier_steps) * 100
            print(f"\n改善率: {improvement:+.1f}%")
    
    print("\n" + "=" * 60)
    print("これが本当の「SOTA」比較！")
    print("同じ問題設定（地図を作りながら探索）での公正な比較")
    print("=" * 60)


if __name__ == "__main__":
    compare_with_gedig()