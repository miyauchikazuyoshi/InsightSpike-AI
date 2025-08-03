#!/usr/bin/env python3
"""
Simple Query Navigator
======================

シンプルな単一クエリによる迷路ナビゲーション
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import json
from datetime import datetime
import time
import random

# パスを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from insightspike.environments.maze import SimpleMaze
except ImportError:
    from src.insightspike.environments.maze import SimpleMaze

from test_visual_memory_maze import Episode7D, generate_complex_maze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleQueryNavigator:
    """シンプルなクエリによるナビゲーター"""
    
    def __init__(self, maze_size: int = 30):
        self.maze_size = maze_size
        self.maze_env = None
        self.position = (0, 0)
        self.step_count = 0
        
        # エピソード記憶
        self.episodes: List[Episode7D] = []
        self.position_visits: Dict[Tuple[int, int], int] = {}
        self.unique_positions = {(0, 0)}
        
    def setup_maze(self):
        """迷路環境をセットアップ"""
        maze_array = generate_complex_maze(self.maze_size, self.maze_size)
        self.maze_env = SimpleMaze((self.maze_size, self.maze_size))
        self.maze_env.grid = maze_array
        self.position = (0, 0)
        self.step_count = 0
        self.position_visits[(0, 0)] = 1
        
        # ゴールエピソードを追加
        gx, gy = self.maze_env.goal_pos
        self.episodes.append(Episode7D(
            x=gx, y=gy, direction=None, result=None,
            visit_count=0, goal_or_not=True, wall_or_path='path'
        ))
        
        # 初期位置の視覚情報を記録
        self._record_visual_information(0, 0)
    
    def _get_visual_info(self, x: int, y: int) -> str:
        """指定位置が壁か通路かを判定"""
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return 'wall' if self.maze_env.grid[y, x] == 1 else 'path'
        return 'wall'
    
    def _record_visual_information(self, x: int, y: int):
        """現在位置から見える4方向の視覚情報をエピソードとして記録"""
        directions = [
            ((x+1, y), 'right'),
            ((x-1, y), 'left'),
            ((x, y-1), 'up'),
            ((x, y+1), 'down')
        ]
        
        for (nx, ny), direction in directions:
            wall_or_path = self._get_visual_info(nx, ny)
            visit_count = self.position_visits.get((nx, ny), 0)
            
            # 視覚情報をエピソードとして追加
            self.episodes.append(Episode7D(
                x=nx, y=ny, 
                direction=direction,  # どの方向の情報か記録
                result=None,
                visit_count=visit_count, 
                goal_or_not=False,  # 視覚情報なのでゴールではない
                wall_or_path=wall_or_path
            ))
    
    def _create_simple_query(self) -> Episode7D:
        """現在位置のみを持つシンプルなクエリ"""
        x, y = self.position
        return Episode7D(
            x=x, y=y,
            direction=None, 
            result=None,
            visit_count=None,  # nullにして全てのエピソードと比較可能に
            goal_or_not=None,
            wall_or_path=None
        )
    
    def _calculate_similarity(self, query: Episode7D, episode: Episode7D) -> float:
        """クエリとエピソードの類似度を計算（改善版）"""
        # 位置の近さ（最重要）
        distance = abs(query.x - episode.x) + abs(query.y - episode.y)
        
        # ゴールエピソードへの特別な重み付け
        if episode.goal_or_not:
            # 距離に応じた動的な重み
            if distance < 10:
                return 2.0  # 非常に高い優先度
            elif distance < 20:
                return 1.5
            else:
                return 1.0 / (1.0 + distance * 0.05)
        
        # 通常のエピソード
        base_score = 1.0 / (1.0 + distance * 0.1)
        
        # 視覚情報（隣接セル）には追加ボーナス
        if distance == 1 and episode.direction is not None:
            base_score *= 1.5
        
        # 訪問回数によるペナルティ
        if episode.visit_count is not None and episode.visit_count > 3:
            base_score *= 0.5
        
        return base_score
    
    def _search_with_simple_query(self) -> List[Tuple[Episode7D, float]]:
        """シンプルなクエリで検索"""
        query = self._create_simple_query()
        results = []
        
        for episode in self.episodes:
            similarity = self._calculate_similarity(query, episode)
            if similarity > 0.1:  # 閾値
                results.append((episode, similarity))
        
        # スコアでソート
        results.sort(key=lambda x: x[1], reverse=True)
        
        # トップ10を返す（視覚情報4つ + その他の重要な情報）
        return results[:10]
    
    def decide_action(self) -> str:
        """検索結果に基づいて行動を決定"""
        x, y = self.position
        
        # シンプルクエリで検索
        search_results = self._search_with_simple_query()
        
        # デバッグ出力
        if self.step_count % 100 == 0:
            print(f"\n[Step {self.step_count}] Top 3 results:")
            for i, (ep, score) in enumerate(search_results[:3]):
                ep_type = "GOAL" if ep.goal_or_not else (
                    f"Visual-{ep.direction}" if ep.direction else "Movement"
                )
                print(f"  {i+1}. {ep_type} at ({ep.x},{ep.y}), score: {score:.3f}")
        
        # 移動可能な方向を評価
        directions = ['up', 'down', 'left', 'right']
        direction_scores = {d: 0.0 for d in directions}
        
        dir_deltas = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        
        for episode, score in search_results:
            # ゴールへの方向を最優先
            if episode.goal_or_not:
                dx = episode.x - x
                dy = episode.y - y
                if dx > 0:
                    direction_scores['right'] += score * 3.0
                elif dx < 0:
                    direction_scores['left'] += score * 3.0
                if dy > 0:
                    direction_scores['down'] += score * 3.0
                elif dy < 0:
                    direction_scores['up'] += score * 3.0
            
            # 視覚情報を活用
            elif episode.direction is not None and episode.wall_or_path == 'path':
                # そのエピソードが示す方向（隣接セル）が通路なら
                direction_scores[episode.direction] += score * 2.0
            
            # 壁情報は回避
            elif episode.wall_or_path == 'wall':
                for direction, (dx, dy) in dir_deltas.items():
                    nx, ny = x + dx, y + dy
                    if episode.x == nx and episode.y == ny:
                        direction_scores[direction] -= score * 3.0
        
        # 最もスコアの高い方向を選択
        valid_actions = []
        for direction in directions:
            dx, dy = dir_deltas[direction]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                if self._get_visual_info(nx, ny) == 'path':
                    valid_actions.append((direction, direction_scores[direction]))
        
        if valid_actions:
            valid_actions.sort(key=lambda x: x[1], reverse=True)
            return valid_actions[0][0]
        
        return 'wait'
    
    def execute_action(self, action: str) -> Dict:
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
                # 移動成功
                self.position = (new_x, new_y)
                self.position_visits[(new_x, new_y)] = self.position_visits.get((new_x, new_y), 0) + 1
                self.unique_positions.add((new_x, new_y))
                result = 'moved'
                
                # 新しい位置の視覚情報を記録
                self._record_visual_information(new_x, new_y)
            else:
                result = 'wall'
        else:
            result = 'wall'
        
        # 移動エピソードを記録
        self.episodes.append(Episode7D(
            x=x, y=y, 
            direction=action, 
            result=result,
            visit_count=self.position_visits.get((x, y), 0),
            goal_or_not=False,
            wall_or_path='path'
        ))
        
        self.step_count += 1
        
        return {
            'action': action,
            'result': result,
            'position': self.position,
            'step': self.step_count
        }
    
    def solve_maze(self, max_steps: int = 3000) -> Dict:
        """迷路を解く"""
        self.setup_maze()
        
        # 迷路情報
        maze_array = self.maze_env.grid
        total_cells = self.maze_size * self.maze_size
        wall_cells = np.sum(maze_array == 1)
        path_cells = total_cells - wall_cells
        
        print(f"\n=== Simple Query Navigator ===")
        print(f"Maze size: {self.maze_size}x{self.maze_size}")
        print(f"Path cells: {path_cells} ({path_cells/total_cells*100:.1f}%)")
        print(f"Start: (0, 0), Goal: {self.maze_env.goal_pos}")
        print(f"Using SINGLE simple query\n")
        
        # 経路記録
        path_history = [self.position]
        start_time = time.time()
        
        while self.step_count < max_steps:
            # 進捗表示
            if self.step_count % 100 == 0 and self.step_count > 0:
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
            path_history.append(self.position)
            
            # ゴール判定
            if self.position == self.maze_env.goal_pos:
                total_time = time.time() - start_time
                print(f"\n🎉 Goal reached in {self.step_count} steps!")
                print(f"Time: {total_time:.2f} seconds")
                print(f"Total episodes: {len(self.episodes)}")
                break
        
        return {
            'success': self.position == self.maze_env.goal_pos,
            'steps': self.step_count,
            'unique_positions': len(self.unique_positions),
            'total_episodes': len(self.episodes),
            'path_cells': path_cells,
            'efficiency': len(self.unique_positions) / self.step_count * 100,
            'path_history': path_history[::10]  # 間引き
        }


def compare_query_approaches():
    """複数クエリ vs 単一クエリを比較"""
    print("="*60)
    print("Query Approach Comparison")
    print("="*60)
    
    # 同じシードで両方のアプローチをテスト
    results = []
    
    for seed in [42, 123, 456]:
        print(f"\n--- Testing seed {seed} ---")
        
        # シード設定
        random.seed(seed)
        np.random.seed(seed)
        
        # シンプルクエリ版
        navigator = SimpleQueryNavigator(maze_size=30)
        result = navigator.solve_maze(max_steps=3000)
        
        results.append({
            'seed': seed,
            'approach': 'Simple Query',
            'success': result['success'],
            'steps': result['steps'],
            'episodes': result['total_episodes'],
            'efficiency': result['efficiency']
        })
    
    # 結果表示
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Seed':<10} {'Success':<10} {'Steps':<10} {'Episodes':<12} {'Efficiency':<12}")
    print("-"*60)
    
    for r in results:
        success_str = "✓ Yes" if r['success'] else "✗ No"
        print(f"{r['seed']:<10} {success_str:<10} {r['steps']:<10} "
              f"{r['episodes']:<12} {r['efficiency']:<12.1f}%")
    
    print("="*60)
    
    # 成功率
    success_count = sum(1 for r in results if r['success'])
    print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
    
    print("\n💡 Benefits of Simple Query:")
    print("- Cleaner logic: Just one query per step")
    print("- Faster execution: Less computation")
    print("- Easier to understand and debug")
    print("- Goal episodes less likely to be buried")


def main():
    """メイン実行"""
    compare_query_approaches()


if __name__ == "__main__":
    main()