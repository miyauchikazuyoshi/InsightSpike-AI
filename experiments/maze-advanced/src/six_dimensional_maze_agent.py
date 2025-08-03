#!/usr/bin/env python3
"""
Six-Dimensional Maze Agent
==========================

6次元ベクトル: [X, Y, action, result, visits, goal_info]
- goal_info: ゴール=1.0, 通常=0.0, 未知=-1.0
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.config.presets import ConfigPresets
from insightspike.environments.maze import SimpleMaze
from donut_search_maze import DonutSearchMaze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MazeState:
    """迷路の現在状態"""
    position: Tuple[int, int]
    visited_positions: List[Tuple[int, int]]
    discovered_cells: Dict[Tuple[int, int], float]  # 位置 -> goal_info
    last_action: Optional[int] = None
    last_result: Optional[str] = None
    step_count: int = 0


class SixDimensionalVectorSpace:
    """6次元ベクトル空間での処理"""
    
    def __init__(self, maze_size: Tuple[int, int]):
        self.maze_width, self.maze_height = maze_size
        
    def state_to_vector(self, 
                       position: Tuple[int, int], 
                       action: Optional[int] = None,
                       result: Optional[str] = None,
                       visits: int = 0,
                       goal_info: float = -1.0) -> np.ndarray:
        """位置と状態を6次元ベクトルに変換"""
        # 位置を正規化
        norm_x = position[0] / (self.maze_width - 1) if self.maze_width > 1 else 0
        norm_y = position[1] / (self.maze_height - 1) if self.maze_height > 1 else 0
        
        # 行動を正規化
        if action is not None:
            norm_action = action * 0.25  # 0, 0.25, 0.5, 0.75
        else:
            norm_action = 0.5
        
        # 結果をエンコード
        result_map = {'wall': -1.0, 'empty': 0.0, 'goal': 1.0, None: 0.0}
        norm_result = result_map.get(result, 0.0)
        
        # 訪問回数を正規化
        norm_visits = min(visits / 10.0, 1.0)
        
        # 6次元ベクトル: [X, Y, action, result, visits, goal_info]
        return np.array([norm_x, norm_y, norm_action, norm_result, norm_visits, goal_info])


class SixDimensionalMazeAgent:
    """6次元ベクトルを使う探索エージェント"""
    
    def __init__(self, config=None):
        self.config = config or ConfigPresets.experiment()
        self.donut_search = DonutSearchMaze(dimension=6)  # 6次元に拡張
        self.vector_space = None
        self.maze_env = None
        self.current_state = None
        self.goal_discovered = False
        self.goal_position = None
        
    def setup_maze(self, maze_query: Dict):
        """迷路環境をセットアップ"""
        maze_array = np.array(maze_query['maze'])
        
        self.maze_env = SimpleMaze(
            size=maze_array.shape[::-1],
            maze_type='custom',
            maze_layout=maze_array,
            start_pos=maze_query['start'],
            goal_pos=maze_query['goal']
        )
        
        self.vector_space = SixDimensionalVectorSpace(self.maze_env.size)
        
        self.current_state = MazeState(
            position=self.maze_env.start_pos,
            visited_positions=[self.maze_env.start_pos],
            discovered_cells={}  # 最初は何も知らない
        )
        
        # 初期エピソードを追加（ゴール情報なし）
        self._add_initial_episodes()
        
    def _add_initial_episodes(self):
        """初期エピソード（スタート地点の4方向）を追加"""
        start_x, start_y = self.maze_env.start_pos
        
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        action_names = ['up', 'right', 'down', 'left']
        
        for action, ((dx, dy), name) in enumerate(zip(directions, action_names)):
            nx, ny = start_x + dx, start_y + dy
            
            # 結果を判定（ただしゴールかどうかは知らない）
            if (0 <= nx < self.maze_env.size[0] and 
                0 <= ny < self.maze_env.size[1]):
                if self.maze_env.grid[ny, nx] == 0 or (nx, ny) == self.maze_env.goal_pos:
                    result = 'empty'  # ゴールも最初は'empty'として扱う
                else:
                    result = 'wall'
            else:
                result = 'wall'
            
            # エピソードベクトル作成（goal_info=-1.0: 未知）
            episode_vector = self.vector_space.state_to_vector(
                position=(start_x, start_y),
                action=action,
                result=result,
                visits=1,
                goal_info=-1.0  # 未知
            )
            
            episode_id = f"INITIAL_{name.upper()}"
            self.donut_search.add_episode(episode_id, episode_vector, (start_x, start_y))
            
            logger.info(f"Added initial episode: {name} from start -> {result} (goal_info=-1.0)")
        
    def process_state(self, state: MazeState) -> int:
        """6次元ベクトルを活用して方向を決定"""
        # 現在位置のゴール情報を取得
        current_goal_info = state.discovered_cells.get(state.position, -1.0)
        
        # 現在状態のベクトル
        current_vector = self.vector_space.state_to_vector(
            position=state.position,
            action=state.last_action,
            result=state.last_result,
            visits=state.visited_positions.count(state.position),
            goal_info=current_goal_info
        )
        
        # 現在のエピソードを記録
        episode_id = f"step_{state.step_count}"
        self.donut_search.add_episode(episode_id, current_vector, state.position)
        
        # ドーナツ検索
        result = self.donut_search.donut_search(
            current_vector,
            inner_radius=0.1,
            outer_radius=1.5
        )
        
        logger.info(f"Donut search: inner={len(result.inner_nodes)}, " +
                   f"candidates={len(result.candidates)}, " +
                   f"outer={len(result.outer_nodes)}")
        
        # 可能な行動を取得
        possible_actions = self._get_possible_actions()
        if not possible_actions:
            return 0
            
        # 各行動のスコアを計算
        action_scores = {}
        
        for action in possible_actions:
            # この行動を取った場合の予測位置
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            next_pos = (state.position[0] + dx, state.position[1] + dy)
            
            # 次の位置のゴール情報（既知なら使う、未知なら-1.0）
            next_goal_info = state.discovered_cells.get(next_pos, -1.0)
            
            # 予測ベクトル
            predicted_vector = self.vector_space.state_to_vector(
                position=next_pos,
                action=action,
                result='empty',  # 仮定
                visits=state.visited_positions.count(next_pos),
                goal_info=next_goal_info
            )
            
            # スコア計算
            score = 0.0
            
            # 1. ゴール情報に基づくスコア
            if next_goal_info == 1.0:  # 既知のゴール
                score += 100.0
            elif next_goal_info == 0.0:  # 既知の通常マス
                score += 1.0
            else:  # 未知のマス
                score += 10.0  # 探索ボーナス
            
            # 2. 訪問回数ペナルティ
            visit_penalty = state.visited_positions.count(next_pos) * 5.0
            score -= visit_penalty
            
            # 3. ゴール発見後の戦略
            if self.goal_discovered and self.goal_position:
                # ゴール位置へのマンハッタン距離
                goal_x, goal_y = self.goal_position
                distance = abs(next_pos[0] - goal_x) + abs(next_pos[1] - goal_y)
                score -= distance * 2.0  # 距離が近いほど高スコア
            
            action_scores[action] = score
            
        # 最高スコアの行動を選択
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        
        action_names = ['↑', '→', '↓', '←']
        logger.info(f"Action scores: {[(action_names[a], round(s, 2)) for a, s in action_scores.items()]}")
        logger.info(f"Selected action: {best_action} ({action_names[best_action]})")
        logger.info(f"Current goal_info: {current_goal_info}, Discovered cells: {len(state.discovered_cells)}")
        
        return best_action
        
    def _get_possible_actions(self) -> List[int]:
        """可能な行動のリスト"""
        actions = []
        x, y = self.current_state.position
        
        for action, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.maze_env.size[0] and 
                0 <= ny < self.maze_env.size[1] and
                (self.maze_env.grid[ny, nx] == 0 or (nx, ny) == self.maze_env.goal_pos)):
                actions.append(action)
                
        return actions
        
    def execute_action(self, action: int) -> str:
        """行動を実行"""
        old_pos = self.current_state.position
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
        
        if (0 <= new_x < self.maze_env.size[0] and 
            0 <= new_y < self.maze_env.size[1]):
            
            if self.maze_env.grid[new_y, new_x] == 0 or (new_x, new_y) == self.maze_env.goal_pos:
                # 移動成功
                self.current_state.position = (new_x, new_y)
                
                # ゴール判定と情報更新
                if (new_x, new_y) == self.maze_env.goal_pos:
                    result = 'goal'
                    self.current_state.discovered_cells[(new_x, new_y)] = 1.0  # ゴール
                    if not self.goal_discovered:
                        self.goal_discovered = True
                        self.goal_position = (new_x, new_y)
                        logger.info(f"🎯 Goal discovered at {self.goal_position}!")
                else:
                    result = 'empty'
                    self.current_state.discovered_cells[(new_x, new_y)] = 0.0  # 通常マス
            else:
                result = 'wall'
        else:
            result = 'wall'
            
        self.current_state.last_action = action
        self.current_state.last_result = result
        # 移動に成功した場合のみ訪問位置に追加
        if result != 'wall':
            self.current_state.visited_positions.append(self.current_state.position)
        self.current_state.step_count += 1
        
        return result
        
    def solve_maze(self, maze_query: Dict, max_steps: int = 50) -> Dict:
        """迷路を解く"""
        self.setup_maze(maze_query)
        
        print(f"\n=== 6-Dimensional Maze Solving ===")
        print(f"Start: {self.maze_env.start_pos}, Goal: Unknown initially")
        print(f"6D Vector: [X, Y, action, result, visits, goal_info]\n")
        
        path_for_viz = []
        
        while self.current_state.step_count < max_steps:
            print(f"Step {self.current_state.step_count}: Position {self.current_state.position}", end=" ")
            
            # 行動決定
            action = self.process_state(self.current_state)
            
            # 行動実行
            result = self.execute_action(action)
            actions = ['↑', '→', '↓', '←']
            print(f"→ {actions[action]} → {result}")
            
            path_for_viz.append(self.current_state.position)
            
            # ゴール判定
            if self.current_state.position == self.maze_env.goal_pos:
                print(f"\n🎉 Goal reached in {self.current_state.step_count} steps!")
                break
                
        # 簡単な可視化
        self._visualize_result(path_for_viz)
        
        return {
            'success': self.current_state.position == self.maze_env.goal_pos,
            'steps': self.current_state.step_count,
            'path': self.current_state.visited_positions,
            'unique_positions': len(set(self.current_state.visited_positions)),
            'discovered_cells': len(self.current_state.discovered_cells),
            'goal_discovered_at_step': next((i for i, pos in enumerate(self.current_state.visited_positions) 
                                           if pos == self.goal_position), None) if self.goal_discovered else None
        }
        
    def _visualize_result(self, path: List[Tuple[int, int]]):
        """結果を可視化（発見情報を含む）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左: 迷路と経路
        maze_display = self.maze_env.grid.copy().astype(float)
        ax1.imshow(maze_display, cmap='binary', alpha=0.3)
        
        # 経路を描画
        if path:
            path_array = np.array(path)
            ax1.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=2, alpha=0.7)
            
            # 訪問順を番号で表示
            for i, pos in enumerate(path[:20]):
                ax1.text(pos[0], pos[1], str(i), fontsize=8, ha='center', va='center')
        
        # スタートとゴール
        ax1.plot(*self.maze_env.start_pos, 'go', markersize=15, label='Start')
        ax1.plot(*self.maze_env.goal_pos, 'r*', markersize=20, label='Goal')
        
        ax1.set_title(f'Path (Steps: {self.current_state.step_count})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右: 発見情報マップ
        discovery_map = np.full(self.maze_env.grid.shape, -1.0)  # 未知=-1
        
        for pos, goal_info in self.current_state.discovered_cells.items():
            x, y = pos
            discovery_map[y, x] = goal_info
            
        im = ax2.imshow(discovery_map, cmap='RdYlGn', vmin=-1, vmax=1, alpha=0.8)
        ax2.set_title('Discovery Map (Red=Unknown, Yellow=Normal, Green=Goal)')
        
        # カラーバーを追加
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Goal Info')
        
        plt.tight_layout()
        plt.savefig('results/six_dimensional_solution.png')
        plt.close()
        print("\nVisualization saved to results/six_dimensional_solution.png")


def main():
    """メイン実行関数"""
    # 3x3の簡単な迷路
    test_maze_3x3 = {
        "maze": [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        "start": (0, 0),
        "goal": (2, 2)
    }
    
    # 5x5の迷路
    test_maze_5x5 = {
        "maze": [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        "start": (0, 0),
        "goal": (4, 4)
    }
    
    # 3x3でテスト
    print("Testing with 3x3 maze...")
    agent = SixDimensionalMazeAgent()
    result = agent.solve_maze(test_maze_3x3)
    
    print(f"\n=== Result ===")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Discovered cells: {result['discovered_cells']}")
    print(f"Goal discovered at step: {result['goal_discovered_at_step']}")
    print(f"Efficiency: {result['unique_positions'] / result['steps']:.2%}")
    
    # 成功したら5x5でもテスト
    if result['success']:
        print("\n\nTesting with 5x5 maze...")
        agent2 = SixDimensionalMazeAgent()
        result2 = agent2.solve_maze(test_maze_5x5)
        
        print(f"\n=== Result ===")
        print(f"Success: {result2['success']}")
        print(f"Steps: {result2['steps']}")
        print(f"Discovered cells: {result2['discovered_cells']}")
        print(f"Goal discovered at step: {result2['goal_discovered_at_step']}")
        print(f"Efficiency: {result2['unique_positions'] / result2['steps']:.2%}")


if __name__ == "__main__":
    main()