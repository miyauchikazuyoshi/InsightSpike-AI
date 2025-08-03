#!/usr/bin/env python3
"""
Autonomous Maze Agent V5 with Initial Episodes
=============================================

初期エピソード（スタート地点の4方向＋ゴール）を持つ自律エージェント
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

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.config.presets import ConfigPresets
from insightspike.core.episode import Episode
from insightspike.environments.maze import SimpleMaze
from donut_search_maze import DonutSearchMaze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MazeState:
    """迷路の現在状態"""
    position: Tuple[int, int]
    visited_positions: List[Tuple[int, int]]
    last_action: Optional[int] = None
    last_result: Optional[str] = None
    step_count: int = 0


class CompactVectorSpace:
    """5次元ベクトル空間での処理"""
    
    def __init__(self, maze_size: Tuple[int, int]):
        self.maze_width, self.maze_height = maze_size
        
    def state_to_vector(self, position: Tuple[int, int], 
                       action: Optional[int] = None,
                       result: Optional[str] = None,
                       visits: int = 0) -> np.ndarray:
        """位置と状態を5次元ベクトルに変換"""
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
        
        return np.array([norm_x, norm_y, norm_action, norm_result, norm_visits])


class AutonomousMazeAgentV5:
    """初期エピソードを持つ自律エージェント"""
    
    def __init__(self, config=None):
        self.config = config or ConfigPresets.experiment()
        self.donut_search = DonutSearchMaze(dimension=5)
        self.vector_space = None
        self.maze_env = None
        self.current_state = None
        self.goal_episode_id = "GOAL_EPISODE"  # 特別なID
        self.initial_episodes = {}  # 初期エピソードを保存
        
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
        
        self.vector_space = CompactVectorSpace(self.maze_env.size)
        
        self.current_state = MazeState(
            position=self.maze_env.start_pos,
            visited_positions=[self.maze_env.start_pos]
        )
        
        # 初期エピソードを追加
        self._add_initial_episodes()
        
    def _add_initial_episodes(self):
        """初期エピソード（スタート地点の4方向＋ゴール）を追加"""
        start_x, start_y = self.maze_env.start_pos
        goal_x, goal_y = self.maze_env.goal_pos
        
        # 1. ゴールエピソード
        goal_vector = self.vector_space.state_to_vector(
            position=(goal_x, goal_y),
            action=None,
            result='goal',
            visits=1
        )
        self.donut_search.add_episode(self.goal_episode_id, goal_vector, (goal_x, goal_y))
        self.initial_episodes[self.goal_episode_id] = goal_vector
        logger.info(f"Added goal episode at {self.maze_env.goal_pos}")
        
        # 2. スタート地点からの4方向エピソード
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        action_names = ['up', 'right', 'down', 'left']
        
        for action, ((dx, dy), name) in enumerate(zip(directions, action_names)):
            nx, ny = start_x + dx, start_y + dy
            
            # 結果を判定
            if (0 <= nx < self.maze_env.size[0] and 
                0 <= ny < self.maze_env.size[1]):
                # SimpleMazeのゴール位置は特別な値を持つ可能性があるので注意
                if (nx, ny) == self.maze_env.goal_pos:
                    result = 'goal'
                elif self._is_passable(nx, ny):
                    result = 'empty'
                else:
                    result = 'wall'
            else:
                result = 'wall'
            
            # エピソードベクトル作成
            episode_vector = self.vector_space.state_to_vector(
                position=(start_x, start_y),
                action=action,
                result=result,
                visits=1
            )
            
            episode_id = f"INITIAL_{name.upper()}"
            self.donut_search.add_episode(episode_id, episode_vector, (start_x, start_y))
            self.initial_episodes[episode_id] = episode_vector
            
            logger.info(f"Added initial episode: {name} from start -> {result}")
        
        logger.info(f"Total initial episodes: {len(self.initial_episodes)}")
        
    def _is_passable(self, x: int, y: int) -> bool:
        """指定位置が通行可能かチェック（ゴール位置も通行可能とする）"""
        if (x, y) == self.maze_env.goal_pos:
            return True
        return self.maze_env.grid[y, x] == 0
        
    def process_state(self, state: MazeState) -> int:
        """ドーナツ検索と初期エピソードを活用して方向を決定"""
        # 現在状態のベクトル
        current_vector = self.vector_space.state_to_vector(
            position=state.position,
            action=state.last_action,
            result=state.last_result,
            visits=state.visited_positions.count(state.position)
        )
        
        # 現在のエピソードを記録
        episode_id = f"step_{state.step_count}"
        self.donut_search.add_episode(episode_id, current_vector, state.position)
        
        # ゴールエピソードまでの距離を計算
        goal_vector = self.donut_search.episode_vectors[self.goal_episode_id]
        goal_distance = np.linalg.norm(current_vector - goal_vector)
        logger.info(f"Distance to goal episode: {goal_distance:.3f}")
        
        # ドーナツ検索（ゴールに近い領域を探索）
        result = self.donut_search.donut_search(
            current_vector,
            inner_radius=0.1,  # 既に探索した近い領域
            outer_radius=goal_distance + 0.2  # ゴールまでの距離より少し遠く
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
            
            # 予測ベクトル
            predicted_vector = self.vector_space.state_to_vector(
                position=next_pos,
                action=action,
                result='empty',  # 仮定
                visits=state.visited_positions.count(next_pos)
            )
            
            # ゴールエピソードへの距離
            predicted_distance = np.linalg.norm(predicted_vector - goal_vector)
            
            # スコア（距離が小さいほど良い）
            score = -predicted_distance
            
            # 訪問回数によるペナルティ
            visit_penalty = state.visited_positions.count(next_pos) * 0.1
            score -= visit_penalty
            
            # 初期エピソードとの類似性ボーナス
            for ep_id, ep_vec in self.initial_episodes.items():
                if 'INITIAL_' in ep_id:  # 初期方向エピソードの場合
                    similarity = 1.0 - np.linalg.norm(predicted_vector - ep_vec)
                    if similarity > 0.7:  # 高い類似性がある場合
                        score += similarity * 0.2
            
            action_scores[action] = score
            
        # 最高スコアの行動を選択
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        
        logger.info(f"Action scores: {action_scores}")
        logger.info(f"Selected action: {best_action}")
        
        return best_action
        
    def _get_possible_actions(self) -> List[int]:
        """可能な行動のリスト（ゴール位置も通行可能とする）"""
        actions = []
        x, y = self.current_state.position
        
        for action, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.maze_env.size[0] and 
                0 <= ny < self.maze_env.size[1] and
                self._is_passable(nx, ny)):
                actions.append(action)
                
        return actions
        
    def execute_action(self, action: int) -> str:
        """行動を実行"""
        old_pos = self.current_state.position
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
        
        if (0 <= new_x < self.maze_env.size[0] and 
            0 <= new_y < self.maze_env.size[1]):
            
            if self._is_passable(new_x, new_y):
                # 移動成功
                self.current_state.position = (new_x, new_y)
                if (new_x, new_y) == self.maze_env.goal_pos:
                    result = 'goal'
                else:
                    result = 'empty'
            else:
                result = 'wall'
        else:
            result = 'wall'
            
        self.current_state.last_action = action
        self.current_state.last_result = result
        self.current_state.visited_positions.append(self.current_state.position)
        self.current_state.step_count += 1
        
        return result
        
    def solve_maze(self, maze_query: Dict, max_steps: int = 50) -> Dict:
        """迷路を解く"""
        self.setup_maze(maze_query)
        
        print(f"\n=== Autonomous Maze Solving with Initial Episodes ===")
        print(f"Start: {self.maze_env.start_pos}, Goal: {self.maze_env.goal_pos}")
        print(f"Initial episodes: {len(self.initial_episodes)}")
        print(f"  - Goal episode")
        print(f"  - 4 directional episodes from start\n")
        
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
            'unique_positions': len(set(self.current_state.visited_positions))
        }
        
    def _visualize_result(self, path: List[Tuple[int, int]]):
        """結果を可視化"""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # 迷路を描画（ゴール位置を特別扱い）
        maze_display = self.maze_env.grid.copy()
        gx, gy = self.maze_env.goal_pos
        if maze_display[gy, gx] != 0:
            maze_display[gy, gx] = 0  # ゴール位置を通行可能として表示
        
        ax.imshow(maze_display, cmap='binary')
        
        # 経路を描画
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=2, alpha=0.7)
            
            # 訪問順を番号で表示
            for i, pos in enumerate(path[:20]):  # 最初の20ステップまで
                ax.text(pos[0], pos[1], str(i), fontsize=8, ha='center', va='center')
        
        # スタートとゴール
        ax.plot(*self.maze_env.start_pos, 'go', markersize=15, label='Start')
        ax.plot(*self.maze_env.goal_pos, 'r*', markersize=20, label='Goal')
        
        ax.set_title(f'Maze Solution (Steps: {self.current_state.step_count})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig('results/initial_episodes_solution.png')
        plt.close()
        print("\nVisualization saved to results/initial_episodes_solution.png")


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
    agent = AutonomousMazeAgentV5()
    result = agent.solve_maze(test_maze_3x3)
    
    print(f"\n=== Result ===")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Efficiency: {result['unique_positions'] / result['steps']:.2%}")
    
    # 成功したら5x5でもテスト
    if result['success']:
        print("\n\nTesting with 5x5 maze...")
        agent2 = AutonomousMazeAgentV5()
        result2 = agent2.solve_maze(test_maze_5x5)
        
        print(f"\n=== Result ===")
        print(f"Success: {result2['success']}")
        print(f"Steps: {result2['steps']}")
        print(f"Efficiency: {result2['unique_positions'] / result2['steps']:.2%}")


if __name__ == "__main__":
    main()