#!/usr/bin/env python3
"""
Post-Action Query Agent
=======================

行動後の状態でクエリベクトルを生成するエージェント
6次元: [X, Y, null(action), null(result), visits, goal_info]
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


class PostActionVectorSpace:
    """行動後クエリベクトル空間"""
    
    def __init__(self, maze_size: Tuple[int, int]):
        self.maze_width, self.maze_height = maze_size
        
    def create_query_vector(self, 
                          position: Tuple[int, int], 
                          visits: int = 0,
                          goal_info: float = -1.0) -> np.ndarray:
        """
        行動後の状態からクエリベクトルを生成
        action=null, result=nullの状態
        """
        # 位置を正規化
        norm_x = position[0] / (self.maze_width - 1) if self.maze_width > 1 else 0
        norm_y = position[1] / (self.maze_height - 1) if self.maze_height > 1 else 0
        
        # action, resultはnull状態（0.5で表現）
        norm_action = 0.5
        norm_result = 0.5
        
        # 訪問回数を正規化
        norm_visits = min(visits / 10.0, 1.0)
        
        # 6次元ベクトル: [X, Y, action(null), result(null), visits, goal_info]
        return np.array([norm_x, norm_y, norm_action, norm_result, norm_visits, goal_info])
    
    def create_response_vector(self,
                             position: Tuple[int, int],
                             action: int,
                             result: str,
                             visits: int = 0,
                             goal_info: float = -1.0) -> np.ndarray:
        """
        行動実行後のレスポンスベクトルを生成
        """
        # 位置を正規化
        norm_x = position[0] / (self.maze_width - 1) if self.maze_width > 1 else 0
        norm_y = position[1] / (self.maze_height - 1) if self.maze_height > 1 else 0
        
        # 行動を正規化
        norm_action = action * 0.25  # 0, 0.25, 0.5, 0.75
        
        # 結果をエンコード
        result_map = {'wall': -1.0, 'empty': 0.0, 'goal': 1.0}
        norm_result = result_map.get(result, 0.0)
        
        # 訪問回数を正規化
        norm_visits = min(visits / 10.0, 1.0)
        
        return np.array([norm_x, norm_y, norm_action, norm_result, norm_visits, goal_info])


class PostActionQueryAgent:
    """行動後クエリを使うエージェント"""
    
    def __init__(self, config=None):
        self.config = config or ConfigPresets.experiment()
        self.donut_search = DonutSearchMaze(dimension=6)
        self.vector_space = None
        self.maze_env = None
        self.current_state = None
        self.goal_discovered = False
        self.goal_position = None
        
        # エピソード記憶
        self.query_response_pairs = []  # (query_vector, action, response_vector)のリスト
        
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
        
        self.vector_space = PostActionVectorSpace(self.maze_env.size)
        
        self.current_state = MazeState(
            position=self.maze_env.start_pos,
            visited_positions=[self.maze_env.start_pos],
            discovered_cells={self.maze_env.start_pos: 0.0}  # スタート地点は通常マス
        )
        
    def decide_action(self, state: MazeState) -> int:
        """現在状態からクエリベクトルを生成して行動を決定"""
        
        # 現在位置のゴール情報
        current_goal_info = state.discovered_cells.get(state.position, -1.0)
        
        # クエリベクトル生成（行動後の状態）
        query_vector = self.vector_space.create_query_vector(
            position=state.position,
            visits=state.visited_positions.count(state.position),
            goal_info=current_goal_info
        )
        
        logger.info(f"Query vector at {state.position}: {query_vector}")
        
        # ドーナツ検索で類似エピソードを探す
        if len(self.query_response_pairs) > 0:
            # エピソードをドーナツ検索に追加
            for i, (q_vec, action, r_vec) in enumerate(self.query_response_pairs):
                self.donut_search.add_episode(f"episode_{i}", q_vec, None)
            
            result = self.donut_search.donut_search(
                query_vector,
                inner_radius=0.05,  # より厳密な類似性
                outer_radius=0.5
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
            score = 0.0
            
            # 過去の経験から学習
            for q_vec, past_action, r_vec in self.query_response_pairs:
                if past_action == action:
                    # 同じ行動の結果を参照
                    similarity = 1.0 - np.linalg.norm(query_vector - q_vec)
                    
                    # レスポンスベクトルのゴール情報で重み付け
                    if r_vec[5] == 1.0:  # ゴール発見
                        score += similarity * 100.0
                    elif r_vec[3] == 1.0:  # ゴールへの移動（result='goal'）
                        score += similarity * 50.0
                    elif r_vec[3] == 0.0:  # 通常の移動（result='empty'）
                        score += similarity * 10.0
                    elif r_vec[3] == -1.0:  # 壁（result='wall'）
                        score -= similarity * 20.0
            
            # 探索ボーナス（未知の方向を優先）
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            next_pos = (state.position[0] + dx, state.position[1] + dy)
            if next_pos not in state.discovered_cells:
                score += 20.0
            
            # 訪問回数ペナルティ（軽減）
            visit_penalty = state.visited_positions.count(next_pos) * 2.0
            score -= visit_penalty
            
            action_scores[action] = score
            
        # 最高スコアの行動を選択（同点の場合はランダム）
        max_score = max(action_scores.values())
        best_actions = [a for a, s in action_scores.items() if s == max_score]
        best_action = np.random.choice(best_actions)
        
        action_names = ['↑', '→', '↓', '←']
        logger.info(f"Action scores: {[(action_names[a], round(s, 2)) for a, s in action_scores.items()]}")
        logger.info(f"Selected action: {best_action} ({action_names[best_action]})")
        
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
        """行動を実行してレスポンスベクトルを記録"""
        old_pos = self.current_state.position
        old_goal_info = self.current_state.discovered_cells.get(old_pos, -1.0)
        old_visits = self.current_state.visited_positions.count(old_pos)
        
        # クエリベクトル（行動前）
        query_vector = self.vector_space.create_query_vector(
            position=old_pos,
            visits=old_visits,
            goal_info=old_goal_info
        )
        
        # 行動実行
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
            
        # 移動成功時のみ訪問記録
        if result != 'wall':
            self.current_state.visited_positions.append(self.current_state.position)
            
        self.current_state.last_action = action
        self.current_state.last_result = result
        self.current_state.step_count += 1
        
        # レスポンスベクトル作成
        new_goal_info = self.current_state.discovered_cells.get(self.current_state.position, old_goal_info)
        response_vector = self.vector_space.create_response_vector(
            position=old_pos,  # 行動開始位置
            action=action,
            result=result,
            visits=old_visits,
            goal_info=new_goal_info  # 更新されたゴール情報
        )
        
        # クエリ・レスポンスペアを記録
        self.query_response_pairs.append((query_vector, action, response_vector))
        
        logger.info(f"Response vector: {response_vector}")
        
        return result
        
    def solve_maze(self, maze_query: Dict, max_steps: int = 50) -> Dict:
        """迷路を解く"""
        self.setup_maze(maze_query)
        
        print(f"\n=== Post-Action Query Agent ===")
        print(f"Start: {self.maze_env.start_pos}, Goal: Unknown initially")
        print(f"Query: [X, Y, null, null, visits, goal_info]")
        print(f"Response: [X, Y, action, result, visits, goal_info]\n")
        
        path_for_viz = []
        
        while self.current_state.step_count < max_steps:
            print(f"Step {self.current_state.step_count}: Position {self.current_state.position}", end=" ")
            
            # 行動決定
            action = self.decide_action(self.current_state)
            
            # 行動実行
            result = self.execute_action(action)
            actions = ['↑', '→', '↓', '←']
            print(f"→ {actions[action]} → {result}")
            
            path_for_viz.append(self.current_state.position)
            
            # ゴール判定
            if self.current_state.position == self.maze_env.goal_pos:
                print(f"\n🎉 Goal reached in {self.current_state.step_count} steps!")
                
                # ゴール到達時の特別なクエリ・レスポンスペア
                final_query = self.vector_space.create_query_vector(
                    position=self.current_state.position,
                    visits=self.current_state.visited_positions.count(self.current_state.position),
                    goal_info=1.0
                )
                # ゴールでの「滞在」を表すレスポンス
                final_response = self.vector_space.create_response_vector(
                    position=self.current_state.position,
                    action=4,  # 特別な値：滞在
                    result='goal',
                    visits=self.current_state.visited_positions.count(self.current_state.position),
                    goal_info=1.0
                )
                self.query_response_pairs.append((final_query, 4, final_response))
                
                break
                
        # 簡単な可視化
        self._visualize_result(path_for_viz)
        
        return {
            'success': self.current_state.position == self.maze_env.goal_pos,
            'steps': self.current_state.step_count,
            'path': self.current_state.visited_positions,
            'unique_positions': len(set(self.current_state.visited_positions)),
            'discovered_cells': len(self.current_state.discovered_cells),
            'episodes_recorded': len(self.query_response_pairs)
        }
        
    def _visualize_result(self, path: List[Tuple[int, int]]):
        """結果を可視化"""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # 迷路を描画
        maze_display = self.maze_env.grid.copy().astype(float)
        ax.imshow(maze_display, cmap='binary', alpha=0.3)
        
        # 経路を描画
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=2, alpha=0.7)
            
            # 訪問順を番号で表示
            for i, pos in enumerate(path[:20]):
                ax.text(pos[0], pos[1], str(i), fontsize=8, ha='center', va='center')
        
        # スタートとゴール
        ax.plot(*self.maze_env.start_pos, 'go', markersize=15, label='Start')
        ax.plot(*self.maze_env.goal_pos, 'r*', markersize=20, label='Goal')
        
        ax.set_title(f'Post-Action Query Agent (Steps: {self.current_state.step_count})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/post_action_query_solution.png')
        plt.close()
        print("\nVisualization saved to results/post_action_query_solution.png")


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
    agent = PostActionQueryAgent()
    result = agent.solve_maze(test_maze_3x3)
    
    print(f"\n=== Result ===")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Episodes recorded: {result['episodes_recorded']}")
    print(f"Efficiency: {result['unique_positions'] / result['steps']:.2%}")
    
    # 成功したら5x5でもテスト
    if result['success']:
        print("\n\nTesting with 5x5 maze...")
        agent2 = PostActionQueryAgent()
        result2 = agent2.solve_maze(test_maze_5x5)
        
        print(f"\n=== Result ===")
        print(f"Success: {result2['success']}")
        print(f"Steps: {result2['steps']}")
        print(f"Episodes recorded: {result2['episodes_recorded']}")
        print(f"Efficiency: {result2['unique_positions'] / result2['steps']:.2%}")


if __name__ == "__main__":
    main()