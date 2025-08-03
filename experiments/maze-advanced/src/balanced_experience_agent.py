#!/usr/bin/env python3
"""
Balanced Experience-Based Agent
===============================

経験ベースの学習に最小限のガイダンスを追加
- 基本は類似性ベース
- 探索促進のための新規性ボーナス
- 方向性の連続性を考慮
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
import json
from datetime import datetime
from collections import deque

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from donut_search_maze import DonutSearchMaze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MazeState:
    position: Tuple[int, int]
    visited_positions: List[Tuple[int, int]]
    last_action: Optional[int] = None
    last_result: Optional[str] = None
    step_count: int = 0
    
    # 追加：最近の行動履歴
    recent_actions: deque = None
    
    def __post_init__(self):
        if self.recent_actions is None:
            self.recent_actions = deque(maxlen=5)


class BalancedExperienceAgent:
    """バランスの取れた経験ベースエージェント"""
    
    def __init__(self, maze_size: int):
        self.maze_size = maze_size
        self.vector_space = None
        self.maze_env = None
        self.current_state = None
        
        # エピソード記憶
        self.episodes = []  # (state_vector, action, result_vector)
        
        # 探索用のドーナツサーチ
        self.donut_search = None
        
        # 記録用
        self.decision_log = []
        
    def create_maze(self) -> np.ndarray:
        """迷路を生成"""
        if self.maze_size == 5:
            maze = np.array([
                [0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ])
        else:
            # 10x10
            maze = np.zeros((10, 10))
            maze[1:4, 2] = 1
            maze[2, 2:7] = 1
            maze[4:7, 4] = 1
            maze[6, 4:8] = 1
            maze[8, 1:5] = 1
            return maze
        return maze
    
    def setup_maze(self):
        """迷路環境をセットアップ"""
        maze_array = self.create_maze()
        
        self.maze_env = SimpleMaze(
            size=(self.maze_size, self.maze_size),
            maze_type='custom',
            maze_layout=maze_array,
            start_pos=(0, 0),
            goal_pos=(self.maze_size-1, self.maze_size-1)
        )
        
        self.vector_space = EnhancedVectorSpace(self.maze_env.size)
        
        self.current_state = MazeState(
            position=self.maze_env.start_pos,
            visited_positions=[self.maze_env.start_pos]
        )
        
        # ドーナツサーチの初期化
        self.donut_search = DonutSearchMaze(self.maze_env)
        
        # 初期エピソードを追加
        self._add_initial_episodes()
        
    def _add_initial_episodes(self):
        """初期エピソードを追加（ゴール＋初期探索）"""
        # ゴールエピソード
        goal_x, goal_y = self.maze_env.goal_pos
        goal_vector = self.vector_space.create_state_vector(
            position=(goal_x, goal_y),
            action=None,
            result='goal',
            visit_count=1
        )
        self.episodes.append((goal_vector, 4, goal_vector))  # 特別な滞在アクション
        
        # スタート地点からの4方向探索
        start_x, start_y = self.maze_env.start_pos
        for action, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            nx, ny = start_x + dx, start_y + dy
            
            start_vector = self.vector_space.create_state_vector(
                position=(start_x, start_y),
                action=None,
                result=None,
                visit_count=1
            )
            
            if (0 <= nx < self.maze_env.size[0] and 
                0 <= ny < self.maze_env.size[1] and
                (self.maze_env.grid[ny, nx] == 0 or (nx, ny) == self.maze_env.goal_pos)):
                result = 'goal' if (nx, ny) == self.maze_env.goal_pos else 'empty'
                result_vector = self.vector_space.create_state_vector(
                    position=(nx, ny),
                    action=action,
                    result=result,
                    visit_count=1
                )
            else:
                result = 'wall'
                result_vector = self.vector_space.create_state_vector(
                    position=(start_x, start_y),
                    action=action,
                    result=result,
                    visit_count=1
                )
            
            self.episodes.append((start_vector, action, result_vector))
        
        logger.info(f"Added {len(self.episodes)} initial episodes")
        
    def decide_action_balanced(self, state: MazeState) -> int:
        """バランスの取れた行動決定"""
        
        # 現在状態のベクトル
        visit_count = state.visited_positions.count(state.position)
        current_vector = self.vector_space.create_state_vector(
            position=state.position,
            action=state.last_action,
            result=state.last_result,
            visit_count=visit_count
        )
        
        # 可能な行動
        possible_actions = self._get_possible_actions()
        if not possible_actions:
            return 0
        
        # 各行動のスコアを計算
        action_scores = {}
        
        for action in possible_actions:
            # 1. 基本の類似性スコア
            similarity_score = self._calculate_similarity_score(
                current_vector, action
            )
            
            # 2. 新規性ボーナス（未探索方向への促進）
            novelty_bonus = self._calculate_novelty_bonus(
                state, action
            )
            
            # 3. 方向性の連続性（同じ方向への継続を促進）
            continuity_bonus = self._calculate_continuity_bonus(
                state, action
            )
            
            # 4. ドーナツサーチによる探索価値
            exploration_value = self._calculate_exploration_value(
                state.position, action
            )
            
            # 総合スコア
            total_score = (
                similarity_score * 0.4 +
                novelty_bonus * 0.3 +
                continuity_bonus * 0.1 +
                exploration_value * 0.2
            )
            
            action_scores[action] = {
                'total': total_score,
                'similarity': similarity_score,
                'novelty': novelty_bonus,
                'continuity': continuity_bonus,
                'exploration': exploration_value
            }
        
        # 最高スコアの行動を選択（ε-greedy）
        epsilon = max(0.1, 0.3 - state.step_count * 0.001)  # 時間とともに減少
        
        if np.random.random() < epsilon:
            selected_action = np.random.choice(possible_actions)
            reason = 'exploration'
        else:
            max_score = max(scores['total'] for scores in action_scores.values())
            best_actions = [a for a, scores in action_scores.items() 
                           if scores['total'] == max_score]
            selected_action = np.random.choice(best_actions)
            reason = 'exploitation'
        
        # 決定をログに記録
        action_names = ['↑', '→', '↓', '←']
        self.decision_log.append({
            'step': state.step_count,
            'position': state.position,
            'state_vector': current_vector.tolist(),
            'action_scores': {
                action_names[a]: {k: round(v, 3) for k, v in scores.items()}
                for a, scores in action_scores.items()
            },
            'selected_action': action_names[selected_action],
            'reason': reason,
            'epsilon': round(epsilon, 3)
        })
        
        return selected_action
    
    def _calculate_similarity_score(self, current_vector: np.ndarray, action: int) -> float:
        """類似性スコアの計算"""
        max_similarity = -1.0
        
        for ep_state, ep_action, ep_result in self.episodes:
            if ep_action == action:
                # 成功エピソードを重視
                result_weight = 1.0
                if hasattr(ep_result, '__iter__') and len(ep_result) >= 4:
                    if ep_result[3] == 1.0:  # goal
                        result_weight = 2.0
                    elif ep_result[3] == -1.0:  # wall
                        result_weight = 0.5
                
                similarity = np.dot(current_vector, ep_state) / (
                    np.linalg.norm(current_vector) * np.linalg.norm(ep_state) + 1e-10
                )
                
                weighted_similarity = similarity * result_weight
                max_similarity = max(max_similarity, weighted_similarity)
        
        return max_similarity
    
    def _calculate_novelty_bonus(self, state: MazeState, action: int) -> float:
        """新規性ボーナスの計算"""
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        nx, ny = state.position[0] + dx, state.position[1] + dy
        
        if (nx, ny) not in state.visited_positions:
            return 1.0  # 未訪問地点へのボーナス
        else:
            # 訪問回数に応じて減衰
            visit_count = state.visited_positions.count((nx, ny))
            return 1.0 / (1.0 + visit_count)
    
    def _calculate_continuity_bonus(self, state: MazeState, action: int) -> float:
        """方向性の連続性ボーナス"""
        if not state.recent_actions:
            return 0.5
        
        # 最近の行動と同じ方向なら高スコア
        if state.last_action == action:
            return 1.0
        
        # 逆方向なら低スコア
        opposite = (action + 2) % 4
        if state.last_action == opposite:
            return 0.0
        
        return 0.5
    
    def _calculate_exploration_value(self, position: Tuple[int, int], action: int) -> float:
        """ドーナツサーチによる探索価値"""
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        nx, ny = position[0] + dx, position[1] + dy
        
        if not (0 <= nx < self.maze_env.size[0] and 0 <= ny < self.maze_env.size[1]):
            return 0.0
        
        # 現在位置からの距離
        current_distance = np.sqrt((nx - position[0])**2 + (ny - position[1])**2)
        
        # ゴールまでの距離
        goal_distance = np.sqrt(
            (nx - self.maze_env.goal_pos[0])**2 + 
            (ny - self.maze_env.goal_pos[1])**2
        )
        
        # ドーナツ領域内なら高評価
        if 0.2 < goal_distance / max(self.maze_env.size) < 0.8:
            return 0.8
        else:
            return 0.2
    
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
        """行動を実行して経験を記録"""
        old_pos = self.current_state.position
        visit_count = self.current_state.visited_positions.count(old_pos)
        
        old_vector = self.vector_space.create_state_vector(
            position=old_pos,
            action=self.current_state.last_action,
            result=self.current_state.last_result,
            visit_count=visit_count
        )
        
        # 行動実行
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
        
        if (0 <= new_x < self.maze_env.size[0] and 
            0 <= new_y < self.maze_env.size[1]):
            
            if self.maze_env.grid[new_y, new_x] == 0 or (new_x, new_y) == self.maze_env.goal_pos:
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
            
        if result != 'wall':
            self.current_state.visited_positions.append(self.current_state.position)
            
        self.current_state.last_action = action
        self.current_state.last_result = result
        self.current_state.recent_actions.append(action)
        self.current_state.step_count += 1
        
        # 結果ベクトル
        new_visit_count = self.current_state.visited_positions.count(self.current_state.position)
        result_vector = self.vector_space.create_state_vector(
            position=self.current_state.position,
            action=action,
            result=result,
            visit_count=new_visit_count
        )
        
        # エピソードとして記録
        self.episodes.append((old_vector, action, result_vector))
        
        # ドーナツサーチの更新
        if self.donut_search:
            self.donut_search.update_position(self.current_state.position)
        
        return result
    
    def solve_maze(self, max_steps: int = 100) -> Dict:
        """迷路を解く"""
        self.setup_maze()
        
        print(f"\n=== Balanced Experience {self.maze_size}x{self.maze_size} Maze ===")
        print(f"Similarity + Novelty + Continuity + Exploration")
        print(f"Start: (0, 0), Goal: ({self.maze_size-1}, {self.maze_size-1})\n")
        
        path_history = [self.current_state.position]
        
        while self.current_state.step_count < max_steps:
            if self.current_state.step_count % 10 == 0:
                print(f"Step {self.current_state.step_count}: Position {self.current_state.position}")
            
            # バランスの取れた行動決定
            action = self.decide_action_balanced(self.current_state)
            
            # 行動実行
            result = self.execute_action(action)
            
            path_history.append(self.current_state.position)
            
            # ゴール判定
            if self.current_state.position == self.maze_env.goal_pos:
                print(f"\n🎉 Goal reached in {self.current_state.step_count} steps!")
                break
        
        # 結果を保存
        self._save_results(path_history)
        self._visualize_path(path_history)
        
        return {
            'success': self.current_state.position == self.maze_env.goal_pos,
            'steps': self.current_state.step_count,
            'path_length': len(path_history),
            'unique_positions': len(set(path_history)),
            'total_episodes': len(self.episodes)
        }
    
    def _save_results(self, path_history: List[Tuple[int, int]]):
        """結果をJSONで保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/balanced_experience_{self.maze_size}x{self.maze_size}_{timestamp}.json"
        
        data = {
            'maze_size': self.maze_size,
            'timestamp': timestamp,
            'total_steps': self.current_state.step_count,
            'success': self.current_state.position == self.maze_env.goal_pos,
            'path': [list(pos) for pos in path_history],
            'unique_positions': len(set(path_history)),
            'decision_log': self.decision_log[:20],  # 最初の20ステップのみ
            'total_episodes': len(self.episodes)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"\nDecision log saved to: {filename}")
    
    def _visualize_path(self, path_history: List[Tuple[int, int]]):
        """経路を可視化"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 迷路を描画
        maze_display = self.maze_env.grid.copy().astype(float)
        ax.imshow(maze_display, cmap='binary', alpha=0.3)
        
        # 訪問頻度マップ
        visit_map = np.zeros_like(maze_display)
        for pos in path_history:
            x, y = pos
            visit_map[y, x] += 1
        
        # 正規化してオーバーレイ
        if visit_map.max() > 0:
            visit_map = visit_map / visit_map.max()
        
        combined = np.where(maze_display == 1, -0.5, visit_map)
        im = ax.imshow(combined, cmap='RdYlGn', vmin=-0.5, vmax=1, alpha=0.8)
        
        # パスを線で描画
        if len(path_history) > 1:
            path_array = np.array(path_history)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=1, alpha=0.5)
        
        # スタートとゴール
        ax.plot(0, 0, 'go', markersize=15, label='Start')
        ax.plot(self.maze_size-1, self.maze_size-1, 'r*', markersize=20, label='Goal')
        
        ax.set_title(f'Balanced Experience Path (Steps: {self.current_state.step_count})')
        ax.legend()
        
        plt.colorbar(im, ax=ax, label='Visit Frequency')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/balanced_experience_path_{self.maze_size}x{self.maze_size}_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        
        print(f"Path visualization saved to: {filename}")


class EnhancedVectorSpace:
    """拡張された状態ベクトル空間（5次元）"""
    
    def __init__(self, maze_size: Tuple[int, int]):
        self.maze_width, self.maze_height = maze_size
        
    def create_state_vector(self, position: Tuple[int, int], 
                          action: Optional[int] = None,
                          result: Optional[str] = None,
                          visit_count: int = 0) -> np.ndarray:
        """状態を5次元ベクトルに変換"""
        # 位置を正規化
        norm_x = position[0] / (self.maze_width - 1) if self.maze_width > 1 else 0
        norm_y = position[1] / (self.maze_height - 1) if self.maze_height > 1 else 0
        
        # 行動を正規化
        if action is not None:
            norm_action = action * 0.25
        else:
            norm_action = 0.5
        
        # 結果をエンコード
        result_map = {'wall': -1.0, 'empty': 0.0, 'goal': 1.0, None: 0.0}
        norm_result = result_map.get(result, 0.0)
        
        # 訪問回数を正規化（対数スケール）
        norm_visits = np.log1p(visit_count) / 10.0
        
        return np.array([norm_x, norm_y, norm_action, norm_result, norm_visits])


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("Testing Balanced Experience-Based Agent")
    print("Similarity + Novelty + Continuity + Exploration")
    print("=" * 60)
    
    # 5x5迷路でテスト
    agent_5 = BalancedExperienceAgent(maze_size=5)
    result_5 = agent_5.solve_maze(max_steps=100)
    
    print(f"\n=== 5x5 Maze Result ===")
    print(f"Success: {result_5['success']}")
    print(f"Steps: {result_5['steps']}")
    print(f"Unique positions: {result_5['unique_positions']}")
    print(f"Path efficiency: {result_5['unique_positions'] / result_5['path_length']:.2%}")
    print(f"Total episodes learned: {result_5['total_episodes']}")
    
    # 10x10迷路でテスト
    print("\n" + "=" * 60)
    
    agent_10 = BalancedExperienceAgent(maze_size=10)
    result_10 = agent_10.solve_maze(max_steps=200)
    
    print(f"\n=== 10x10 Maze Result ===")
    print(f"Success: {result_10['success']}")
    print(f"Steps: {result_10['steps']}")
    print(f"Unique positions: {result_10['unique_positions']}")
    print(f"Path efficiency: {result_10['unique_positions'] / result_10['path_length']:.2%}")
    print(f"Total episodes learned: {result_10['total_episodes']}")


if __name__ == "__main__":
    main()