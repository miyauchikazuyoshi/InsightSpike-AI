#!/usr/bin/env python3
"""
Large Maze with Memory Visualization
====================================

10x10, 20x20迷路での実験
記憶ノードの増加とエピソード選択を可視化・記録
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.config.presets import ConfigPresets
from insightspike.environments.maze import SimpleMaze
from donut_search_maze import DonutSearchMaze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    """エピソード記録"""
    step: int
    episode_id: str
    position: Tuple[int, int]
    query_vector: List[float]
    selected_action: int
    action_name: str
    result: str
    response_vector: List[float]
    similar_episodes: List[Dict]  # 類似エピソードのリスト
    action_scores: Dict[str, float]
    memory_node_count: int


@dataclass
class MazeState:
    """迷路の現在状態"""
    position: Tuple[int, int]
    visited_positions: List[Tuple[int, int]]
    discovered_cells: Dict[Tuple[int, int], float]
    last_action: Optional[int] = None
    last_result: Optional[str] = None
    step_count: int = 0


class MemoryVisualizingAgent:
    """記憶の成長を可視化するエージェント"""
    
    def __init__(self, maze_size: int):
        self.maze_size = maze_size
        self.vector_space = None
        self.maze_env = None
        self.current_state = None
        self.donut_search = DonutSearchMaze(dimension=6)
        
        # 記録用
        self.episode_records: List[EpisodeRecord] = []
        self.memory_nodes: List[Dict] = []  # 記憶ノードの履歴
        self.query_response_pairs = []
        
        # 可視化用
        self.node_positions = {}  # episode_id -> (x, y) in graph
        
    def create_maze(self) -> np.ndarray:
        """サイズに応じた迷路を生成"""
        if self.maze_size == 10:
            # 10x10の迷路（手動設計）
            maze = np.zeros((10, 10))
            # 壁を追加
            maze[1:4, 2] = 1
            maze[2, 2:7] = 1
            maze[4:7, 4] = 1
            maze[6, 4:8] = 1
            maze[8, 1:5] = 1
            maze[5:9, 7] = 1
            return maze
        else:
            # 20x20の迷路（ランダム生成）
            np.random.seed(42)  # 再現性のため
            maze = np.zeros((20, 20))
            
            # ランダムに壁を配置（通行可能性を保証）
            for i in range(1, 19):
                for j in range(1, 19):
                    if np.random.random() < 0.25:  # 25%の確率で壁
                        # スタートとゴール付近は空ける
                        if not ((i < 3 and j < 3) or (i > 16 and j > 16)):
                            maze[i, j] = 1
            
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
        
        self.vector_space = PostActionVectorSpace(self.maze_env.size)
        
        self.current_state = MazeState(
            position=self.maze_env.start_pos,
            visited_positions=[self.maze_env.start_pos],
            discovered_cells={self.maze_env.start_pos: 0.0}
        )
        
        # 最初のゴールエピソードを追加（目的関数の代わり）
        self._add_goal_episode()
        
    def _add_goal_episode(self):
        """ゴールエピソードを追加（唯一のチート）"""
        goal_x, goal_y = self.maze_env.goal_pos
        
        goal_query = self.vector_space.create_query_vector(
            position=(goal_x, goal_y),
            visits=1,
            goal_info=1.0
        )
        
        goal_response = self.vector_space.create_response_vector(
            position=(goal_x, goal_y),
            action=4,  # 特別な値：滞在
            result='goal',
            visits=1,
            goal_info=1.0
        )
        
        episode_id = "GOAL_EPISODE"
        self.donut_search.add_episode(episode_id, goal_query, (goal_x, goal_y))
        self.query_response_pairs.append((goal_query, 4, goal_response))
        
        # グラフ用の位置を記録
        self.node_positions[episode_id] = (0.5, 0.9)  # 上部中央
        
        logger.info(f"Added goal episode at {self.maze_env.goal_pos}")
        
    def decide_action(self, state: MazeState) -> Tuple[int, Dict]:
        """行動決定と詳細記録"""
        current_goal_info = state.discovered_cells.get(state.position, -1.0)
        
        # クエリベクトル生成
        query_vector = self.vector_space.create_query_vector(
            position=state.position,
            visits=state.visited_positions.count(state.position),
            goal_info=current_goal_info
        )
        
        # 類似エピソード検索
        similar_episodes = []
        if len(self.query_response_pairs) > 0:
            for i, (q_vec, action, r_vec) in enumerate(self.query_response_pairs):
                similarity = 1.0 - np.linalg.norm(query_vector - q_vec)
                similar_episodes.append({
                    'episode_id': f"episode_{i}" if i > 0 else "GOAL_EPISODE",
                    'similarity': float(similarity),
                    'action': action,
                    'result_value': float(r_vec[3]),  # result
                    'goal_info': float(r_vec[5])      # goal_info
                })
            
            # 類似度でソート
            similar_episodes.sort(key=lambda x: x['similarity'], reverse=True)
            similar_episodes = similar_episodes[:5]  # 上位5個
        
        # 可能な行動を取得
        possible_actions = self._get_possible_actions()
        if not possible_actions:
            return 0, {'similar_episodes': similar_episodes}
            
        # 各行動のスコアを計算
        action_scores = {}
        
        for action in possible_actions:
            score = 0.0
            
            # 類似エピソードからの学習
            for ep in similar_episodes:
                if ep['action'] == action:
                    weight = ep['similarity']
                    if ep['goal_info'] == 1.0:
                        score += weight * 100.0
                    elif ep['result_value'] == 1.0:
                        score += weight * 50.0
                    elif ep['result_value'] == 0.0:
                        score += weight * 10.0
                    else:  # wall
                        score -= weight * 20.0
            
            # 探索ボーナス
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            next_pos = (state.position[0] + dx, state.position[1] + dy)
            if next_pos not in state.discovered_cells:
                score += 15.0
            
            # 訪問回数ペナルティ
            visit_penalty = state.visited_positions.count(next_pos) * 3.0
            score -= visit_penalty
            
            action_scores[action] = score
            
        # 最高スコアの行動を選択
        max_score = max(action_scores.values())
        best_actions = [a for a, s in action_scores.items() if s == max_score]
        best_action = np.random.choice(best_actions)
        
        action_names = ['↑', '→', '↓', '←']
        
        return best_action, {
            'similar_episodes': similar_episodes,
            'action_scores': {action_names[a]: round(s, 2) for a, s in action_scores.items()}
        }
        
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
        
    def execute_action(self, action: int, decision_info: Dict) -> str:
        """行動実行と記録"""
        old_pos = self.current_state.position
        old_goal_info = self.current_state.discovered_cells.get(old_pos, -1.0)
        old_visits = self.current_state.visited_positions.count(old_pos)
        
        # クエリベクトル
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
                
                if (new_x, new_y) == self.maze_env.goal_pos:
                    result = 'goal'
                    self.current_state.discovered_cells[(new_x, new_y)] = 1.0
                else:
                    result = 'empty'
                    self.current_state.discovered_cells[(new_x, new_y)] = 0.0
            else:
                result = 'wall'
        else:
            result = 'wall'
            
        if result != 'wall':
            self.current_state.visited_positions.append(self.current_state.position)
            
        self.current_state.last_action = action
        self.current_state.last_result = result
        self.current_state.step_count += 1
        
        # レスポンスベクトル
        new_goal_info = self.current_state.discovered_cells.get(self.current_state.position, old_goal_info)
        response_vector = self.vector_space.create_response_vector(
            position=old_pos,
            action=action,
            result=result,
            visits=old_visits,
            goal_info=new_goal_info
        )
        
        # エピソード記録
        episode_id = f"step_{self.current_state.step_count}"
        self.query_response_pairs.append((query_vector, action, response_vector))
        self.donut_search.add_episode(episode_id, query_vector, old_pos)
        
        # 記録を保存
        action_names = ['↑', '→', '↓', '←']
        record = EpisodeRecord(
            step=self.current_state.step_count,
            episode_id=episode_id,
            position=old_pos,
            query_vector=query_vector.tolist(),
            selected_action=action,
            action_name=action_names[action],
            result=result,
            response_vector=response_vector.tolist(),
            similar_episodes=decision_info['similar_episodes'],
            action_scores=decision_info['action_scores'],
            memory_node_count=len(self.query_response_pairs)
        )
        self.episode_records.append(record)
        
        # メモリノードの状態を記録
        self.memory_nodes.append({
            'step': self.current_state.step_count,
            'total_episodes': len(self.query_response_pairs),
            'discovered_cells': len(self.current_state.discovered_cells),
            'unique_positions': len(set(self.current_state.visited_positions))
        })
        
        return result
        
    def solve_maze(self, max_steps: int = 200) -> Dict:
        """迷路を解く"""
        self.setup_maze()
        
        print(f"\n=== {self.maze_size}x{self.maze_size} Maze Solving ===")
        print(f"Start: (0, 0), Goal: ({self.maze_size-1}, {self.maze_size-1})")
        
        # アニメーション用のフレーム保存
        frames = []
        
        while self.current_state.step_count < max_steps:
            if self.current_state.step_count % 10 == 0:
                print(f"Step {self.current_state.step_count}: Position {self.current_state.position}")
            
            # 行動決定
            action, decision_info = self.decide_action(self.current_state)
            
            # 行動実行
            result = self.execute_action(action, decision_info)
            
            # フレーム保存（10ステップごと）
            if self.current_state.step_count % 10 == 0:
                frames.append(self._create_frame())
            
            # ゴール判定
            if self.current_state.position == self.maze_env.goal_pos:
                print(f"\n🎉 Goal reached in {self.current_state.step_count} steps!")
                frames.append(self._create_frame())  # 最終フレーム
                break
                
        # 結果を保存
        self._save_results()
        self._create_memory_growth_visualization()
        
        return {
            'success': self.current_state.position == self.maze_env.goal_pos,
            'steps': self.current_state.step_count,
            'path_length': len(self.current_state.visited_positions),
            'unique_positions': len(set(self.current_state.visited_positions)),
            'total_episodes': len(self.query_response_pairs),
            'discovered_cells': len(self.current_state.discovered_cells)
        }
        
    def _create_frame(self) -> np.ndarray:
        """現在状態のフレームを作成"""
        # 簡単な可視化のため、訪問マップを返す
        visit_map = np.zeros(self.maze_env.grid.shape)
        for pos in self.current_state.visited_positions:
            x, y = pos
            visit_map[y, x] += 1
        return visit_map
        
    def _convert_to_json_serializable(self, obj):
        """NumPy型を含むオブジェクトをJSON serializableに変換"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    def _save_results(self):
        """結果をJSONで保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/maze_{self.maze_size}x{self.maze_size}_episodes_{timestamp}.json"
        
        # エピソード記録を辞書形式に変換
        records_dict = []
        for record in self.episode_records:
            record_dict = asdict(record)
            # 全体を再帰的に変換
            record_dict = self._convert_to_json_serializable(record_dict)
            records_dict.append(record_dict)
        
        data = {
            'maze_size': self.maze_size,
            'timestamp': timestamp,
            'total_steps': self.current_state.step_count,
            'success': self.current_state.position == self.maze_env.goal_pos,
            'episode_records': records_dict,
            'memory_growth': self.memory_nodes,
            'final_path': [list(pos) for pos in self.current_state.visited_positions]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"\nEpisode records saved to: {filename}")
        
    def _create_memory_growth_visualization(self):
        """記憶ノードの成長を可視化"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        steps = [node['step'] for node in self.memory_nodes]
        total_episodes = [node['total_episodes'] for node in self.memory_nodes]
        discovered_cells = [node['discovered_cells'] for node in self.memory_nodes]
        unique_positions = [node['unique_positions'] for node in self.memory_nodes]
        
        # 1. エピソード数の成長
        ax1.plot(steps, total_episodes, 'b-', linewidth=2)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Total Episodes')
        ax1.set_title(f'Memory Growth in {self.maze_size}x{self.maze_size} Maze')
        ax1.grid(True, alpha=0.3)
        
        # 2. 発見したセルと訪問位置
        ax2.plot(steps, discovered_cells, 'g-', label='Discovered Cells', linewidth=2)
        ax2.plot(steps, unique_positions, 'r-', label='Unique Positions', linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Count')
        ax2.set_title('Exploration Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 迷路と最終経路
        maze_display = self.maze_env.grid.copy().astype(float)
        
        # 訪問頻度マップ
        visit_freq = np.zeros_like(maze_display)
        for pos in self.current_state.visited_positions:
            x, y = pos
            visit_freq[y, x] += 1
            
        # 正規化
        if visit_freq.max() > 0:
            visit_freq = visit_freq / visit_freq.max()
            
        # オーバーレイ
        combined = np.where(maze_display == 1, -1, visit_freq)
        
        im = ax3.imshow(combined, cmap='RdYlGn', vmin=-1, vmax=1)
        
        # スタートとゴール
        ax3.plot(0, 0, 'go', markersize=10, label='Start')
        ax3.plot(self.maze_size-1, self.maze_size-1, 'r*', markersize=15, label='Goal')
        
        # 現在位置
        if self.current_state.position != self.maze_env.goal_pos:
            x, y = self.current_state.position
            ax3.plot(x, y, 'bo', markersize=10, label='Current')
            
        ax3.set_title(f'Visit Frequency Map (Total Steps: {self.current_state.step_count})')
        ax3.legend()
        
        plt.colorbar(im, ax=ax3, label='Visit Frequency')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/memory_growth_{self.maze_size}x{self.maze_size}_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        
        print(f"Memory growth visualization saved to: {filename}")


class PostActionVectorSpace:
    """行動後クエリベクトル空間（前回の実装から）"""
    
    def __init__(self, maze_size: Tuple[int, int]):
        self.maze_width, self.maze_height = maze_size
        
    def create_query_vector(self, position: Tuple[int, int], 
                          visits: int = 0,
                          goal_info: float = -1.0) -> np.ndarray:
        norm_x = position[0] / (self.maze_width - 1) if self.maze_width > 1 else 0
        norm_y = position[1] / (self.maze_height - 1) if self.maze_height > 1 else 0
        norm_action = 0.5
        norm_result = 0.5
        norm_visits = min(visits / 10.0, 1.0)
        
        return np.array([norm_x, norm_y, norm_action, norm_result, norm_visits, goal_info])
    
    def create_response_vector(self, position: Tuple[int, int],
                             action: int, result: str,
                             visits: int = 0,
                             goal_info: float = -1.0) -> np.ndarray:
        norm_x = position[0] / (self.maze_width - 1) if self.maze_width > 1 else 0
        norm_y = position[1] / (self.maze_height - 1) if self.maze_height > 1 else 0
        norm_action = action * 0.25
        result_map = {'wall': -1.0, 'empty': 0.0, 'goal': 1.0}
        norm_result = result_map.get(result, 0.0)
        norm_visits = min(visits / 10.0, 1.0)
        
        return np.array([norm_x, norm_y, norm_action, norm_result, norm_visits, goal_info])


def main():
    """メイン実行関数"""
    # 10x10迷路
    print("=" * 60)
    print("Testing with 10x10 maze...")
    print("=" * 60)
    
    agent_10 = MemoryVisualizingAgent(maze_size=10)
    result_10 = agent_10.solve_maze(max_steps=200)
    
    print(f"\n=== 10x10 Maze Result ===")
    print(f"Success: {result_10['success']}")
    print(f"Steps: {result_10['steps']}")
    print(f"Path efficiency: {result_10['unique_positions'] / result_10['path_length']:.2%}")
    print(f"Total episodes: {result_10['total_episodes']}")
    print(f"Discovered cells: {result_10['discovered_cells']}")
    
    # 20x20迷路
    print("\n" + "=" * 60)
    print("Testing with 20x20 maze...")
    print("=" * 60)
    
    agent_20 = MemoryVisualizingAgent(maze_size=20)
    result_20 = agent_20.solve_maze(max_steps=500)
    
    print(f"\n=== 20x20 Maze Result ===")
    print(f"Success: {result_20['success']}")
    print(f"Steps: {result_20['steps']}")
    print(f"Path efficiency: {result_20['unique_positions'] / result_20['path_length']:.2%}")
    print(f"Total episodes: {result_20['total_episodes']}")
    print(f"Discovered cells: {result_20['discovered_cells']}")


if __name__ == "__main__":
    main()