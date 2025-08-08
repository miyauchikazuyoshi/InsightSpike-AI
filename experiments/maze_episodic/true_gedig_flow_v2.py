#!/usr/bin/env python3
"""
True geDIG Flow Navigator V2
============================

修正版：より正確なgeDIG計算
- グラフ編集距離を適切に計算
- 情報利得をより現実的に評価
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
from dataclasses import dataclass

@dataclass
class Episode:
    """エピソード記憶"""
    id: int
    vector: np.ndarray  # 6D: [x, y, direction, result, wall/path, goal/not]
    
class TrueGeDIGFlowNavigatorV2:
    """修正版geDIGナビゲーター"""
    
    def __init__(self, maze: np.ndarray,
                 inner_radius: float = 0.0,
                 outer_radius: float = 1.5):
        self.maze = maze
        self.height, self.width = maze.shape
        self.start_pos = (1, 1)
        self.goal_pos = (self.width-2, self.height-2)
        self.position = self.start_pos
        
        # ドーナツ検索パラメータ
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        
        # エピソード記憶とグラフ
        self.episodes = []
        self.episode_graph = nx.Graph()
        
        # 初期記憶を生成
        self._create_initial_memories()
        
        # 統計
        self.steps = 0
        self.path = [self.position]
        
    def _create_initial_memories(self):
        """初期記憶の生成（4方向の壁エピソード+ゴール）"""
        # 方向エンコーディング
        direction_encoding = {
            'up': 0.0, 'right': 0.25, 'down': 0.5, 'left': 0.75
        }
        
        # 初期4方向のエピソード（原点での壁エピソード）
        for direction, encoding in direction_encoding.items():
            vector = np.array([
                0.0,      # x座標（原点）
                0.0,      # y座標（原点）
                encoding, # 移動方向
                0.0,      # 結果（壁）
                0.0,      # 壁
                0.0       # ゴールではない
            ])
            
            episode = Episode(id=len(self.episodes), vector=vector)
            self.episodes.append(episode)
            self.episode_graph.add_node(episode.id)
        
        # ゴールエピソード
        goal_vector = np.array([
            self.goal_pos[0] / self.width,   # ゴールx座標
            self.goal_pos[1] / self.height,  # ゴールy座標
            0.5,                              # 方向（中立）
            1.0,                              # 結果（成功）
            1.0,                              # 通路
            1.0                               # ゴール
        ])
        
        goal_episode = Episode(id=len(self.episodes), vector=goal_vector)
        self.episodes.append(goal_episode)
        self.episode_graph.add_node(goal_episode.id)
        
    def _create_query(self, position: Tuple[int, int]) -> np.ndarray:
        """クエリの生成"""
        return np.array([
            position[0] / self.width,   # 現x座標
            position[1] / self.height,  # 現y座標
            0.5,                        # null (方向)
            0.5,                        # null (結果)
            0.5,                        # null (壁/通路)
            1.0                         # ゴール（目的）
        ])
    
    def _donut_search(self, query: np.ndarray) -> List[int]:
        """ドーナツ検索による類似エピソード取得"""
        candidates = []
        
        for i, episode in enumerate(self.episodes):
            # ノルム距離
            distance = np.linalg.norm(query - episode.vector)
            
            # ドーナツ検索（内径0で全て含む）
            if distance <= self.outer_radius:
                candidates.append(i)
        
        return candidates
    
    def _calculate_gedig(self, episode_ids: List[int], current_pos: Tuple[int, int]) -> float:
        """エピソード集合のgeDIG計算（改良版）"""
        if not episode_ids:
            return float('inf')
        
        # サブグラフ作成
        subgraph = self.episode_graph.subgraph(episode_ids)
        
        # GED計算：ノード数 + エッジ数
        ged = len(episode_ids) + subgraph.number_of_edges()
        
        # IG計算：より現実的な情報利得
        ig = 0.0
        
        for ep_id in episode_ids:
            episode = self.episodes[ep_id]
            
            # ゴール情報の価値（距離に応じて減衰）
            if episode.vector[5] > 0:  # ゴールエピソード
                goal_x = episode.vector[0] * self.width
                goal_y = episode.vector[1] * self.height
                dist_to_goal = abs(current_pos[0] - goal_x) + abs(current_pos[1] - goal_y)
                # 距離が遠いほど価値が低い
                goal_value = 1.0 / (1.0 + dist_to_goal / 10.0)
                ig += goal_value
            
            # 移動成功情報の価値
            if episode.vector[3] > 0:  # 成功エピソード
                # 位置の近さに応じて価値が高い
                ep_x = episode.vector[0] * self.width
                ep_y = episode.vector[1] * self.height
                dist = abs(current_pos[0] - ep_x) + abs(current_pos[1] - ep_y)
                if dist < 3:  # 近いエピソードは価値が高い
                    ig += 0.3
            
            # 壁情報の価値
            if episode.vector[3] == 0:  # 壁エピソード
                # 近い壁情報は価値がある
                ep_x = episode.vector[0] * self.width
                ep_y = episode.vector[1] * self.height
                dist = abs(current_pos[0] - ep_x) + abs(current_pos[1] - ep_y)
                if dist < 2:
                    ig += 0.2
        
        # geDIG = GED - IG
        return ged - ig
    
    def _find_minimum_gedig_subset(self, candidate_ids: List[int]) -> List[int]:
        """geDIG最小化する部分集合を見つける"""
        if not candidate_ids:
            return []
        
        best_subset = []
        best_gedig = float('inf')
        
        # 各候補を起点として評価（最大3つまでの組み合わせ）
        for i, start_id in enumerate(candidate_ids):
            # 単独
            gedig = self._calculate_gedig([start_id], self.position)
            if gedig < best_gedig:
                best_gedig = gedig
                best_subset = [start_id]
            
            # ペア
            for j, second_id in enumerate(candidate_ids[i+1:], i+1):
                subset = [start_id, second_id]
                gedig = self._calculate_gedig(subset, self.position)
                if gedig < best_gedig:
                    best_gedig = gedig
                    best_subset = subset
            
            # トリプル（計算量を抑えるため制限）
            if len(candidate_ids) > 2:
                for j in range(i+1, min(i+3, len(candidate_ids))):
                    for k in range(j+1, min(j+2, len(candidate_ids))):
                        if k < len(candidate_ids):
                            subset = [start_id, candidate_ids[j], candidate_ids[k]]
                            gedig = self._calculate_gedig(subset, self.position)
                            if gedig < best_gedig:
                                best_gedig = gedig
                                best_subset = subset
        
        return best_subset
    
    def _form_insight_episode(self, episode_ids: List[int]) -> np.ndarray:
        """洞察エピソードの形成"""
        if not episode_ids:
            # エピソードがない場合はランダム
            return np.array([0.5, 0.5, np.random.random(), 0.5, 0.5, 0.0])
        
        # 選択されたエピソードの統合
        vectors = [self.episodes[ep_id].vector for ep_id in episode_ids]
        
        # 距離ベースの重み付け（近いエピソードを重視）
        weights = []
        for vector in vectors:
            ep_x = vector[0] * self.width
            ep_y = vector[1] * self.height
            dist = abs(self.position[0] - ep_x) + abs(self.position[1] - ep_y)
            weight = 1.0 / (1.0 + dist)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        insight_vector = np.zeros(6)
        for i, vector in enumerate(vectors):
            insight_vector += weights[i] * vector
        
        return insight_vector
    
    def _normalize_direction(self, insight_vector: np.ndarray) -> str:
        """洞察エピソードから方向を正規化"""
        direction_value = insight_vector[2]
        
        # 方向成分を4方向に正規化
        if direction_value < 0.125:
            return 'up'
        elif direction_value < 0.375:
            return 'right'
        elif direction_value < 0.625:
            return 'down'
        else:
            return 'left'
    
    def _add_episode(self, position: Tuple[int, int], direction: str, 
                    result: str, reached_goal: bool):
        """新しいエピソードを追加"""
        direction_encoding = {
            'up': 0.0, 'right': 0.25, 'down': 0.5, 'left': 0.75
        }
        
        result_encoding = {
            'success': 1.0, 'wall': 0.0, 'visited': 0.5
        }
        
        # その位置が壁か通路かの情報
        is_wall = self.maze[position[1], position[0]] == 1
        wall_path_info = 0.0 if is_wall else 1.0
        
        vector = np.array([
            position[0] / self.width,
            position[1] / self.height,
            direction_encoding[direction],
            result_encoding[result],
            wall_path_info,
            1.0 if reached_goal else 0.0
        ])
        
        episode = Episode(id=len(self.episodes), vector=vector)
        self.episodes.append(episode)
        self.episode_graph.add_node(episode.id)
        
        # 類似エピソードとエッジを張る
        for i, other in enumerate(self.episodes[:-1]):
            similarity = 1.0 - np.linalg.norm(vector - other.vector)
            if similarity > 0.5:  # 閾値
                self.episode_graph.add_edge(episode.id, i, weight=similarity)
    
    def navigate_step(self) -> Tuple[str, bool]:
        """1ステップのナビゲーション"""
        # クエリ生成
        query = self._create_query(self.position)
        
        # ドーナツ検索
        candidates = self._donut_search(query)
        
        # geDIG最小化する組み合わせ選択
        selected_episodes = self._find_minimum_gedig_subset(candidates)
        
        # 洞察エピソード形成
        insight = self._form_insight_episode(selected_episodes)
        
        # 方向決定
        direction = self._normalize_direction(insight)
        
        # 実行
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[direction]
        new_pos = (self.position[0] + dx, self.position[1] + dy)
        
        # 実行前の位置を記録
        old_pos = self.position
        
        # 結果判定
        result = 'wall'
        if (0 <= new_pos[0] < self.width and 
            0 <= new_pos[1] < self.height and
            self.maze[new_pos[1], new_pos[0]] == 0):
            result = 'success'
            self.position = new_pos
            self.path.append(new_pos)
        
        reached_goal = (self.position == self.goal_pos)
        
        # エピソード追加（実行前の位置から実行した記録）
        self._add_episode(old_pos, direction, result, reached_goal)
        
        self.steps += 1
        
        return direction, reached_goal
    
    def navigate(self, max_steps: int = 1000) -> Dict:
        """完全なナビゲーション"""
        print(f"True geDIG Flow Navigation V2")
        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
        print(f"Initial episodes: {len(self.episodes)}")
        
        while self.steps < max_steps:
            if self.steps % 50 == 0:
                dist = abs(self.position[0] - self.goal_pos[0]) + \
                       abs(self.position[1] - self.goal_pos[1])
                print(f"Step {self.steps}: pos={self.position}, dist={dist}, "
                      f"episodes={len(self.episodes)}")
            
            direction, reached_goal = self.navigate_step()
            
            if reached_goal:
                print(f"\n✓ Goal reached in {self.steps} steps!")
                break
        
        success = self.position == self.goal_pos
        
        return {
            'success': success,
            'steps': self.steps,
            'path': self.path,
            'episodes': len(self.episodes),
            'graph_edges': self.episode_graph.number_of_edges()
        }


def test_v2():
    """Test improved geDIG flow"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    
    # 15x15 maze
    maze = create_complex_maze(15, seed=42)
    
    nav = TrueGeDIGFlowNavigatorV2(maze)
    result = nav.navigate(max_steps=500)
    
    print(f"\nFinal statistics:")
    print(f"- Episodes: {result['episodes']}")
    print(f"- Graph edges: {result['graph_edges']}")
    print(f"- Path length: {len(result['path'])}")
    
    if result['success']:
        visualize_maze_with_path(maze, result['path'], 'true_gedig_flow_v2.png')
        print("Saved: true_gedig_flow_v2.png")


if __name__ == "__main__":
    test_v2()