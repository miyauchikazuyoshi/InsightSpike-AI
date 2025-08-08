#!/usr/bin/env python3
"""
Pure Memory Agent Final Version
純粋geDIG評価による記憶駆動エージェント
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.implementations.datastore.factory import DataStoreFactory
from pure_gedig_index import PureGeDIGIndex


class PureMemoryAgentFinal:
    """
    最終版：純粋記憶エージェント
    - 推論結果は破棄
    - 実際の経験のみ記憶
    - 純粋なgeDIG評価
    """
    
    def __init__(self, 
                 maze: np.ndarray,
                 datastore_path: str,
                 config: Optional[Dict] = None):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        
        # 設定
        self.config = config or {}
        self.max_depth = self.config.get('max_depth', 3)
        self.search_k = self.config.get('search_k', 20)
        
        # DataStore（ログ用）
        self.datastore = DataStoreFactory.create(
            "filesystem",
            base_path=datastore_path
        )
        
        # 純粋geDIGインデックス
        self.memory = PureGeDIGIndex(
            dimension=7,
            config={
                'similarity_threshold': 0.5,
                'gedig_threshold': 0.6,
                'gedig_weight': 0.3,
                'max_edges_per_node': 10
            }
        )
        
        # 訪問記録
        self.visit_counts = {}
        self.episode_count = 0
        
        # 行動マッピング
        self.actions = ['up', 'right', 'down', 'left']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # 統計
        self.stats = {
            'steps': 0,
            'wall_hits': 0,
            'path': [self.position],
            'inference_times': [],
            'depth_usage': {i: 0 for i in range(1, self.max_depth+1)}
        }
    
    def _find_start(self) -> Tuple[int, int]:
        """スタート位置を検索"""
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (1, 1)
    
    def _find_goal(self) -> Tuple[int, int]:
        """ゴール位置を検索"""
        for i in range(self.height-1, -1, -1):
            for j in range(self.width-1, -1, -1):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (self.height-2, self.width-2)
    
    def _create_episode_vector(self, x: int, y: int, direction: str, 
                               success: bool, is_wall: bool,
                               episode_type: str = 'movement') -> np.ndarray:
        """7次元エピソードベクトル生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        # 0-1: 位置（正規化）
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # 2: 方向
        if direction:
            vec[2] = self.action_to_idx[direction] / 3.0
        else:
            vec[2] = 0.5  # 中立
        
        # 3: 成功フラグ
        if episode_type == 'visual':
            vec[3] = 0.5  # 視覚は中立
        else:
            vec[3] = 1.0 if success else 0.0
        
        # 4: 壁/通路
        vec[4] = -1.0 if is_wall else 1.0
        
        # 5: 訪問回数
        visit_count = self.visit_counts.get((x, y), 0)
        vec[5] = np.log1p(visit_count) / 10.0
        
        # 6: ゴールフラグ（ゴール到達時のみ1.0）
        if success and episode_type == 'movement':
            next_pos = (x + self.action_deltas[direction][0],
                       y + self.action_deltas[direction][1])
            vec[6] = 1.0 if next_pos == self.goal else 0.0
        else:
            vec[6] = 0.0
        
        return vec
    
    def _add_visual_observations(self):
        """現在位置から4方向の視覚観測を追加"""
        x, y = self.position
        
        for direction in self.actions:
            dx, dy = self.action_deltas[direction]
            nx, ny = x + dx, y + dy
            
            # 壁チェック
            is_wall = True
            if 0 <= nx < self.height and 0 <= ny < self.width:
                is_wall = (self.maze[nx, ny] == 1)
            
            # エピソードベクトル生成
            vec = self._create_episode_vector(
                x, y, direction, False, is_wall, 'visual'
            )
            
            # 記憶に追加（実際の観測）
            self.memory.add_experience({
                'vec': vec,
                'type': 'visual',
                'pos': (x, y),
                'direction': direction,
                'is_wall': is_wall
            })
            
            # DataStoreにもログ
            self.datastore.save_episodes([{
                'text': f"Visual from ({x},{y}) {direction}: {'wall' if is_wall else 'path'}",
                'vector': vec.tolist(),
                'metadata': {'type': 'visual', 'pos': (x, y)}
            }])
            
            self.episode_count += 1
    
    def _create_query_vector(self) -> np.ndarray:
        """純粋クエリベクトル（ゴール指向）"""
        vec = np.zeros(7, dtype=np.float32)
        
        x, y = self.position
        vec[0] = x / self.height
        vec[1] = y / self.width
        vec[2] = 0.5  # 方向中立
        vec[3] = 1.0  # 成功を求める
        vec[4] = 1.0  # 通路を好む
        vec[5] = 0.0  # 未訪問を探索
        vec[6] = 1.0  # ゴール関連を優先
        
        return vec
    
    def _select_adaptive_depth(self, indices: List[int]) -> int:
        """簡易的な適応的深度選択"""
        if not indices:
            return 1
        
        # エピソード数に応じて深度を調整
        num_experiences = len(self.memory.experience_vectors)
        
        if num_experiences < 100:
            return 1
        elif num_experiences < 500:
            return min(2, self.max_depth)
        else:
            # 多くの経験がある場合は深い推論
            return min(3, self.max_depth)
    
    def get_action(self) -> str:
        """行動を決定（推論は一時的）"""
        # 訪問回数更新
        if self.position not in self.visit_counts:
            self.visit_counts[self.position] = 0
        self.visit_counts[self.position] += 1
        
        # 視覚観測を追加
        self._add_visual_observations()
        
        # クエリ生成
        query = self._create_query_vector()
        
        # 記憶検索
        start_time = time.time()
        indices, scores = self.memory.search_experiences(query, k=self.search_k)
        
        if not indices:
            # 記憶がない場合はランダム
            return np.random.choice(self.actions)
        
        # 適応的深度選択
        depth = self._select_adaptive_depth(indices)
        self.stats['depth_usage'][depth] += 1
        
        # メッセージパッシング（結果は破棄される）
        insight = self.memory.pure_message_passing(indices, depth)
        
        inference_time = (time.time() - start_time) * 1000
        self.stats['inference_times'].append(inference_time)
        
        # 洞察から方向を抽出
        direction_value = insight[2] * 3.0
        direction_idx = int(round(direction_value))
        
        # 確率分布に変換
        probs = np.ones(4) * 0.15
        if 0 <= direction_idx < 4:
            confidence = max(0.1, min(1.0, insight[3]))
            probs[direction_idx] += 0.5 * confidence
        
        # 正規化
        probs = probs / probs.sum()
        
        # 行動選択
        return np.random.choice(self.actions, p=probs)
    
    def execute_action(self, action: str) -> bool:
        """行動を実行し、結果を記憶"""
        x, y = self.position
        dx, dy = self.action_deltas[action]
        new_x, new_y = x + dx, y + dy
        
        # 移動試行
        success = False
        is_wall = True
        
        if 0 <= new_x < self.height and 0 <= new_y < self.width:
            if self.maze[new_x, new_y] == 0:
                self.position = (new_x, new_y)
                self.stats['path'].append(self.position)
                success = True
                is_wall = False
        
        if not success:
            self.stats['wall_hits'] += 1
        
        # 移動エピソードベクトル
        vec = self._create_episode_vector(
            x, y, action, success, is_wall, 'movement'
        )
        
        # 実際の結果を記憶（これは永続的）
        self.memory.add_experience({
            'vec': vec,
            'type': 'movement',
            'pos': (x, y),
            'action': action,
            'success': success,
            'is_wall': is_wall
        })
        
        # DataStoreにログ
        self.datastore.save_episodes([{
            'text': f"Move from ({x},{y}) {action}: {'success' if success else 'wall'}",
            'vector': vec.tolist(),
            'metadata': {
                'type': 'movement',
                'success': success,
                'pos': (x, y)
            }
        }])
        
        self.episode_count += 1
        self.stats['steps'] += 1
        
        return success
    
    def is_goal_reached(self) -> bool:
        """ゴール到達確認"""
        return self.position == self.goal
    
    def get_statistics(self) -> Dict:
        """統計情報取得"""
        memory_stats = self.memory.get_statistics()
        
        return {
            'position': self.position,
            'goal': self.goal,
            'distance_to_goal': abs(self.position[0] - self.goal[0]) + 
                               abs(self.position[1] - self.goal[1]),
            'steps': self.stats['steps'],
            'wall_hits': self.stats['wall_hits'],
            'wall_hit_rate': self.stats['wall_hits'] / max(1, self.stats['steps']),
            'path_length': len(self.stats['path']),
            'total_episodes': self.episode_count,
            'depth_usage': self.stats['depth_usage'],
            'avg_inference_time': np.mean(self.stats['inference_times']) 
                                 if self.stats['inference_times'] else 0,
            'memory_stats': memory_stats
        }