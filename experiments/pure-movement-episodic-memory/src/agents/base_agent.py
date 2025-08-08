#!/usr/bin/env python3
"""
基底エージェントクラス
全ての純粋記憶エージェントの共通インターフェース
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../maze-optimized-search/src'))

from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex
from insightspike.implementations.datastore.factory import DataStoreFactory


class BaseMemoryAgent(ABC):
    """純粋記憶エージェントの基底クラス"""
    
    def __init__(self, 
                 maze: np.ndarray,
                 datastore_path: str,
                 config: Optional[Dict] = None):
        """
        Args:
            maze: 迷路配列
            datastore_path: DataStore保存パス
            config: 設定パラメータ
        """
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        
        # 設定
        self.config = config or {}
        self.max_depth = self.config.get('max_depth', 5)
        self.search_k = self.config.get('search_k', 30)
        
        # DataStore初期化
        self.datastore = DataStoreFactory.create(
            "filesystem",
            base_path=datastore_path
        )
        
        # Index初期化（7次元移動エピソード）
        self.index = GeDIGAwareIntegratedIndex(
            dimension=7,
            config=self.config.get('gedig_config', {
                'similarity_threshold': 0.5,
                'gedig_threshold': 0.6,
                'gedig_weight': 0.3,
                'max_edges_per_node': 15
            })
        )
        
        # 記憶システム
        self.visit_counts = {}
        self.episode_id = 0
        self.episodes_metadata = []
        
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
            'wall_hits': 0,
            'path': [self.position],
            'search_times': [],
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
    
    def _create_movement_episode(self, 
                                 x: int, y: int, 
                                 direction: str,
                                 success: bool, 
                                 is_wall: bool) -> np.ndarray:
        """7次元移動エピソードベクトルを生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        # 0-1: 位置（正規化）
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # 2: 移動方向（正規化）
        vec[2] = self.action_to_idx[direction] / 3.0
        
        # 3: 成功フラグ
        vec[3] = 1.0 if success else 0.0
        
        # 4: 壁/通路
        vec[4] = -1.0 if is_wall else 1.0
        
        # 5: 訪問回数（対数正規化）
        visit_count = self.visit_counts.get((x, y), 0)
        vec[5] = np.log1p(visit_count) / 10.0
        
        # 6: ゴールフラグ
        is_goal = (success and (x + self.action_deltas[direction][0], 
                               y + self.action_deltas[direction][1]) == self.goal)
        vec[6] = 1.0 if is_goal else 0.0
        
        return vec
    
    def _create_visual_episode(self, x: int, y: int, direction: str) -> np.ndarray:
        """視覚観測エピソードを生成"""
        dx, dy = self.action_deltas[direction]
        nx, ny = x + dx, y + dy
        
        # 壁チェック
        is_wall = True
        if 0 <= nx < self.height and 0 <= ny < self.width:
            is_wall = (self.maze[nx, ny] == 1)
        
        vec = np.zeros(7, dtype=np.float32)
        vec[0] = x / self.height
        vec[1] = y / self.width
        vec[2] = self.action_to_idx[direction] / 3.0
        vec[3] = 0.5  # 未実行
        vec[4] = -1.0 if is_wall else 1.0
        vec[5] = np.log1p(self.visit_counts.get((x, y), 0)) / 10.0
        vec[6] = 0.0  # 視覚エピソードはゴールフラグなし
        
        return vec
    
    def _add_visual_observations(self):
        """4方向の視覚エピソードを追加"""
        x, y = self.position
        
        for direction in self.actions:
            vec = self._create_visual_episode(x, y, direction)
            
            metadata = {
                'type': 'visual',
                'position': (x, y),
                'direction': direction,
                'episode_id': self.episode_id,
                'timestamp': 0  # サブクラスでtime.time()を使用
            }
            
            # Indexに追加
            self.index.add_episode({
                'vec': vec,
                'text': f"Visual observation from ({x},{y}) looking {direction}",
                'pos': (x, y),
                'c_value': 0.5,
                **metadata
            })
            self.episodes_metadata.append(metadata)
            
            # DataStoreに保存
            self.datastore.save_episodes([{
                'text': f"Visual observation from ({x},{y}) looking {direction}",
                'vector': vec.tolist(),
                'metadata': metadata
            }])
            
            self.episode_id += 1
    
    @abstractmethod
    def _create_query(self) -> np.ndarray:
        """クエリベクトルを生成（サブクラスで実装）"""
        pass
    
    @abstractmethod
    def _message_passing(self, indices: List[int], depth: int) -> np.ndarray:
        """メッセージパッシング（サブクラスで実装）"""
        pass
    
    @abstractmethod
    def get_action(self) -> str:
        """行動を決定（サブクラスで実装）"""
        pass
    
    def execute_action(self, action: str) -> bool:
        """行動を実行し、移動エピソードを記録"""
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
        
        # 移動エピソードを記録
        vec = self._create_movement_episode(x, y, action, success, is_wall)
        
        metadata = {
            'type': 'movement',
            'position': (x, y),
            'action': action,
            'success': success,
            'is_wall': is_wall,
            'episode_id': self.episode_id,
            'timestamp': 0  # サブクラスでtime.time()を使用
        }
        
        # Indexに追加
        self.index.add_episode({
            'vec': vec,
            'text': f"Movement from ({x},{y}) {action} {'success' if success else 'failed'}",
            'pos': (x, y),
            'c_value': 0.8 if success else 0.3,
            **metadata
        })
        self.episodes_metadata.append(metadata)
        
        # DataStoreに保存
        self.datastore.save_episodes([{
            'text': f"Movement from ({x},{y}) {action} {'success' if success else 'failed'}",
            'vector': vec.tolist(),
            'metadata': metadata
        }])
        
        self.episode_id += 1
        
        return success
    
    def get_statistics(self) -> Dict:
        """現在の統計情報を取得"""
        return {
            'position': self.position,
            'goal': self.goal,
            'distance_to_goal': abs(self.position[0] - self.goal[0]) + 
                               abs(self.position[1] - self.goal[1]),
            'total_episodes': self.episode_id,
            'wall_hits': self.stats['wall_hits'],
            'path_length': len(self.stats['path']),
            'visit_counts': self.visit_counts,
            'depth_usage': self.stats['depth_usage'],
            'avg_search_time': np.mean(self.stats['search_times']) 
                              if self.stats['search_times'] else 0
        }
    
    def is_goal_reached(self) -> bool:
        """ゴールに到達したか確認"""
        return self.position == self.goal
    
    def reset(self, keep_memory: bool = False):
        """エージェントをリセット"""
        self.position = self._find_start()
        self.stats['path'] = [self.position]
        self.stats['wall_hits'] = 0
        
        if not keep_memory:
            self.visit_counts = {}
            self.episode_id = 0
            self.episodes_metadata = []
            self.stats['search_times'] = []
            self.stats['depth_usage'] = {i: 0 for i in range(1, self.max_depth+1)}