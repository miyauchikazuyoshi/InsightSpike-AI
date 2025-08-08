#!/usr/bin/env python3
"""
Pure Memory Agent with Goal Beacon Episode
ゴールビーコンエピソードを持つ純粋記憶エージェント

ゴール到達時に「ゴールの位置情報のみ」を持つ特殊なエピソードを生成
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../maze-optimized-search/src'))

from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex
from insightspike.implementations.datastore.factory import DataStoreFactory


class PureMemoryAgentWithGoalBeacon:
    """ゴールビーコンエピソードを持つ純粋記憶エージェント"""
    
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
        self.gedig_improvement_threshold = self.config.get('gedig_improvement_threshold', 0.05)
        
        # DataStore初期化
        self.datastore = DataStoreFactory.create(
            "filesystem",
            base_path=datastore_path
        )
        
        # Index初期化（7次元移動エピソード）
        self.index = GeDIGAwareIntegratedIndex(
            dimension=7,
            config={
                'similarity_threshold': 0.5,
                'gedig_threshold': 0.6,
                'gedig_weight': 0.3,
                'max_edges_per_node': 15
            }
        )
        
        # 記憶システム
        self.visit_counts = {}
        self.episode_id = 0
        self.episodes_metadata = []
        self.goal_beacon_created = False
        
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
            'depth_usage': {i: 0 for i in range(1, self.max_depth+1)},
            'adaptive_depth_selections': [],
            'gedig_evaluations': [],
            'goal_beacon_activations': 0,
            'beacon_search_ranks': []
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
    
    def _create_goal_beacon_episode(self) -> np.ndarray:
        """ゴールビーコンエピソード（位置情報のみ）を生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        # ゴール位置のみを記録
        vec[0] = self.goal[0] / self.height  # ゴールのx座標
        vec[1] = self.goal[1] / self.width   # ゴールのy座標
        vec[2] = 0.5  # 方向は中立（どの方向でも良い）
        vec[3] = 0.5  # 成功/失敗は中立（不明）
        vec[4] = 0.0  # 壁/通路は中立
        vec[5] = 0.0  # 訪問回数は0（未訪問として扱う）
        vec[6] = 1.0  # ゴールフラグ（これがビーコン）
        
        return vec
    
    def _add_goal_beacon(self):
        """ゴールビーコンをメモリに追加"""
        if self.goal_beacon_created:
            return
        
        vec = self._create_goal_beacon_episode()
        
        metadata = {
            'type': 'goal_beacon',
            'position': self.goal,
            'episode_id': self.episode_id,
            'timestamp': time.time(),
            'special': True
        }
        
        # Indexに追加（高いc_valueで重要性を示す）
        idx = self.index.add_episode({
            'vec': vec,
            'text': f"GOAL BEACON at ({self.goal[0]},{self.goal[1]})",
            'pos': self.goal,
            'c_value': 1.0,  # 最高の信頼度
            **metadata
        })
        
        self.episodes_metadata.append(metadata)
        
        # DataStoreに保存
        self.datastore.save_episodes([{
            'text': f"GOAL BEACON at ({self.goal[0]},{self.goal[1]})",
            'vector': vec.tolist(),
            'metadata': metadata
        }])
        
        self.episode_id += 1
        self.goal_beacon_created = True
        
        print(f"  🎯 ゴールビーコン生成！位置: {self.goal}")
    
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
        
        # 6: ゴールフラグ（通常のエピソードでは0）
        vec[6] = 0.0
        
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
    
    def _create_beacon_seeking_query(self) -> np.ndarray:
        """ビーコン探索クエリ（ゴールビーコンを探す）"""
        vec = np.zeros(7, dtype=np.float32)
        
        x, y = self.position
        
        # 現在位置
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # 方向: 中立
        vec[2] = 0.5
        
        # 成功: 中立（ビーコンは成功/失敗の概念がない）
        vec[3] = 0.5
        
        # 壁/通路: 中立
        vec[4] = 0.0
        
        # 訪問回数: 0（未訪問を探す）
        vec[5] = 0.0
        
        # ゴール: 1.0（ビーコンを探す！）
        vec[6] = 1.0
        
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
                'timestamp': time.time()
            }
            
            # Indexに追加
            idx = self.index.add_episode({
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
    
    def _evaluate_gedig_at_depth(self, indices: List[int], depth: int) -> float:
        """指定深度でのgeDIG値を評価"""
        if depth <= 0 or not indices:
            return float('inf')
        
        messages = {}
        for i, idx in enumerate(indices[:20]):
            messages[idx] = 1.0 / (i + 1)
        
        graph = self.index.graph
        total_gedig = 0
        gedig_count = 0
        
        for d in range(depth):
            new_messages = {}
            decay = 0.9 ** d
            
            for node, value in messages.items():
                if d < depth - 1:
                    new_messages[node] = value * 0.8 * decay
                
                if node in graph:
                    for neighbor in graph.neighbors(node):
                        edge_data = graph[node][neighbor]
                        gedig = edge_data.get('gedig', 1.0)
                        
                        total_gedig += gedig
                        gedig_count += 1
                        
                        weight = edge_data.get('weight', 0.5)
                        propagation = value * weight * decay
                        
                        if neighbor not in new_messages:
                            new_messages[neighbor] = propagation
                        else:
                            new_messages[neighbor] = max(
                                new_messages[neighbor],
                                propagation
                            )
            
            messages = new_messages
            if not messages:
                break
        
        if gedig_count > 0:
            return total_gedig / gedig_count
        return float('inf')
    
    def _adaptive_depth_selection(self, indices: List[int]) -> int:
        """geDIG値に基づいて最適な深度を選択"""
        if not indices:
            return 1
        
        base_gedig = self._evaluate_gedig_at_depth(indices, 1)
        optimal_depth = 1
        gedig_history = [(1, base_gedig)]
        
        for depth in range(2, self.max_depth + 1):
            current_gedig = self._evaluate_gedig_at_depth(indices, depth)
            gedig_history.append((depth, current_gedig))
            
            improvement = (base_gedig - current_gedig) / (base_gedig + 0.001)
            
            if improvement > self.gedig_improvement_threshold:
                optimal_depth = depth
                base_gedig = current_gedig
            else:
                break
        
        self.stats['adaptive_depth_selections'].append(optimal_depth)
        self.stats['gedig_evaluations'].append(gedig_history)
        
        return optimal_depth
    
    def _beacon_guided_message_passing(self, indices: List[int], depth: int) -> np.ndarray:
        """ビーコンガイド付きメッセージパッシング"""
        if depth <= 0 or not indices:
            return np.zeros(7)
        
        # ビーコンエピソードをチェック
        beacon_boost = 1.0
        for idx in indices[:10]:
            if idx < len(self.index.metadata):
                meta = self.index.metadata[idx]
                if meta.get('type') == 'goal_beacon':
                    beacon_boost = 2.0  # ビーコンが見つかったらブースト
                    self.stats['goal_beacon_activations'] += 1
                    break
        
        # 初期メッセージ
        messages = {}
        for i, idx in enumerate(indices[:20]):
            weight = 1.0 / (i + 1)
            # ビーコンエピソードは強い初期重み
            if idx < len(self.index.metadata):
                if self.index.metadata[idx].get('type') == 'goal_beacon':
                    weight *= beacon_boost
            messages[idx] = weight
        
        graph = self.index.graph
        
        # 指定深度まで伝播
        for d in range(depth):
            new_messages = {}
            decay = 0.9 ** d
            
            for node, value in messages.items():
                if d < depth - 1:
                    new_messages[node] = value * 0.8 * decay
                
                if node in graph:
                    for neighbor, edge_data in graph[node].items():
                        weight = edge_data.get('weight', 0.5)
                        gedig = edge_data.get('gedig', 1.0)
                        
                        quality = weight * (1.0 - gedig * 0.3)
                        propagation = value * quality * decay
                        
                        if neighbor not in new_messages:
                            new_messages[neighbor] = propagation
                        else:
                            new_messages[neighbor] = max(
                                new_messages[neighbor],
                                propagation
                            )
            
            messages = new_messages
            if not messages:
                break
        
        # 集約
        direction = np.zeros(7)
        total_weight = 0
        
        for idx, value in messages.items():
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                vec = episode['vec']
                
                # ビーコンエピソードの影響を強化
                if episode.get('type') == 'goal_beacon':
                    weight = value * beacon_boost
                else:
                    weight = value
                
                direction += vec * weight
                total_weight += weight
        
        if total_weight > 0:
            direction = direction / total_weight
        
        return direction
    
    def get_action(self) -> str:
        """行動を決定"""
        # 訪問回数更新
        if self.position not in self.visit_counts:
            self.visit_counts[self.position] = 0
        self.visit_counts[self.position] += 1
        
        # 視覚観測を追加
        self._add_visual_observations()
        
        # ビーコン探索クエリ
        query = self._create_beacon_seeking_query()
        
        # 検索
        start_time = time.time()
        indices, scores = self.index.search(query, k=self.search_k, mode='hybrid')
        
        # ビーコンの順位を記録
        for rank, idx in enumerate(indices[:10]):
            if idx < len(self.index.metadata):
                if self.index.metadata[idx].get('type') == 'goal_beacon':
                    self.stats['beacon_search_ranks'].append(rank + 1)
                    break
        
        if len(indices) == 0:
            return np.random.choice(self.actions)
        
        # geDIG適応的深度選択
        optimal_depth = self._adaptive_depth_selection(indices.tolist())
        self.stats['depth_usage'][optimal_depth] += 1
        
        # ビーコンガイド付きメッセージパッシング
        insight = self._beacon_guided_message_passing(indices.tolist(), optimal_depth)
        
        search_time = (time.time() - start_time) * 1000
        self.stats['search_times'].append(search_time)
        
        # 方向成分を抽出
        direction_value = insight[2] * 3.0
        direction_idx = int(round(direction_value))
        
        # 確率分布に変換
        probs = np.ones(4) * 0.15
        
        if 0 <= direction_idx < 4:
            confidence = max(0.1, min(1.0, insight[3]))
            probs[direction_idx] += 0.4 * confidence
        
        probs = probs / probs.sum()
        
        return np.random.choice(self.actions, p=probs)
    
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
                
                # ゴール到達時にビーコン生成
                if self.position == self.goal and not self.goal_beacon_created:
                    self._add_goal_beacon()
        
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
            'timestamp': time.time()
        }
        
        # Indexに追加
        idx = self.index.add_episode({
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
        avg_depth = 0
        if self.stats['adaptive_depth_selections']:
            avg_depth = np.mean(self.stats['adaptive_depth_selections'])
        
        avg_beacon_rank = 0
        if self.stats['beacon_search_ranks']:
            avg_beacon_rank = np.mean(self.stats['beacon_search_ranks'])
        
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
                              if self.stats['search_times'] else 0,
            'avg_adaptive_depth': avg_depth,
            'goal_beacon_created': self.goal_beacon_created,
            'goal_beacon_activations': self.stats['goal_beacon_activations'],
            'avg_beacon_rank': avg_beacon_rank
        }
    
    def is_goal_reached(self) -> bool:
        """ゴールに到達したか確認"""
        return self.position == self.goal