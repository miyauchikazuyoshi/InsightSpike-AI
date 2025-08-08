#!/usr/bin/env python3
"""
Pure Memory Agent with OptimizedNumpyIndex
メインコードの高速検索を活用した純粋記憶エージェント
"""

import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.vector_index.factory import VectorIndexFactory


class PureMemoryAgentOptimized:
    """
    メインコードの最適化された検索を使用する純粋記憶エージェント
    - OptimizedNumpyIndexによる高速類似度検索
    - scalable_graph_builderのgeDIG評価ロジック
    - graph_memory_searchのメッセージパッシング
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
        self.max_depth = self.config.get('max_depth', 5)
        self.search_k = self.config.get('search_k', 30)
        self.gedig_threshold = self.config.get('gedig_threshold', 0.6)
        self.gedig_weight = self.config.get('gedig_weight', 0.3)
        
        # DataStore（永続化用）
        self.datastore = DataStoreFactory.create(
            "filesystem",
            base_path=datastore_path
        )
        
        # 高速ベクトル検索インデックス（メインコード）
        self.vector_index = VectorIndexFactory.create_index(
            dimension=7,
            index_type="numpy",  # OptimizedNumpyIndexを使用
            optimize=True,       # 最適化を有効化
            normalize=True       # 正規化ベクトルのキャッシュ
        )
        
        # グラフ構造（NetworkX）
        self.experience_graph = nx.Graph()
        
        # 空間インデックス（位置ベースの高速アクセス）
        self.spatial_index = defaultdict(list)
        
        # エピソードメタデータ
        self.experience_metadata = []
        
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
            'depth_usage': {i: 0 for i in range(1, self.max_depth+1)},
            'search_times': [],
            'gedig_values': []
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
            vec[2] = 0.5
        
        # 3: 成功フラグ
        if episode_type == 'visual':
            vec[3] = 0.5
        else:
            vec[3] = 1.0 if success else 0.0
        
        # 4: 壁/通路
        vec[4] = -1.0 if is_wall else 1.0
        
        # 5: 訪問回数
        visit_count = self.visit_counts.get((x, y), 0)
        vec[5] = np.log1p(visit_count) / 10.0
        
        # 6: ゴールフラグ
        if success and episode_type == 'movement':
            next_pos = (x + self.action_deltas[direction][0],
                       y + self.action_deltas[direction][1])
            vec[6] = 1.0 if next_pos == self.goal else 0.0
        else:
            vec[6] = 0.0
        
        return vec
    
    def _add_experience(self, vec: np.ndarray, metadata: Dict) -> int:
        """経験を追加し、高速インデックスを更新"""
        # ベクトルインデックスに追加
        idx = len(self.experience_metadata)
        self.vector_index.add(vec.reshape(1, -1))
        
        # メタデータ保存
        self.experience_metadata.append(metadata)
        
        # グラフノード追加
        self.experience_graph.add_node(idx, **metadata)
        
        # 空間インデックス更新
        if 'pos' in metadata:
            pos_key = f"{metadata['pos'][0]},{metadata['pos'][1]}"
            self.spatial_index[pos_key].append(idx)
        
        # geDIG評価でエッジ生成
        if idx > 0:
            self._create_gedig_edges(idx, vec)
        
        return idx
    
    def _calculate_gedig(self, idx1: int, idx2: int, similarity: float) -> float:
        """
        メインコードのscalable_graph_builderロジックに基づくgeDIG計算
        GED (Generalized Edit Distance) - IG (Information Gain)
        """
        meta1 = self.experience_metadata[idx1]
        meta2 = self.experience_metadata[idx2]
        
        # 1. 空間距離（マンハッタン距離）
        if 'pos' in meta1 and 'pos' in meta2:
            pos1 = meta1['pos']
            pos2 = meta2['pos']
            spatial_distance = (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])) / (self.height + self.width)
        else:
            spatial_distance = 0.5
        
        # 2. 時間距離（エピソード順序）
        temporal_distance = abs(idx1 - idx2) / max(100, len(self.experience_metadata))
        
        # 3. タイプの違い
        type_difference = 0
        if meta1.get('type') != meta2.get('type'):
            type_difference = 0.3
        
        # 4. 成功/失敗の差
        outcome_difference = 0
        if 'success' in meta1 and 'success' in meta2:
            if meta1['success'] != meta2['success']:
                outcome_difference = 0.2
        
        # GED: 編集距離の総和
        ged = (spatial_distance * 0.3 + 
               temporal_distance * 0.3 + 
               type_difference * 0.2 + 
               outcome_difference * 0.2)
        
        # IG: 情報利得
        ig = similarity * 0.5
        
        # geDIG = GED - IG（低いほど良い）
        return ged - ig
    
    def _create_gedig_edges(self, new_idx: int, new_vec: np.ndarray):
        """geDIG評価に基づくエッジ生成"""
        if len(self.experience_metadata) <= 1:
            return
        
        # 高速類似度検索（OptimizedNumpyIndex使用）
        start_time = time.time()
        distances, indices = self.vector_index.search(
            new_vec.reshape(1, -1), 
            k=min(20, len(self.experience_metadata) - 1)
        )
        search_time = (time.time() - start_time) * 1000
        self.stats['search_times'].append(search_time)
        
        # geDIG評価でエッジ候補を選定
        edge_candidates = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == new_idx or idx < 0:
                continue
            
            # コサイン類似度（距離ではなく類似度なので注意）
            similarity = dist  # OptimizedNumpyIndexは類似度を返す
            
            if similarity < 0.5:
                continue
            
            # geDIG計算
            gedig_value = self._calculate_gedig(new_idx, idx, similarity)
            self.stats['gedig_values'].append(gedig_value)
            
            if gedig_value < self.gedig_threshold:
                edge_candidates.append({
                    'target_idx': idx,
                    'similarity': similarity,
                    'gedig': gedig_value,
                    'info_value': similarity - gedig_value
                })
        
        # 情報価値でソート
        edge_candidates.sort(key=lambda x: x['info_value'], reverse=True)
        
        # 上位k個のエッジを生成
        max_edges = self.config.get('max_edges_per_node', 10)
        for edge in edge_candidates[:max_edges]:
            self.experience_graph.add_edge(
                new_idx,
                edge['target_idx'],
                weight=edge['similarity'],
                gedig=edge['gedig'],
                info_value=edge['info_value']
            )
    
    def _add_visual_observations(self):
        """現在位置から4方向の視覚観測を追加"""
        x, y = self.position
        
        for direction in self.actions:
            dx, dy = self.action_deltas[direction]
            nx, ny = x + dx, y + dy
            
            is_wall = True
            if 0 <= nx < self.height and 0 <= ny < self.width:
                is_wall = (self.maze[nx, ny] == 1)
            
            vec = self._create_episode_vector(
                x, y, direction, False, is_wall, 'visual'
            )
            
            self._add_experience(vec, {
                'type': 'visual',
                'pos': (x, y),
                'direction': direction,
                'is_wall': is_wall
            })
            
            self.episode_count += 1
    
    def _create_task_query(self) -> np.ndarray:
        """タスク定義クエリ"""
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
    
    def _message_passing(self, start_indices: List[int], depth: int) -> np.ndarray:
        """
        graph_memory_searchロジックに基づくメッセージパッシング
        """
        if not start_indices or depth <= 0:
            return np.zeros(7)
        
        # 初期メッセージ（ランクベース）
        messages = {}
        for rank, idx in enumerate(start_indices[:20]):
            if 0 <= idx < len(self.experience_metadata):
                messages[idx] = 1.0 / (rank + 1)
        
        # グラフ伝播
        for d in range(depth):
            new_messages = {}
            decay = 0.7 ** d  # 距離による減衰
            
            for node_idx, msg_value in messages.items():
                # 自己ループ
                if d < depth - 1:
                    new_messages[node_idx] = msg_value * 0.5 * decay
                
                # 隣接ノードへ伝播
                if node_idx in self.experience_graph:
                    for neighbor_idx in self.experience_graph.neighbors(node_idx):
                        edge_data = self.experience_graph[node_idx][neighbor_idx]
                        
                        # geDIG値による重み付け
                        gedig = edge_data.get('gedig', 1.0)
                        propagation_weight = (1.0 - min(gedig, 1.0)) * decay
                        propagated_value = msg_value * propagation_weight
                        
                        if neighbor_idx in new_messages:
                            new_messages[neighbor_idx] = max(
                                new_messages[neighbor_idx],
                                propagated_value
                            )
                        else:
                            new_messages[neighbor_idx] = propagated_value
            
            messages = new_messages
            if not messages:
                break
        
        # 重み付き集約
        aggregated = np.zeros(7)
        total_weight = 0
        
        for idx, weight in messages.items():
            if idx < len(self.experience_metadata):
                # ベクトルを取得（本来はキャッシュすべき）
                meta = self.experience_metadata[idx]
                vec = self._reconstruct_vector(meta)
                aggregated += vec * weight
                total_weight += weight
        
        if total_weight > 0:
            aggregated = aggregated / total_weight
        
        return aggregated
    
    def _reconstruct_vector(self, metadata: Dict) -> np.ndarray:
        """メタデータからベクトルを再構築（簡易版）"""
        vec = np.zeros(7, dtype=np.float32)
        
        if 'pos' in metadata:
            vec[0] = metadata['pos'][0] / self.height
            vec[1] = metadata['pos'][1] / self.width
        
        if 'direction' in metadata:
            vec[2] = self.action_to_idx.get(metadata['direction'], 2) / 3.0
        
        vec[3] = 1.0 if metadata.get('success', False) else 0.0
        vec[4] = -1.0 if metadata.get('is_wall', False) else 1.0
        
        return vec
    
    def _select_depth_by_gedig(self) -> int:
        """geDIG値に基づく適応的深度選択"""
        if not self.stats['gedig_values']:
            return 1
        
        # 最近のgeDIG値の平均
        recent_gedig = np.mean(self.stats['gedig_values'][-50:])
        
        # geDIG値が低い（良い）ほど深い推論
        if recent_gedig < -0.3:
            return min(5, self.max_depth)
        elif recent_gedig < 0:
            return min(4, self.max_depth)
        elif recent_gedig < 0.3:
            return min(3, self.max_depth)
        elif recent_gedig < 0.5:
            return 2
        else:
            return 1
    
    def get_action(self) -> str:
        """行動を決定"""
        # 訪問回数更新
        if self.position not in self.visit_counts:
            self.visit_counts[self.position] = 0
        self.visit_counts[self.position] += 1
        
        # 視覚観測を追加
        self._add_visual_observations()
        
        # クエリ生成
        query = self._create_task_query()
        
        # 高速類似度検索
        start_time = time.time()
        distances, indices = self.vector_index.search(
            query.reshape(1, -1), 
            k=self.search_k
        )
        search_time = (time.time() - start_time) * 1000
        
        if indices[0][0] < 0:
            return np.random.choice(self.actions)
        
        # geDIG基準で深度選択
        depth = self._select_depth_by_gedig()
        self.stats['depth_usage'][depth] += 1
        
        # メッセージパッシング
        insight = self._message_passing(indices[0].tolist(), depth)
        
        # 方向抽出
        direction_value = insight[2] * 3.0
        direction_idx = int(round(direction_value))
        
        probs = np.ones(4) * 0.15
        if 0 <= direction_idx < 4:
            confidence = max(0.1, min(1.0, insight[3]))
            probs[direction_idx] += 0.5 * confidence
        
        probs = probs / probs.sum()
        
        return np.random.choice(self.actions, p=probs)
    
    def execute_action(self, action: str) -> bool:
        """行動を実行し、結果を記憶"""
        x, y = self.position
        dx, dy = self.action_deltas[action]
        new_x, new_y = x + dx, y + dy
        
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
        
        # 実際の結果を記憶
        self._add_experience(vec, {
            'type': 'movement',
            'pos': (x, y),
            'action': action,
            'success': success,
            'is_wall': is_wall
        })
        
        # DataStoreにもログ
        self.datastore.save_episodes([{
            'text': f"Move from ({x},{y}) {action}: {'success' if success else 'wall'}",
            'vector': vec.tolist(),
            'metadata': {
                'type': 'movement',
                'success': success
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
        avg_search_time = np.mean(self.stats['search_times']) if self.stats['search_times'] else 0
        avg_gedig = np.mean(self.stats['gedig_values']) if self.stats['gedig_values'] else 0
        
        return {
            'position': self.position,
            'goal': self.goal,
            'distance_to_goal': abs(self.position[0] - self.goal[0]) + 
                               abs(self.position[1] - self.goal[1]),
            'steps': self.stats['steps'],
            'wall_hits': self.stats['wall_hits'],
            'wall_hit_rate': self.stats['wall_hits'] / max(1, self.stats['steps']),
            'total_episodes': self.episode_count,
            'depth_usage': self.stats['depth_usage'],
            'avg_search_time_ms': avg_search_time,
            'avg_gedig': avg_gedig,
            'graph_nodes': self.experience_graph.number_of_nodes(),
            'graph_edges': self.experience_graph.number_of_edges()
        }