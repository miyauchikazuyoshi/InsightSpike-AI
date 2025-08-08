#!/usr/bin/env python3
"""
DataStore + geDIGエッジ統合版
エピソードを保存し、geDIG評価でエッジを張る
"""

import numpy as np
import json
import time
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.episode import Episode
from insightspike.vector_index.factory import VectorIndexFactory


class MainAgentWithGedigEdges:
    """DataStore + geDIGエッジ評価を統合したエージェント"""
    
    def __init__(self, maze: np.ndarray, datastore_path: str = "data/maze_gedig", 
                 config: Optional[Dict] = None):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.height - 2, self.width - 2)
        
        # 設定
        self.config = config or {}
        self.max_edges_per_node = self.config.get('max_edges_per_node', 7)  # マジカルナンバー
        self.gedig_threshold = self.config.get('gedig_threshold', 0.5)
        self.max_depth = self.config.get('max_depth', 20)
        self.search_k = self.config.get('search_k', 50)
        
        # 行動定義
        self.actions = ['up', 'right', 'down', 'left']
        self.action_deltas = {
            'up': (-1, 0), 'right': (0, 1),
            'down': (1, 0), 'left': (0, -1)
        }
        
        # DataStore作成
        self.datastore = DataStoreFactory.create("filesystem", base_path=datastore_path)
        
        # ベクトルインデックス（7次元エピソードベクトル用）
        self.vector_index = VectorIndexFactory.create_index(
            dimension=7,
            index_type="numpy",
            optimize=True,
            normalize=True
        )
        
        # グラフ構造（エピソード間のエッジ）
        self.episode_graph = nx.DiGraph()
        
        # エピソードメタデータ（メモリ内キャッシュ）
        self.episodes = []
        
        # 統計
        self.steps = 0
        self.wall_hits = 0
        self.gedig_values = []
        
        print(f"✅ DataStore + geDIGエッジ統合版初期化")
        print(f"  DataStore: {datastore_path}")
        print(f"  max_edges_per_node: {self.max_edges_per_node}")
        print(f"  gedig_threshold: {self.gedig_threshold}")
        
        # 既存エピソードを読み込み
        self._load_existing_episodes()
    
    def _load_existing_episodes(self):
        """既存のエピソードを読み込み"""
        existing = self.datastore.list_episodes()
        if existing:
            print(f"  既存エピソード数: {len(existing)}")
            # TODO: 既存エピソードからグラフを再構築
    
    def _create_episode_vector(self, episode_type: str, **kwargs) -> np.ndarray:
        """エピソードを7次元ベクトルに変換"""
        vec = np.zeros(7, dtype=np.float32)
        
        x, y = self.position
        vec[0] = x / self.height  # 正規化位置X
        vec[1] = y / self.width   # 正規化位置Y
        
        if episode_type == 'visual':
            # 方向エンコーディング
            direction = kwargs.get('direction', 'up')
            direction_map = {'up': 0, 'right': 0.33, 'down': 0.66, 'left': 1.0}
            vec[2] = direction_map.get(direction, 0.5)
            
            # 壁情報
            vec[4] = -1.0 if kwargs.get('is_wall', False) else 1.0
            
            # タイプマーカー
            vec[6] = 0.0  # 視覚
            
        elif episode_type == 'movement':
            # 行動エンコーディング
            action = kwargs.get('action', 'up')
            action_map = {'up': 0, 'right': 0.33, 'down': 0.66, 'left': 1.0}
            vec[2] = action_map.get(action, 0.5)
            
            # 成功/失敗
            vec[3] = 1.0 if kwargs.get('success', False) else 0.0
            vec[4] = 1.0 if kwargs.get('success', False) else -1.0
            
            # タイプマーカー
            vec[6] = 1.0  # 移動
        
        # ゴールへの距離（正規化）
        distance = abs(x - self.goal[0]) + abs(y - self.goal[1])
        max_distance = self.height + self.width
        vec[5] = 1.0 - (distance / max_distance)
        
        return vec
    
    def _calculate_gedig(self, vec1: np.ndarray, vec2: np.ndarray, 
                        meta1: Dict, meta2: Dict) -> float:
        """
        geDIG評価（Generalized Edit Distance - Information Gain）
        """
        # 位置的距離
        pos1 = meta1.get('position', [0, 0])
        pos2 = meta2.get('position', [0, 0])
        spatial_distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # 時間的距離
        step1 = meta1.get('step', 0)
        step2 = meta2.get('step', 0)
        temporal_distance = abs(step1 - step2)
        
        # タイプの違い
        type1 = meta1.get('type', '')
        type2 = meta2.get('type', '')
        type_difference = 0 if type1 == type2 else 1
        
        # 成功/失敗の違い
        success1 = meta1.get('success', None)
        success2 = meta2.get('success', None)
        outcome_difference = 0
        if success1 is not None and success2 is not None:
            outcome_difference = 0 if success1 == success2 else 1
        
        # GED: 編集距離
        ged = (spatial_distance * 0.3 + 
               temporal_distance * 0.001 +  # 時間の影響を小さく
               type_difference * 0.2 + 
               outcome_difference * 0.2)
        
        # IG: 情報利得（コサイン類似度）
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        ig = max(0, similarity) * 0.5
        
        # geDIG = GED - IG
        gedig = ged - ig
        
        return gedig
    
    def _create_gedig_edges(self, new_idx: int, new_vec: np.ndarray, new_meta: Dict):
        """
        新しいエピソードに対してgeDIG評価でエッジを張る
        """
        if len(self.episodes) <= 1:
            return
        
        # ベクトル検索で類似エピソードを取得
        if self.vector_index.total_vectors > 0:
            distances, indices = self.vector_index.search(
                new_vec.reshape(1, -1),
                k=min(self.search_k, len(self.episodes) - 1)
            )
            
            # geDIG評価でエッジ候補を選定
            edge_candidates = []
            
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.episodes) or idx == new_idx:
                    continue
                
                target_meta = self.episodes[idx]['metadata']
                target_vec = self.episodes[idx]['vector']
                
                # geDIG計算
                gedig_value = self._calculate_gedig(
                    new_vec, target_vec, new_meta, target_meta
                )
                
                self.gedig_values.append(gedig_value)
                
                # 閾値以下なら候補に追加
                if gedig_value < self.gedig_threshold:
                    edge_candidates.append({
                        'target_idx': idx,
                        'gedig': gedig_value,
                        'similarity': 1.0 - dist  # 距離を類似度に変換
                    })
            
            # geDIG値でソート（低いほど良い）
            edge_candidates.sort(key=lambda x: x['gedig'])
            
            # 上位k個のエッジを生成
            for edge in edge_candidates[:self.max_edges_per_node]:
                self.episode_graph.add_edge(
                    new_idx,
                    edge['target_idx'],
                    weight=1.0 - edge['gedig'],  # エッジの重みは逆数
                    gedig=edge['gedig'],
                    similarity=edge['similarity']
                )
                
                if self.steps < 3:  # デバッグ出力
                    print(f"    エッジ: {new_idx} → {edge['target_idx']}, "
                          f"geDIG={edge['gedig']:.3f}")
    
    def add_episode(self, text: str, episode_type: str, **kwargs) -> str:
        """
        エピソードをDataStoreに保存し、geDIGエッジを張る
        """
        # メタデータ作成
        metadata = {
            'type': episode_type,
            'position': list(self.position),
            'step': self.steps,
            **kwargs
        }
        
        # ベクトル作成
        vec = self._create_episode_vector(episode_type, **kwargs)
        
        # エピソード作成
        episode = Episode(
            text=text,
            timestamp=time.time(),
            metadata=metadata
        )
        
        # DataStoreに保存
        episode_id = self.datastore.store_episode(episode)
        
        # インデックスを決定
        idx = len(self.episodes)
        
        # メモリ内キャッシュに追加
        self.episodes.append({
            'id': episode_id,
            'text': text,
            'metadata': metadata,
            'vector': vec
        })
        
        # ベクトルインデックスに追加
        self.vector_index.add(vec.reshape(1, -1))
        
        # グラフノード追加
        self.episode_graph.add_node(idx, **metadata)
        
        # geDIGエッジを生成（重要！）
        self._create_gedig_edges(idx, vec, metadata)
        
        return episode_id
    
    def add_visual_observations(self):
        """視覚観測をエピソードとして追加"""
        x, y = self.position
        
        for direction in self.actions:
            dx, dy = self.action_deltas[direction]
            nx, ny = x + dx, y + dy
            
            is_wall = True
            if 0 <= nx < self.height and 0 <= ny < self.width:
                is_wall = (self.maze[nx, ny] == 1)
            
            text = f"At ({x},{y}) looking {direction}: {'wall' if is_wall else 'passage'}"
            
            self.add_episode(
                text=text,
                episode_type='visual',
                direction=direction,
                is_wall=is_wall
            )
    
    def add_movement_episode(self, action: str, success: bool):
        """移動エピソードを追加"""
        x, y = self.position
        text = f"From ({x},{y}) moved {action}: {'success' if success else 'hit wall'}"
        
        self.add_episode(
            text=text,
            episode_type='movement',
            action=action,
            success=success
        )
    
    def _message_passing(self, start_indices: List[int], depth: int) -> np.ndarray:
        """
        グラフ上でメッセージパッシング
        """
        if not start_indices or depth <= 0:
            return np.zeros(7)
        
        # 初期メッセージ
        messages = {}
        for rank, idx in enumerate(start_indices[:20]):
            if 0 <= idx < len(self.episodes):
                messages[idx] = 1.0 / (rank + 1)
        
        # グラフ伝播
        for d in range(depth):
            new_messages = {}
            decay = 0.7 ** d
            
            for node_idx, msg_value in messages.items():
                # 自己ループ
                if d < depth - 1:
                    new_messages[node_idx] = msg_value * 0.5 * decay
                
                # 隣接ノードへ伝播
                if self.episode_graph.has_node(node_idx):
                    for neighbor_idx in self.episode_graph.neighbors(node_idx):
                        edge_data = self.episode_graph[node_idx][neighbor_idx]
                        
                        # エッジの重みで伝播
                        weight = edge_data.get('weight', 1.0)
                        propagated = msg_value * weight * decay
                        
                        if neighbor_idx in new_messages:
                            new_messages[neighbor_idx] = max(
                                new_messages[neighbor_idx], propagated
                            )
                        else:
                            new_messages[neighbor_idx] = propagated
            
            messages = new_messages
            if not messages:
                break
        
        # 重み付き集約
        aggregated = np.zeros(7)
        total_weight = 0
        
        for idx, weight in messages.items():
            if idx < len(self.episodes):
                vec = self.episodes[idx]['vector']
                aggregated += vec * weight
                total_weight += weight
        
        if total_weight > 0:
            aggregated = aggregated / total_weight
        
        return aggregated
    
    def get_action(self) -> str:
        """行動決定（geDIGグラフとメッセージパッシング使用）"""
        # 視覚観測
        self.add_visual_observations()
        
        # クエリベクトル作成
        query_vec = np.zeros(7, dtype=np.float32)
        x, y = self.position
        query_vec[0] = x / self.height
        query_vec[1] = y / self.width
        query_vec[2] = 0.5  # 方向中立
        query_vec[3] = 1.0  # 成功を求める
        query_vec[4] = 1.0  # 通路を好む
        query_vec[5] = 1.0 - (abs(x - self.goal[0]) + abs(y - self.goal[1])) / (self.height + self.width)
        query_vec[6] = 0.5  # タイプ中立
        
        # 類似エピソード検索
        if self.vector_index.total_vectors > 0:
            distances, indices = self.vector_index.search(
                query_vec.reshape(1, -1),
                k=min(30, self.vector_index.total_vectors)
            )
            
            # 深度を適応的に選択
            if self.gedig_values:
                recent_gedig = np.mean(self.gedig_values[-50:])
                if recent_gedig < -0.3:
                    depth = min(5, self.max_depth)
                elif recent_gedig < 0:
                    depth = min(4, self.max_depth)
                elif recent_gedig < 0.3:
                    depth = min(3, self.max_depth)
                else:
                    depth = 2
            else:
                depth = 3
            
            # メッセージパッシング
            aggregated = self._message_passing(indices[0].tolist(), depth)
            
            # 最も近いエピソードの行動を選択
            best_action = None
            best_score = -999
            
            for idx in indices[0][:10]:
                if 0 <= idx < len(self.episodes):
                    ep_meta = self.episodes[idx]['metadata']
                    
                    if ep_meta.get('type') == 'movement' and ep_meta.get('success'):
                        action = ep_meta.get('action')
                        if action:
                            # エピソードベクトルとの類似度
                            ep_vec = self.episodes[idx]['vector']
                            score = np.dot(aggregated, ep_vec)
                            
                            if score > best_score:
                                best_score = score
                                best_action = action
            
            if best_action:
                return best_action
        
        # フォールバック：壁がない方向
        safe_directions = []
        for idx in range(len(self.episodes) - 4, len(self.episodes)):
            if idx >= 0:
                ep_meta = self.episodes[idx]['metadata']
                if ep_meta.get('type') == 'visual' and not ep_meta.get('is_wall'):
                    direction = ep_meta.get('direction')
                    if direction:
                        safe_directions.append(direction)
        
        if safe_directions:
            return np.random.choice(safe_directions)
        
        return np.random.choice(self.actions)
    
    def execute_action(self, action: str) -> bool:
        """行動実行"""
        dx, dy = self.action_deltas[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        success = False
        if (0 <= new_x < self.height and 
            0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            self.position = (new_x, new_y)
            success = True
        else:
            self.wall_hits += 1
        
        # 移動エピソードを追加
        self.add_movement_episode(action, success)
        
        self.steps += 1
        return success
    
    def is_goal_reached(self) -> bool:
        return self.position == self.goal
    
    def get_statistics(self) -> Dict:
        """統計情報"""
        distance = abs(self.position[0] - self.goal[0]) + \
                  abs(self.position[1] - self.goal[1])
        
        avg_gedig = np.mean(self.gedig_values) if self.gedig_values else 0
        
        return {
            'steps': self.steps,
            'wall_hits': self.wall_hits,
            'wall_hit_rate': self.wall_hits / max(1, self.steps),
            'distance_to_goal': distance,
            'episode_count': len(self.episodes),
            'graph_nodes': self.episode_graph.number_of_nodes(),
            'graph_edges': self.episode_graph.number_of_edges(),
            'avg_gedig': avg_gedig
        }


def test_gedig_integration():
    """geDIG統合版のテスト"""
    print("="*70)
    print("🧪 DataStore + geDIGエッジ統合版テスト")
    print("="*70)
    
    # 11×11迷路
    from test_true_perfect_maze import generate_perfect_maze_dfs
    maze = generate_perfect_maze_dfs((11, 11), seed=42)
    
    print("\n迷路構造:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == 9 and j == 9:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # エージェント作成
    agent = MainAgentWithGedigEdges(
        maze=maze,
        datastore_path="data/maze_gedig_edges",
        config={
            'max_edges_per_node': 7,  # マジカルナンバー
            'gedig_threshold': 0.5,
            'max_depth': 20,
            'search_k': 50
        }
    )
    
    print("\n実行開始...")
    print("-" * 70)
    
    for step in range(300):
        if agent.is_goal_reached():
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            break
        
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # 進捗表示
        if step < 3 or step % 50 == 49:
            stats = agent.get_statistics()
            print(f"\nStep {step+1}:")
            print(f"  位置: {agent.position}")
            print(f"  距離: {stats['distance_to_goal']}")
            print(f"  エピソード数: {stats['episode_count']}")
            print(f"  グラフエッジ数: {stats['graph_edges']}")
            print(f"  平均geDIG: {stats['avg_gedig']:.4f}")
    else:
        print(f"\n⏰ タイムアウト")
    
    # 最終統計
    stats = agent.get_statistics()
    
    print("\n" + "="*70)
    print("📊 最終結果")
    print("="*70)
    
    print(f"\nゴール到達: {'✅' if agent.is_goal_reached() else '❌'}")
    print(f"総ステップ: {stats['steps']}")
    print(f"壁衝突率: {stats['wall_hit_rate']:.1%}")
    print(f"エピソード数: {stats['episode_count']}")
    print(f"グラフノード数: {stats['graph_nodes']}")
    print(f"グラフエッジ数: {stats['graph_edges']}")
    print(f"平均geDIG: {stats['avg_gedig']:.4f}")
    print(f"DataStore: {agent.datastore.storage_path}")
    
    print("\n✨ DataStore保存 + geDIGエッジ評価を統合！")


if __name__ == "__main__":
    test_gedig_integration()