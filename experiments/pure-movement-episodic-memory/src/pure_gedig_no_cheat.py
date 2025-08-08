#!/usr/bin/env python3
"""
純粋なgeDIG実装（チートなし）
洞察ベクトル生成 → 方向成分抽出 → 4方向正規化のみ
"""

import numpy as np
import time
import networkx as nx
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.core.episode import Episode
from insightspike.vector_index.factory import VectorIndexFactory


class PureGedigNoCheat:
    """チートなしの純粋geDIG実装"""
    
    def __init__(self, maze: np.ndarray, datastore_path: str = "data/pure_gedig", 
                 config: Optional[Dict] = None):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.height - 2, self.width - 2)
        
        # 設定
        self.config = config or {}
        self.max_edges_per_node = self.config.get('max_edges_per_node', 7)
        self.gedig_threshold = self.config.get('gedig_threshold', 0.5)
        self.max_depth = self.config.get('max_depth', 20)
        self.search_k = self.config.get('search_k', 50)
        
        # 行動定義
        self.actions = ['up', 'right', 'down', 'left']
        self.action_deltas = {
            'up': (-1, 0), 'right': (0, 1),
            'down': (1, 0), 'left': (0, -1)
        }
        # 方向を数値にマップ
        self.action_to_idx = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
        
        # DataStore
        self.datastore = DataStoreFactory.create("filesystem", base_path=datastore_path)
        
        # ベクトルインデックス（7次元）
        self.vector_index = VectorIndexFactory.create_index(
            dimension=7,
            index_type="numpy",
            optimize=True,
            normalize=True
        )
        
        # グラフ構造
        self.episode_graph = nx.DiGraph()
        
        # エピソード
        self.episodes = []
        
        # 統計
        self.steps = 0
        self.wall_hits = 0
        self.gedig_values = []
        self.visit_counts = {}  # 位置ごとの訪問回数
        
        print(f"✅ 純粋geDIG実装（チートなし）")
        print(f"  max_edges: {self.max_edges_per_node}")
        print(f"  max_depth: {self.max_depth}")
    
    def _create_episode_vector(self, x: int, y: int, direction: str, 
                               success: bool, is_wall: bool, 
                               episode_type: str) -> np.ndarray:
        """7次元エピソードベクトル生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        # 位置（正規化）
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # 方向（0-1に正規化）
        vec[2] = self.action_to_idx.get(direction, 0) / 3.0
        
        # 成功/失敗
        vec[3] = 1.0 if success else 0.0
        
        # 壁情報
        vec[4] = -1.0 if is_wall else 1.0
        
        # 訪問回数（エピソード作成時点での訪問頻度）
        visit_count = self.visit_counts.get((x, y), 0)
        vec[5] = min(1.0, visit_count / 10.0)
        
        # ゴール判定（視覚/移動エピソードの場合）
        vec[6] = 1.0 if (x, y) == self.goal else 0.0
        
        return vec
    
    def _create_query_vector(self, x: int, y: int) -> np.ndarray:
        """
        クエリベクトル生成
        (現x座標、現y座標、null(移動方向)、成功希望、null(壁or通路)、訪問回数、ゴール判定）
        """
        vec = np.zeros(7, dtype=np.float32)
        
        # 0,1: 現在位置（正規化）
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # 2: 方向はnull（中間値）
        vec[2] = 0.5
        
        # 3: 成功を希望
        vec[3] = 1.0
        
        # 4: 壁/通路はnull（中間値）
        vec[4] = 0.0
        
        # 5: 訪問回数（正規化、多いほど避けたい）
        visit_count = self.visit_counts.get((x, y), 0)
        # 訪問回数を0-1の範囲に正規化（10回以上は1.0）
        vec[5] = min(1.0, visit_count / 10.0)
        
        # 6: ゴール判定
        vec[6] = 1.0 if (x, y) == self.goal else 0.0
        
        return vec
    
    def _calculate_gedig(self, idx1: int, idx2: int, similarity: float) -> float:
        """geDIG計算"""
        meta1 = self.episodes[idx1]['metadata']
        meta2 = self.episodes[idx2]['metadata']
        
        # 空間距離
        pos1 = meta1.get('position', [0, 0])
        pos2 = meta2.get('position', [0, 0])
        spatial_distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # 時間距離
        step1 = meta1.get('step', 0)
        step2 = meta2.get('step', 0)
        temporal_distance = min(abs(step1 - step2), 100) / 100.0
        
        # タイプの違い
        type1 = meta1.get('type', '')
        type2 = meta2.get('type', '')
        type_difference = 0 if type1 == type2 else 1
        
        # 結果の違い
        success1 = meta1.get('success', None)
        success2 = meta2.get('success', None)
        outcome_difference = 0
        if success1 is not None and success2 is not None:
            outcome_difference = 0 if success1 == success2 else 1
        
        # GED: Generalized Edit Distance
        ged = (spatial_distance / (self.height + self.width) * 0.3 +
               temporal_distance * 0.2 +
               type_difference * 0.2 +
               outcome_difference * 0.3)
        
        # IG: Information Gain (類似度から)
        ig = max(0, similarity) * 0.5
        
        # geDIG = IG - GED（負の値が良い）
        return ig - ged
    
    def _create_gedig_edges(self, new_idx: int, new_vec: np.ndarray):
        """geDIG評価でエッジ生成"""
        if len(self.episodes) <= 1:
            return
        
        # 類似エピソード検索
        if self.vector_index.ntotal > 0:
            distances, indices = self.vector_index.search(
                new_vec.reshape(1, -1),
                k=min(self.search_k, len(self.episodes) - 1)
            )
            
            edge_candidates = []
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.episodes) or idx == new_idx:
                    continue
                
                # コサイン類似度（distanceは1-similarityなので変換）
                similarity = 1.0 - dist
                
                # geDIG計算
                gedig_value = self._calculate_gedig(new_idx, idx, similarity)
                self.gedig_values.append(gedig_value)
                
                # 負の値が良いので、閾値より大きい（より負の）値を選ぶ
                if gedig_value > -self.gedig_threshold:
                    edge_candidates.append({
                        'target_idx': idx,
                        'gedig': gedig_value,
                        'similarity': similarity
                    })
            
            # geDIG値でソート（負の値が良いので降順）
            edge_candidates.sort(key=lambda x: x['gedig'], reverse=True)
            
            # 上位k個のエッジ
            for edge in edge_candidates[:self.max_edges_per_node]:
                self.episode_graph.add_edge(
                    new_idx,
                    edge['target_idx'],
                    weight=1.0 / (1.0 - edge['gedig']),  # 負の値が良いので
                    gedig=edge['gedig']
                )
    
    def _add_episode(self, vec: np.ndarray, metadata: Dict) -> int:
        """エピソード追加"""
        # Episode作成（vecパラメータを追加）
        text = f"Episode at step {self.steps}: {metadata.get('type', 'unknown')}"
        episode = Episode(
            text=text,
            vec=vec,  # ベクトルを追加！
            c=0.5,  # デフォルトのC値
            timestamp=time.time(),
            metadata=metadata
        )
        
        # インデックス
        idx = len(self.episodes)
        
        # キャッシュ追加（DataStore保存はスキップし、メモリ内のみ）
        self.episodes.append({
            'id': idx,  # シンプルなID
            'episode': episode,  # Episodeオブジェクト
            'vector': vec,
            'metadata': metadata
        })
        
        # ベクトルインデックス追加
        self.vector_index.add(vec.reshape(1, -1))
        
        # グラフノード追加
        self.episode_graph.add_node(idx, **metadata)
        
        # geDIGエッジ生成
        self._create_gedig_edges(idx, vec)
        
        # DataStore永続化（定期的に）
        if len(self.episodes) % 100 == 0:
            self._save_to_datastore()
        
        return idx
    
    def _save_to_datastore(self):
        """エピソードをDataStoreに永続化"""
        episodes_to_save = []
        for ep_data in self.episodes:
            episode = ep_data['episode']
            # メタデータのbool型をPython標準型に変換
            metadata = {}
            for key, val in episode.metadata.items():
                if isinstance(val, (np.bool_, np.integer, np.floating)):
                    metadata[key] = val.item()
                elif isinstance(val, np.ndarray):
                    metadata[key] = val.tolist()
                else:
                    metadata[key] = val
                    
            episodes_to_save.append({
                'text': episode.text,
                'vec': episode.vec,
                'c_value': float(episode.c),
                'timestamp': float(episode.timestamp),
                'metadata': metadata
            })
        
        self.datastore.save_episodes(episodes_to_save, namespace="maze_episodes")
    
    def _add_visual_observations(self):
        """視覚観測追加"""
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
            
            metadata = {
                'type': 'visual',
                'position': [x, y],
                'direction': direction,
                'is_wall': is_wall,
                'step': self.steps
            }
            
            self._add_episode(vec, metadata)
    
    def _message_passing(self, start_indices: List[int], depth: int) -> np.ndarray:
        """グラフメッセージパッシング"""
        if not start_indices or depth <= 0:
            return np.zeros(7)
        
        # 初期メッセージ
        messages = {}
        for rank, idx in enumerate(start_indices[:20]):
            if 0 <= idx < len(self.episodes):
                messages[idx] = 1.0 / (rank + 1)
        
        # 伝播
        for d in range(depth):
            new_messages = {}
            decay = 0.7 ** d
            
            for node_idx, msg_value in messages.items():
                # 自己ループ
                if d < depth - 1:
                    new_messages[node_idx] = msg_value * 0.5 * decay
                
                # 隣接ノード
                if self.episode_graph.has_node(node_idx):
                    for neighbor in self.episode_graph.neighbors(node_idx):
                        edge_data = self.episode_graph[node_idx][neighbor]
                        weight = edge_data.get('weight', 1.0)
                        
                        propagated = msg_value * weight * decay
                        
                        if neighbor in new_messages:
                            new_messages[neighbor] = max(new_messages[neighbor], propagated)
                        else:
                            new_messages[neighbor] = propagated
            
            messages = new_messages
            if not messages:
                break
        
        # 集約
        aggregated = np.zeros(7)
        total_weight = 0
        
        for idx, weight in messages.items():
            if idx < len(self.episodes):
                vec = self.episodes[idx]['vector']
                aggregated += vec * weight
                total_weight += weight
        
        if total_weight > 0:
            aggregated /= total_weight
        
        return aggregated
    
    def get_action(self) -> str:
        """
        行動決定（チートなし）
        1. 視覚観測
        2. クエリベクトル生成
        3. 類似検索
        4. メッセージパッシング
        5. 洞察ベクトル生成
        6. 方向成分抽出
        7. 4方向正規化
        """
        # 1. 視覚観測
        self._add_visual_observations()
        
        # 2. クエリベクトル生成（正しい定義で）
        x, y = self.position
        query_vec = self._create_query_vector(x, y)
        
        # 3. 類似検索
        if self.vector_index.ntotal == 0:
            return np.random.choice(self.actions)
        
        distances, indices = self.vector_index.search(
            query_vec.reshape(1, -1),
            k=min(30, self.vector_index.ntotal)
        )
        
        # 4. 深度選択（適応的）
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
        
        # 5. メッセージパッシング → 洞察ベクトル
        insight_vector = self._message_passing(indices[0].tolist(), depth)
        
        # 6. 方向成分抽出（ベクトルの3番目の要素）
        direction_component = insight_vector[2]  # 0-1の範囲
        
        # 7. 4方向に正規化 + 探索要素
        # ε-greedy探索（0%の確率でランダム = 純粋なgeDIG推論）
        if np.random.random() < 0.00:
            return np.random.choice(self.actions)
        
        # 0/3=0.0: up, 1/3=0.333: right, 2/3=0.666: down, 3/3=1.0: left
        if direction_component < 0.166:  # 0.0 ± margin
            return 'up'
        elif direction_component < 0.5:   # 0.333 ± margin
            return 'right'
        elif direction_component < 0.833:  # 0.666 ± margin
            return 'down'
        else:
            return 'left'
    
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
            # 訪問回数を更新
            self.visit_counts[self.position] = self.visit_counts.get(self.position, 0) + 1
        else:
            self.wall_hits += 1
        
        # 移動エピソード追加
        x, y = self.position if not success else (self.position[0] - dx, self.position[1] - dy)
        vec = self._create_episode_vector(
            x, y, action, success, not success, 'movement'
        )
        
        metadata = {
            'type': 'movement',
            'position': [x, y],
            'action': action,
            'success': success,
            'step': self.steps
        }
        
        self._add_episode(vec, metadata)
        
        self.steps += 1
        return success
    
    def is_goal_reached(self) -> bool:
        return self.position == self.goal
    
    def get_statistics(self) -> Dict:
        distance = abs(self.position[0] - self.goal[0]) + \
                  abs(self.position[1] - self.goal[1])
        
        return {
            'steps': self.steps,
            'wall_hits': self.wall_hits,
            'wall_hit_rate': self.wall_hits / max(1, self.steps),
            'distance_to_goal': distance,
            'episodes': len(self.episodes),
            'edges': self.episode_graph.number_of_edges(),
            'avg_gedig': np.mean(self.gedig_values) if self.gedig_values else 0
        }
    
    def finalize(self):
        """実験終了処理"""
        # 最終データを保存
        self._save_to_datastore()
        print(f"✅ {len(self.episodes)}エピソードをDataStoreに保存")


def test_no_cheat():
    """チートなし実装のテスト"""
    print("="*70)
    print("🧪 純粋geDIG実装テスト（チートなし）")
    print("="*70)
    
    # 11×11迷路
    from test_true_perfect_maze import generate_perfect_maze_dfs
    maze = generate_perfect_maze_dfs((11, 11), seed=42)
    
    print("\n迷路:")
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
    
    # エージェント
    agent = PureGedigNoCheat(
        maze=maze,
        datastore_path="data/pure_no_cheat",
        config={
            'max_edges_per_node': 7,
            'gedig_threshold': 0.5,
            'max_depth': 20,
            'search_k': 50
        }
    )
    
    print("\n実行...")
    for step in range(500):
        if agent.is_goal_reached():
            print(f"\n🎉 成功！ {step}ステップ")
            break
        
        action = agent.get_action()
        success = agent.execute_action(action)
        
        if step < 5 or step % 50 == 49:
            stats = agent.get_statistics()
            print(f"Step {step+1}: 位置{agent.position}, "
                  f"距離{stats['distance_to_goal']}, "
                  f"行動={action}, {'成功' if success else '壁'}")
    else:
        print(f"\n⏰ タイムアウト")
    
    stats = agent.get_statistics()
    print(f"\n最終結果:")
    print(f"  ゴール: {'✅' if agent.is_goal_reached() else '❌'}")
    print(f"  ステップ: {stats['steps']}")
    print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
    print(f"  エピソード数: {stats['episodes']}")
    print(f"  エッジ数: {stats['edges']}")
    print(f"  平均geDIG: {stats['avg_gedig']:.3f}")


if __name__ == "__main__":
    test_no_cheat()