#!/usr/bin/env python3
"""
Pure Movement Episodic Memory Navigator
メインコードのIndexとDataStoreを使用した実装
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import sys
import os

# メインコードのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

# InsightSpikeメインコードからインポート
from insightspike.index import IntegratedVectorGraphIndex
from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.environments.proper_maze_generator import ProperMazeGenerator


class PureMovementEpisodicNavigator:
    """純粋な移動エピソード記憶によるナビゲーター"""
    
    def __init__(self, 
                 maze: np.ndarray,
                 datastore_path: str = "data/experiments/pure_movement",
                 max_depth: int = 5):
        """
        Args:
            maze: 迷路配列（0=通路、1=壁）
            datastore_path: DataStore保存パス
            max_depth: メッセージパッシングの最大深度
        """
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        self.max_depth = max_depth
        
        # DataStoreの初期化（FileSystem使用）
        self.datastore = DataStoreFactory.create(
            "filesystem",
            base_path=datastore_path
        )
        
        # 統合インデックスの初期化
        self.index = IntegratedVectorGraphIndex(
            dimension=7,  # 7次元エピソードベクトル
            config={
                'similarity_threshold': 0.4,
                'max_edges_per_node': 20,
                'enable_spatial_index': True,
                'enable_graph_index': True
            }
        )
        
        # メモリシステム
        self.visit_counts = {}
        self.episode_id = 0
        self.path = [self.position]
        self.wall_hits = 0
        
        # アクションマッピング
        self.actions = ['up', 'right', 'down', 'left']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # メトリクス収集
        self.metrics = {
            'search_times': [],
            'hop_usage': {f'{i}-hop': 0 for i in range(1, max_depth+1)},
            'episode_types': {'movement': 0, 'visual': 0}
        }
        
        # 実験セッション情報
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._initialize_session()
    
    def _initialize_session(self):
        """実験セッションの初期化とログ設定"""
        session_info = {
            'session_id': self.session_id,
            'maze_size': (self.height, self.width),
            'start': self.position,
            'goal': self.goal,
            'start_time': datetime.now().isoformat(),
            'config': {
                'max_depth': self.max_depth,
                'dimension': 7
            }
        }
        
        # セッション情報をDataStoreに保存
        self.datastore.save_artifact(
            f"session_{self.session_id}",
            session_info,
            metadata={'type': 'session_info'}
        )
    
    def _find_start(self) -> Tuple[int, int]:
        """スタート位置を検索（通常は(1,1)）"""
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (1, 1)
    
    def _find_goal(self) -> Tuple[int, int]:
        """ゴール位置を検索（通常は右下）"""
        for i in range(self.height-1, -1, -1):
            for j in range(self.width-1, -1, -1):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (self.height-2, self.width-2)
    
    def _update_visit_count(self):
        """現在位置の訪問回数を更新"""
        pos = self.position
        if pos not in self.visit_counts:
            self.visit_counts[pos] = 0
        self.visit_counts[pos] += 1
    
    def _create_episode_vector(self, x: int, y: int, direction: str,
                               success: bool, is_wall: bool) -> np.ndarray:
        """7次元の移動エピソードベクトルを生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        # 次元0-1: 正規化された位置
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # 次元2: 移動方向（0-1に正規化）
        vec[2] = self.action_to_idx[direction] / 3.0
        
        # 次元3: 成功フラグ
        vec[3] = 1.0 if success else 0.0
        
        # 次元4: 壁/通路
        vec[4] = -1.0 if is_wall else 1.0
        
        # 次元5: 訪問回数（対数正規化）
        visit_count = self.visit_counts.get((x, y), 0)
        vec[5] = np.log1p(visit_count) / 10.0
        
        # 次元6: ゴールフラグ
        vec[6] = 1.0 if (x, y) == self.goal else 0.0
        
        return vec
    
    def _create_visual_episode(self, x: int, y: int, direction: str) -> Dict:
        """視覚観測エピソードを生成"""
        dx, dy = self.action_deltas[direction]
        nx, ny = x + dx, y + dy
        
        is_wall = True
        if 0 <= nx < self.height and 0 <= ny < self.width:
            is_wall = (self.maze[nx, ny] == 1)
        
        vec = self._create_episode_vector(x, y, direction, False, is_wall)
        vec[3] = 0.5  # 未実行を示す中立値
        
        return {
            'vector': vec,
            'metadata': {
                'type': 'visual',
                'position': (x, y),
                'direction': direction,
                'is_wall': is_wall,
                'episode_id': self.episode_id,
                'timestamp': time.time()
            }
        }
    
    def _add_visual_observations(self):
        """現在位置から4方向の視覚エピソードを追加"""
        x, y = self.position
        
        for direction in self.actions:
            episode = self._create_visual_episode(x, y, direction)
            
            # インデックスに追加
            idx = self.index.add(
                episode['vector'],
                metadata=episode['metadata']
            )
            
            # DataStoreに保存
            self.datastore.save_episode({
                'episode_id': self.episode_id,
                'vector': episode['vector'].tolist(),
                'metadata': episode['metadata']
            })
            
            self.episode_id += 1
            self.metrics['episode_types']['visual'] += 1
    
    def _create_query_vector(self) -> np.ndarray:
        """現在状態からクエリベクトルを生成（純粋版）"""
        vec = np.zeros(7, dtype=np.float32)
        
        x, y = self.position
        
        # 現在位置
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # 方向: 中立（どの方向でも良い）
        vec[2] = 0.5
        
        # 成功した移動を求める
        vec[3] = 1.0
        
        # 壁/通路: 中立
        vec[4] = 0.0
        
        # 訪問回数: 現在の状態
        vec[5] = np.log1p(self.visit_counts.get((x, y), 0)) / 10.0
        
        # ゴール: 中立
        vec[6] = 0.5
        
        return vec
    
    def _message_passing(self, indices: List[int], depth: int) -> np.ndarray:
        """多段階メッセージパッシング（純粋版）"""
        if depth <= 0 or not indices:
            return np.zeros(7)
        
        # 初期メッセージ
        messages = {idx: 1.0 / (i + 1) for i, idx in enumerate(indices[:20])}
        
        # グラフ取得
        graph = self.index.get_graph()
        
        # 各ホップで伝播
        for d in range(depth):
            new_messages = {}
            decay = 0.8 ** d
            
            for node, value in messages.items():
                # セルフループ
                if d < depth - 1:
                    new_messages[node] = value * 0.7 * decay
                
                # 隣接ノードへ
                if node in graph:
                    for neighbor, edge_data in graph[node].items():
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
        
        # 集約
        direction = np.zeros(7)
        total_weight = 0
        
        for idx, value in messages.items():
            # メタデータ取得
            result = self.index.search_by_id(idx)
            if result:
                vec = result['vector']
                metadata = result.get('metadata', {})
                
                # 成功エピソードを重視（控えめに）
                if metadata.get('type') == 'movement':
                    if metadata.get('success'):
                        weight = value * 1.2  # 控えめな重み
                    else:
                        weight = value * 0.8
                else:
                    weight = value
                
                direction += vec * weight
                total_weight += weight
        
        if total_weight > 0:
            direction = direction / total_weight
        
        return direction
    
    def get_action(self) -> str:
        """記憶から次の行動を決定"""
        self._update_visit_count()
        self._add_visual_observations()
        
        query = self._create_query_vector()
        
        # 検索
        start_time = time.time()
        results = self.index.search(query, k=30)
        search_time = (time.time() - start_time) * 1000
        self.metrics['search_times'].append(search_time)
        
        if not results:
            # 記憶がない場合はランダム
            return np.random.choice(self.actions)
        
        # 多段階で洞察を生成
        insights = []
        for depth in range(1, self.max_depth + 1):
            indices = [r['id'] for r in results]
            insight = self._message_passing(indices, depth)
            insights.append(insight)
            self.metrics['hop_usage'][f'{depth}-hop'] += 1
        
        # 深度ごとの重み付き平均
        final_insight = np.zeros(7)
        for i, insight in enumerate(insights):
            weight = 1.0 / (i + 1)  # 浅い深度を重視
            final_insight += insight * weight
        
        final_insight = final_insight / len(insights)
        
        # 方向成分を抽出
        direction_value = final_insight[2] * 3.0
        direction_idx = int(round(direction_value))
        
        # 確率分布に変換
        probs = np.ones(4) * 0.1
        if 0 <= direction_idx < 4:
            confidence = final_insight[3]  # 成功度を信頼度として使用
            probs[direction_idx] += 0.6 * confidence
        
        probs = probs / probs.sum()
        
        return np.random.choice(self.actions, p=probs)
    
    def move(self, action: str) -> bool:
        """行動を実行し、移動エピソードを記録"""
        if action not in self.actions:
            return False
        
        x, y = self.position
        dx, dy = self.action_deltas[action]
        new_x, new_y = x + dx, y + dy
        
        # 移動試行
        success = False
        is_wall = True
        
        if 0 <= new_x < self.height and 0 <= new_y < self.width:
            if self.maze[new_x, new_y] == 0:
                self.position = (new_x, new_y)
                self.path.append(self.position)
                success = True
                is_wall = False
        
        if not success:
            self.wall_hits += 1
        
        # 移動エピソードを記録
        vec = self._create_episode_vector(x, y, action, success, is_wall)
        
        episode_data = {
            'episode_id': self.episode_id,
            'vector': vec.tolist(),
            'metadata': {
                'type': 'movement',
                'position': (x, y),
                'action': action,
                'success': success,
                'is_wall': is_wall,
                'timestamp': time.time()
            }
        }
        
        # インデックスに追加
        self.index.add(vec, metadata=episode_data['metadata'])
        
        # DataStoreに保存
        self.datastore.save_episode(episode_data)
        
        self.episode_id += 1
        self.metrics['episode_types']['movement'] += 1
        
        return success
    
    def navigate(self, max_steps: int = 25000) -> Dict:
        """迷路をナビゲート"""
        start_time = time.time()
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                
                # 成功時の結果
                result = self._create_result(
                    success=True,
                    steps=step,
                    total_time=total_time
                )
                
                # 最終結果を保存
                self._save_final_result(result)
                
                print(f"\n🎉 SUCCESS! Reached goal in {step} steps")
                return result
            
            # 行動決定と実行
            action = self.get_action()
            self.move(action)
            
            # 進捗レポート
            if step % 1000 == 0 and step > 0:
                self._report_progress(step)
        
        # 失敗時の結果
        total_time = time.time() - start_time
        result = self._create_result(
            success=False,
            steps=max_steps,
            total_time=total_time
        )
        
        self._save_final_result(result)
        
        return result
    
    def _create_result(self, success: bool, steps: int, total_time: float) -> Dict:
        """結果辞書を生成"""
        return {
            'success': success,
            'steps': steps,
            'total_time': total_time,
            'total_episodes': self.episode_id,
            'wall_hits': self.wall_hits,
            'wall_hit_rate': self.wall_hits / max(steps, 1),
            'path_length': len(self.path),
            'visit_counts': self.visit_counts,
            'metrics': self.metrics,
            'final_position': self.position,
            'distance_to_goal': abs(self.position[0] - self.goal[0]) + 
                               abs(self.position[1] - self.goal[1])
        }
    
    def _report_progress(self, step: int):
        """進捗をレポート"""
        dist = abs(self.position[0] - self.goal[0]) + \
               abs(self.position[1] - self.goal[1])
        hit_rate = self.wall_hits / step * 100
        
        print(f"Step {step}: pos={self.position}, dist={dist}, "
              f"wall_hits={self.wall_hits} ({hit_rate:.1f}%), "
              f"episodes={self.episode_id}")
    
    def _save_final_result(self, result: Dict):
        """最終結果をDataStoreに保存"""
        # 結果を保存
        self.datastore.save_artifact(
            f"result_{self.session_id}",
            result,
            metadata={'type': 'final_result'}
        )
        
        # パスを保存
        self.datastore.save_artifact(
            f"path_{self.session_id}",
            {'path': self.path},
            metadata={'type': 'navigation_path'}
        )
        
        print(f"\n📁 Results saved to DataStore: {self.datastore.base_path}")
        print(f"   Session ID: {self.session_id}")
        print(f"   Total episodes: {self.episode_id}")


def run_experiment():
    """実験を実行"""
    print("=" * 60)
    print("Pure Movement Episodic Memory Navigation")
    print("Using InsightSpike MainCode Index & DataStore")
    print("=" * 60)
    
    # 迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(51, 51), seed=42)
    
    print(f"Maze size: 51×51")
    print(f"Max steps: 25000")
    
    # ナビゲーター作成
    navigator = PureMovementEpisodicNavigator(
        maze=maze,
        datastore_path="data/experiments/pure_movement_50x50",
        max_depth=5
    )
    
    # ナビゲート
    result = navigator.navigate(max_steps=25000)
    
    # 結果表示
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Wall hit rate: {result['wall_hit_rate']:.2%}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Path length: {result['path_length']}")
    
    if not result['success']:
        print(f"Final distance to goal: {result['distance_to_goal']}")
    
    return result


if __name__ == "__main__":
    run_experiment()