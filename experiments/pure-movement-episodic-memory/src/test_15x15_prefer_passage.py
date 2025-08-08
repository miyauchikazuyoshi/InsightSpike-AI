#!/usr/bin/env python3
"""
15×15迷路での通路優先検索実験
クエリの4次元目を1.0（通路）に固定
"""

import numpy as np
import time
import networkx as nx
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.core.episode import Episode
from insightspike.vector_index.factory import VectorIndexFactory
from test_true_perfect_maze import generate_perfect_maze_dfs


class PassagePreferenceAgent:
    """通路優先検索エージェント"""
    
    def __init__(self, maze: np.ndarray, use_mask: bool = True, 
                 prefer_passage: bool = True,
                 datastore_path: str = "data/passage_preference",
                 config: Optional[Dict] = None):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.height - 2, self.width - 2)
        self.use_mask = use_mask
        self.prefer_passage = prefer_passage
        
        # 設定
        self.config = config or {}
        self.max_edges_per_node = self.config.get('max_edges_per_node', 7)
        self.gedig_threshold = self.config.get('gedig_threshold', 0.5)
        self.max_depth = self.config.get('max_depth', 10)
        self.search_k = self.config.get('search_k', 30)
        
        # 行動定義
        self.actions = ['up', 'right', 'down', 'left']
        self.action_deltas = {
            'up': (-1, 0), 'right': (0, 1),
            'down': (1, 0), 'left': (0, -1)
        }
        self.action_to_idx = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
        
        # DataStore
        self.datastore = DataStoreFactory.create("filesystem", base_path=datastore_path)
        
        # ベクトルインデックス
        self.vector_index = VectorIndexFactory.create_index(
            dimension=7,
            index_type="numpy",
            optimize=True,
            normalize=True
        )
        
        # 方向マスク（次元2を除外）
        self.mask = np.ones(7, dtype=np.float32)
        if self.use_mask:
            self.mask[2] = 0.0
        
        # グラフ構造
        self.episode_graph = nx.DiGraph()
        
        # エピソード
        self.episodes = []
        
        # 統計
        self.steps = 0
        self.wall_hits = 0
        self.visit_counts = {}
        self.path = [self.position]
        
        print(f"✅ {'通路優先+マスク' if prefer_passage and use_mask else 'マスクのみ' if use_mask else '通常'}検索エージェント初期化")
        print(f"  迷路サイズ: {self.height}×{self.width}")
        print(f"  通路優先: {'有効' if prefer_passage else '無効'}")
        print(f"  方向マスク: {'有効' if use_mask else '無効'}")
    
    def _create_episode_vector(self, x: int, y: int, direction: str,
                               success: bool, is_wall: bool,
                               episode_type: str) -> np.ndarray:
        """7次元エピソードベクトル生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        # 位置（正規化）
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # 方向
        vec[2] = self.action_to_idx.get(direction, 0) / 3.0
        
        # 成功/失敗（視覚エピソードは中立）
        if episode_type == 'visual':
            vec[3] = 0.5
        else:
            vec[3] = 1.0 if success else 0.0
        
        # 壁情報
        vec[4] = -1.0 if is_wall else 1.0
        
        # 訪問回数
        visit_count = self.visit_counts.get((x, y), 0)
        vec[5] = min(1.0, visit_count / 10.0)
        
        # ゴール判定
        vec[6] = 1.0 if (x, y) == self.goal else 0.0
        
        return vec
    
    def _create_query_vector(self, x: int, y: int) -> np.ndarray:
        """クエリベクトル生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        vec[0] = x / self.height
        vec[1] = y / self.width
        vec[2] = 0.5  # 方向NULL
        vec[3] = 1.0  # 成功希望
        
        # 通路優先設定
        if self.prefer_passage:
            vec[4] = 1.0  # 通路を希望（壁を避ける）
        else:
            vec[4] = 0.0  # 壁/通路NULL
            
        vec[5] = min(1.0, self.visit_counts.get((x, y), 0) / 10.0)
        vec[6] = 1.0 if (x, y) == self.goal else 0.0
        
        return vec
    
    def _calculate_gedig(self, idx1: int, idx2: int, similarity: float) -> float:
        """geDIG計算（GED - IG、低いほど良い）"""
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
        
        # GED
        ged = (spatial_distance / (self.height + self.width) * 0.3 +
               temporal_distance * 0.3 +
               type_difference * 0.4)
        
        # IG
        ig = max(0, similarity) * 0.5
        
        # geDIG = GED - IG（低いほど良い）
        return ged - ig
    
    def _add_episode(self, vec: np.ndarray, metadata: Dict) -> int:
        """エピソード追加"""
        text = f"Episode at step {self.steps}: {metadata.get('type', 'unknown')}"
        episode = Episode(
            text=text,
            vec=vec,
            c=0.5,
            timestamp=time.time(),
            metadata=metadata
        )
        
        idx = len(self.episodes)
        
        self.episodes.append({
            'id': idx,
            'episode': episode,
            'vector': vec,
            'metadata': metadata
        })
        
        # ベクトルインデックス追加
        self.vector_index.add(vec.reshape(1, -1))
        
        # グラフノード追加
        self.episode_graph.add_node(idx, **metadata)
        
        # geDIGエッジ生成（簡略化）
        if len(self.episodes) > 1 and self.vector_index.ntotal > 1:
            # マスクした検索
            if self.use_mask:
                masked_vec = vec * self.mask
            else:
                masked_vec = vec
            
            distances, indices = self.vector_index.search(
                masked_vec.reshape(1, -1),
                k=min(self.search_k, len(self.episodes))
            )
            
            edge_count = 0
            for dist, other_idx in zip(distances[0], indices[0]):
                if other_idx != idx and edge_count < self.max_edges_per_node:
                    similarity = 1.0 - dist
                    gedig = self._calculate_gedig(idx, other_idx, similarity)
                    if gedig < self.gedig_threshold:
                        self.episode_graph.add_edge(
                            idx, other_idx,
                            weight=1.0 / (1.0 + gedig),
                            gedig=gedig
                        )
                        edge_count += 1
        
        return idx
    
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
        
        messages = {}
        for rank, idx in enumerate(start_indices[:10]):
            if 0 <= idx < len(self.episodes):
                messages[idx] = 1.0 / (rank + 1)
        
        for d in range(depth):
            new_messages = {}
            decay = 0.8 ** d
            
            for node_idx, msg_value in messages.items():
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
        """行動決定"""
        # 視覚観測
        self._add_visual_observations()
        
        # クエリベクトル生成
        x, y = self.position
        query_vec = self._create_query_vector(x, y)
        
        # マスクした類似検索
        if self.use_mask:
            masked_query = query_vec * self.mask
        else:
            masked_query = query_vec
        
        if self.vector_index.ntotal == 0:
            return np.random.choice(self.actions)
        
        distances, indices = self.vector_index.search(
            masked_query.reshape(1, -1),
            k=min(20, self.vector_index.ntotal)
        )
        
        # メッセージパッシング
        depth = min(5, self.max_depth)
        insight_vector = self._message_passing(indices[0].tolist(), depth)
        
        # 方向成分抽出
        direction_component = insight_vector[2]
        
        # 10%の確率で探索
        if np.random.random() < 0.1:
            return np.random.choice(self.actions)
        
        # 4方向に正規化
        if direction_component < 0.166:
            return 'up'
        elif direction_component < 0.5:
            return 'right'
        elif direction_component < 0.833:
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
        self.path.append(self.position)
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
            'unique_visits': len(set(self.path))
        }


def visualize_results(maze, agent, filename):
    """結果を可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 迷路と経路
    ax1.imshow(maze, cmap='binary')
    
    # 経路を描画
    if len(agent.path) > 1:
        path_x = [p[1] for p in agent.path]
        path_y = [p[0] for p in agent.path]
        ax1.plot(path_x, path_y, 'b-', alpha=0.5, linewidth=2)
    
    ax1.plot(1, 1, 'go', markersize=10, label='Start')
    ax1.plot(agent.goal[1], agent.goal[0], 'r*', markersize=15, label='Goal')
    ax1.plot(agent.position[1], agent.position[0], 'bo', markersize=8, label='Current')
    
    title = "Passage Preference" if agent.prefer_passage else "Normal"
    if agent.use_mask:
        title += " + Mask"
    ax1.set_title(f"{title} - {agent.steps} steps")
    ax1.legend()
    ax1.axis('off')
    
    # 訪問頻度ヒートマップ
    visit_map = np.zeros_like(maze, dtype=float)
    for pos, count in agent.visit_counts.items():
        visit_map[pos] = count
    
    im = ax2.imshow(visit_map, cmap='hot', interpolation='nearest')
    ax2.set_title('Visit Frequency')
    plt.colorbar(im, ax=ax2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def test_passage_preference():
    """通路優先の効果をテスト"""
    print("="*70)
    print("🎯 15×15迷路での通路優先検索実験")
    print("="*70)
    
    # 15×15迷路生成
    maze = generate_perfect_maze_dfs((15, 15), seed=42)
    
    print("\n迷路生成完了")
    print(f"  サイズ: 15×15")
    print(f"  スタート: (1, 1)")
    print(f"  ゴール: (13, 13)")
    
    results = {}
    
    # 3つの設定でテスト
    configs = [
        (False, False, "normal"),     # 通常検索
        (True, False, "mask_only"),   # マスクのみ
        (True, True, "passage_pref"),  # 通路優先+マスク
    ]
    
    for use_mask, prefer_passage, name in configs:
        print(f"\n{'='*50}")
        print(f"実行中: {name}")
        print(f"{'='*50}")
        
        agent = PassagePreferenceAgent(
            maze=maze,
            use_mask=use_mask,
            prefer_passage=prefer_passage,
            datastore_path=f"data/15x15_{name}",
            config={
                'max_edges_per_node': 7,
                'gedig_threshold': 0.5,
                'max_depth': 10,
                'search_k': 30
            }
        )
        
        max_steps = 1000
        
        for step in range(max_steps):
            if agent.is_goal_reached():
                print(f"\n🎉 成功！ {step}ステップでゴール到達")
                break
            
            action = agent.get_action()
            success = agent.execute_action(action)
            
            if step % 100 == 99:
                stats = agent.get_statistics()
                print(f"  Step {step+1}: 位置{agent.position}, "
                      f"距離{stats['distance_to_goal']}, "
                      f"壁衝突率{stats['wall_hit_rate']:.1%}")
        else:
            print(f"\n⏰ {max_steps}ステップで終了")
        
        stats = agent.get_statistics()
        results[name] = stats
        
        # 可視化
        visualize_results(
            maze, agent,
            f"../results/15x15_{name}.png"
        )
        
        print(f"\n📊 最終統計:")
        print(f"  ゴール到達: {'✅' if agent.is_goal_reached() else '❌'}")
        print(f"  総ステップ: {stats['steps']}")
        print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
        print(f"  ユニーク訪問: {stats['unique_visits']}")
        print(f"  エピソード数: {stats['episodes']}")
    
    # 比較結果
    print("\n" + "="*70)
    print("📊 比較結果")
    print("="*70)
    
    for name in ["normal", "mask_only", "passage_pref"]:
        r = results[name]
        print(f"\n{name}:")
        print(f"  ステップ数: {r['steps']}")
        print(f"  壁衝突率: {r['wall_hit_rate']:.1%}")
        print(f"  ユニーク訪問: {r['unique_visits']}")
    
    # 改善率計算
    if results['passage_pref']['wall_hit_rate'] < results['mask_only']['wall_hit_rate']:
        improvement = (results['mask_only']['wall_hit_rate'] - results['passage_pref']['wall_hit_rate']) / results['mask_only']['wall_hit_rate'] * 100
        print(f"\n✨ 通路優先により壁衝突率が{improvement:.1f}%改善")
    
    print("\n💡 分析:")
    print("- 通路優先: 壁エピソードを避け、通路エピソードを選好")
    print("- 視覚情報の活用: 壁を事前に検知して回避")
    print("- 探索効率: より少ない壁衝突で迷路を探索")


if __name__ == "__main__":
    test_passage_preference()