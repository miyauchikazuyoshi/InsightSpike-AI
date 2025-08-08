#!/usr/bin/env python3
"""
矛盾検知による適応的行動選択
同じ場所で異なる結果のエピソードを検出し、戦略を切り替える
"""

import numpy as np
import time
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
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


class ContradictionAwareAgent:
    """矛盾検知型エージェント"""
    
    def __init__(self, maze: np.ndarray, 
                 datastore_path: str = "data/contradiction_aware",
                 config: Optional[Dict] = None):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.height - 2, self.width - 2)
        
        # 設定
        self.config = config or {}
        self.max_edges_per_node = self.config.get('max_edges_per_node', 7)
        self.gedig_threshold = self.config.get('gedig_threshold', 0.5)
        self.max_depth = self.config.get('max_depth', 10)
        self.search_k = self.config.get('search_k', 30)
        self.contradiction_threshold = self.config.get('contradiction_threshold', 0.3)
        
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
        
        # グラフ構造
        self.episode_graph = nx.DiGraph()
        
        # エピソード
        self.episodes = []
        
        # 矛盾追跡
        self.contradictions = {}  # {(x,y): {'successes': set(), 'failures': set()}}
        self.stuck_positions = set()  # 行き詰まり位置
        self.exploration_mode = False  # 探索モード
        self.exploitation_failures = 0  # 連続失敗カウント
        
        # 統計
        self.steps = 0
        self.wall_hits = 0
        self.visit_counts = {}
        self.path = [self.position]
        self.mode_switches = 0
        
        print(f"✅ 矛盾検知エージェント初期化")
        print(f"  迷路サイズ: {self.height}×{self.width}")
        print(f"  矛盾閾値: {self.contradiction_threshold}")
    
    def _create_episode_vector(self, x: int, y: int, direction: str,
                               success: bool, is_wall: bool,
                               episode_type: str) -> np.ndarray:
        """7次元エピソードベクトル生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        vec[0] = x / self.height
        vec[1] = y / self.width
        vec[2] = self.action_to_idx.get(direction, 0) / 3.0
        
        if episode_type == 'visual':
            vec[3] = 0.5
        else:
            vec[3] = 1.0 if success else 0.0
        
        vec[4] = -1.0 if is_wall else 1.0
        vec[5] = min(1.0, self.visit_counts.get((x, y), 0) / 10.0)
        vec[6] = 1.0 if (x, y) == self.goal else 0.0
        
        return vec
    
    def _detect_contradictions(self, x: int, y: int) -> Dict:
        """現在位置での矛盾を検出"""
        contradictions = {
            'has_contradiction': False,
            'stuck': False,
            'conflicting_directions': [],
            'success_rate': {},
            'recommended_action': None
        }
        
        # 現在位置の履歴を収集
        position_key = (x, y)
        if position_key not in self.contradictions:
            self.contradictions[position_key] = {
                'successes': {},
                'failures': {}
            }
        
        pos_data = self.contradictions[position_key]
        
        # 各方向の成功率を計算
        for direction in self.actions:
            success_count = pos_data['successes'].get(direction, 0)
            failure_count = pos_data['failures'].get(direction, 0)
            total = success_count + failure_count
            
            if total > 0:
                contradictions['success_rate'][direction] = success_count / total
                
                # 同じ方向で成功と失敗が混在 = 矛盾
                if success_count > 0 and failure_count > 0:
                    contradictions['has_contradiction'] = True
                    contradictions['conflicting_directions'].append(direction)
        
        # 行き詰まり検出（全方向で失敗率が高い）
        if len(contradictions['success_rate']) >= 3:
            avg_success = np.mean(list(contradictions['success_rate'].values()))
            if avg_success < self.contradiction_threshold:
                contradictions['stuck'] = True
                self.stuck_positions.add(position_key)
        
        # 最も成功率の高い方向を推奨
        if contradictions['success_rate']:
            best_dir = max(contradictions['success_rate'].items(), 
                          key=lambda x: x[1])
            if best_dir[1] > 0:
                contradictions['recommended_action'] = best_dir[0]
        
        return contradictions
    
    def _update_contradiction_history(self, x: int, y: int, 
                                     direction: str, success: bool):
        """矛盾履歴を更新"""
        position_key = (x, y)
        if position_key not in self.contradictions:
            self.contradictions[position_key] = {
                'successes': {},
                'failures': {}
            }
        
        if success:
            self.contradictions[position_key]['successes'][direction] = \
                self.contradictions[position_key]['successes'].get(direction, 0) + 1
        else:
            self.contradictions[position_key]['failures'][direction] = \
                self.contradictions[position_key]['failures'].get(direction, 0) + 1
    
    def _create_query_vector(self, x: int, y: int, mode: str = 'normal') -> np.ndarray:
        """モードに応じたクエリベクトル生成"""
        vec = np.zeros(7, dtype=np.float32)
        
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        if mode == 'exploration':
            # 探索モード：未訪問を優先
            vec[2] = np.random.random()  # ランダムな方向
            vec[3] = 0.5  # 成功/失敗中立
            vec[4] = 0.0  # 壁/通路中立
            vec[5] = 0.0  # 未訪問を好む
        elif mode == 'escape':
            # 脱出モード：異なる方向を試す
            vec[2] = (self.steps % 4) / 3.0  # 順番に方向を変える
            vec[3] = 0.5  # 中立
            vec[4] = 1.0  # 通路を好む
            vec[5] = 1.0  # 訪問済みでもOK
        else:
            # 通常モード
            vec[2] = 0.5  # 方向NULL
            vec[3] = 1.0  # 成功希望
            vec[4] = 1.0  # 通路希望
            vec[5] = min(1.0, self.visit_counts.get((x, y), 0) / 10.0)
        
        vec[6] = 1.0 if (x, y) == self.goal else 0.0
        
        return vec
    
    def _calculate_gedig(self, idx1: int, idx2: int, similarity: float) -> float:
        """geDIG計算"""
        meta1 = self.episodes[idx1]['metadata']
        meta2 = self.episodes[idx2]['metadata']
        
        pos1 = meta1.get('position', [0, 0])
        pos2 = meta2.get('position', [0, 0])
        spatial_distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        step1 = meta1.get('step', 0)
        step2 = meta2.get('step', 0)
        temporal_distance = min(abs(step1 - step2), 100) / 100.0
        
        type1 = meta1.get('type', '')
        type2 = meta2.get('type', '')
        type_difference = 0 if type1 == type2 else 1
        
        ged = (spatial_distance / (self.height + self.width) * 0.3 +
               temporal_distance * 0.3 +
               type_difference * 0.4)
        
        ig = max(0, similarity) * 0.5
        
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
        
        self.vector_index.add(vec.reshape(1, -1))
        self.episode_graph.add_node(idx, **metadata)
        
        # geDIGエッジ生成
        if len(self.episodes) > 1 and self.vector_index.ntotal > 1:
            distances, indices = self.vector_index.search(
                vec.reshape(1, -1),
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
        """矛盾検知に基づく行動決定"""
        x, y = self.position
        
        # 視覚観測
        self._add_visual_observations()
        
        # 矛盾検出
        contradiction_info = self._detect_contradictions(x, y)
        
        # モード決定
        if contradiction_info['stuck']:
            # 行き詰まり検出 → 脱出モード
            mode = 'escape'
            self.exploration_mode = True
            self.mode_switches += 1
            if self.steps % 50 == 0:
                print(f"    🔄 脱出モード: 位置{self.position}で行き詰まり検出")
        elif self.exploitation_failures > 5:
            # 連続失敗 → 探索モード
            mode = 'exploration'
            self.exploration_mode = True
            self.mode_switches += 1
            if self.steps % 50 == 0:
                print(f"    🔍 探索モード: 連続失敗{self.exploitation_failures}回")
        elif contradiction_info['has_contradiction']:
            # 矛盾あり → 推奨行動を使用
            if contradiction_info['recommended_action']:
                return contradiction_info['recommended_action']
            mode = 'normal'
        else:
            # 通常モード
            mode = 'normal'
            self.exploration_mode = False
        
        # クエリベクトル生成
        query_vec = self._create_query_vector(x, y, mode)
        
        if self.vector_index.ntotal == 0:
            return np.random.choice(self.actions)
        
        # 類似検索
        distances, indices = self.vector_index.search(
            query_vec.reshape(1, -1),
            k=min(20, self.vector_index.ntotal)
        )
        
        # メッセージパッシング
        depth = min(5, self.max_depth)
        insight_vector = self._message_passing(indices[0].tolist(), depth)
        
        # 方向決定
        if mode == 'escape':
            # 脱出モード：最も試していない方向
            untried = []
            for direction in self.actions:
                if direction not in contradiction_info['success_rate']:
                    untried.append(direction)
            if untried:
                return np.random.choice(untried)
            # すべて試した場合は最も成功率の低い方向（逆転の発想）
            if contradiction_info['success_rate']:
                worst_dir = min(contradiction_info['success_rate'].items(),
                               key=lambda x: x[1])
                return worst_dir[0]
        
        # 通常の方向決定
        direction_component = insight_vector[2]
        
        # 探索率の動的調整
        if self.exploration_mode:
            exploration_rate = 0.3
        else:
            exploration_rate = 0.1
        
        if np.random.random() < exploration_rate:
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
        """行動実行と矛盾履歴更新"""
        dx, dy = self.action_deltas[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        # 実行前の位置を記録
        old_position = self.position
        
        success = False
        if (0 <= new_x < self.height and 
            0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            self.position = (new_x, new_y)
            success = True
            self.visit_counts[self.position] = self.visit_counts.get(self.position, 0) + 1
            self.exploitation_failures = 0
        else:
            self.wall_hits += 1
            self.exploitation_failures += 1
        
        # 矛盾履歴を更新
        self._update_contradiction_history(old_position[0], old_position[1], 
                                          action, success)
        
        # 移動エピソード追加
        x, y = old_position
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
            'unique_visits': len(set(self.path)),
            'stuck_positions': len(self.stuck_positions),
            'mode_switches': self.mode_switches,
            'contradictions': len([c for c in self.contradictions.values() 
                                  if any(c['failures'].values()) and any(c['successes'].values())])
        }


def visualize_results(maze, agent, filename):
    """結果を可視化（矛盾位置を表示）"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # 迷路と経路
    ax1.imshow(maze, cmap='binary')
    
    if len(agent.path) > 1:
        path_x = [p[1] for p in agent.path]
        path_y = [p[0] for p in agent.path]
        ax1.plot(path_x, path_y, 'b-', alpha=0.5, linewidth=2)
    
    ax1.plot(1, 1, 'go', markersize=10, label='Start')
    ax1.plot(agent.goal[1], agent.goal[0], 'r*', markersize=15, label='Goal')
    ax1.plot(agent.position[1], agent.position[0], 'bo', markersize=8, label='Current')
    
    ax1.set_title(f"Path - {agent.steps} steps")
    ax1.legend()
    ax1.axis('off')
    
    # 訪問頻度
    visit_map = np.zeros_like(maze, dtype=float)
    for pos, count in agent.visit_counts.items():
        visit_map[pos] = count
    
    im2 = ax2.imshow(visit_map, cmap='hot', interpolation='nearest')
    ax2.set_title('Visit Frequency')
    plt.colorbar(im2, ax=ax2)
    ax2.axis('off')
    
    # 矛盾マップ
    contradiction_map = np.zeros_like(maze, dtype=float)
    for (x, y), data in agent.contradictions.items():
        if any(data['failures'].values()) and any(data['successes'].values()):
            contradiction_map[x, y] = 1
    
    # 行き詰まり位置を強調
    for (x, y) in agent.stuck_positions:
        contradiction_map[x, y] = 2
    
    im3 = ax3.imshow(contradiction_map, cmap='coolwarm', interpolation='nearest')
    ax3.set_title('Contradictions (Red=Stuck)')
    plt.colorbar(im3, ax=ax3)
    ax3.axis('off')
    
    # 成功率マップ
    success_map = np.zeros_like(maze, dtype=float)
    for (x, y), data in agent.contradictions.items():
        total_success = sum(data['successes'].values())
        total_failure = sum(data['failures'].values())
        if total_success + total_failure > 0:
            success_map[x, y] = total_success / (total_success + total_failure)
    
    im4 = ax4.imshow(success_map, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    ax4.set_title('Success Rate')
    plt.colorbar(im4, ax=ax4)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def test_contradiction_detection():
    """矛盾検知の効果をテスト"""
    print("="*70)
    print("🎯 15×15迷路での矛盾検知実験")
    print("="*70)
    
    maze = generate_perfect_maze_dfs((15, 15), seed=42)
    
    print("\n迷路生成完了")
    print(f"  サイズ: 15×15")
    print(f"  スタート: (1, 1)")
    print(f"  ゴール: (13, 13)")
    
    agent = ContradictionAwareAgent(
        maze=maze,
        datastore_path="data/15x15_contradiction",
        config={
            'max_edges_per_node': 7,
            'gedig_threshold': 0.5,
            'max_depth': 10,
            'search_k': 30,
            'contradiction_threshold': 0.3
        }
    )
    
    max_steps = 2000  # より長く実行
    
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
                  f"壁衝突率{stats['wall_hit_rate']:.1%}, "
                  f"モード切替{stats['mode_switches']}回")
    else:
        print(f"\n⏰ {max_steps}ステップで終了")
    
    stats = agent.get_statistics()
    
    visualize_results(
        maze, agent,
        "../results/15x15_contradiction.png"
    )
    
    print(f"\n📊 最終統計:")
    print(f"  ゴール到達: {'✅' if agent.is_goal_reached() else '❌'}")
    print(f"  総ステップ: {stats['steps']}")
    print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
    print(f"  ユニーク訪問: {stats['unique_visits']}")
    print(f"  エピソード数: {stats['episodes']}")
    print(f"  行き詰まり検出: {stats['stuck_positions']}箇所")
    print(f"  モード切替: {stats['mode_switches']}回")
    print(f"  矛盾検出: {stats['contradictions']}箇所")
    
    print("\n💡 分析:")
    print("- 矛盾検知により同じ失敗を繰り返さない")
    print("- 行き詰まり検出で脱出モードに切り替え")
    print("- 動的な探索率調整で局所最適を回避")


if __name__ == "__main__":
    test_contradiction_detection()