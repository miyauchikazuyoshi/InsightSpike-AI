#!/usr/bin/env python3
"""改良版メッセージパッシングナビゲーター（行き止まり回避強化）"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class PathMemory:
    """経路の記憶：どこからどこへ到達したか"""
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    action: int
    path_length: int
    is_dead_end: bool
    is_goal: bool
    
    @property
    def value(self) -> float:
        """経路の価値"""
        if self.is_goal:
            return 10.0 / (self.path_length + 1)
        elif self.is_dead_end:
            return -10.0
        else:
            return 0.0


@dataclass 
class PositionNode:
    """位置ノード：メッセージパッシングの単位"""
    position: Tuple[int, int]
    is_junction: bool = False
    is_dead_end: bool = False
    is_goal: bool = False
    is_start: bool = False
    neighbors: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # action -> value のメッセージ
    action_values: Dict[int, float] = field(default_factory=dict)
    visit_count: int = 0
    path_memories: List[PathMemory] = field(default_factory=list)
    
    def get_action_value(self, action: int) -> float:
        """行動の価値を取得"""
        if action in self.action_values:
            return self.action_values[action]
        return 0.0


class ImprovedMessagePassingNavigator:
    """改良版メッセージパッシングナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.nodes: Dict[Tuple[int, int], PositionNode] = {}
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.current_path: List[Tuple[int, int]] = []
        self.path_start_pos: Optional[Tuple[int, int]] = None
        self.path_start_action: Optional[int] = None
        self.visited_count: Dict[Tuple[int, int], int] = defaultdict(int)
        
    def _get_or_create_node(self, pos: Tuple[int, int]) -> PositionNode:
        """ノードを取得または作成"""
        if pos not in self.nodes:
            self.nodes[pos] = PositionNode(position=pos)
        return self.nodes[pos]
        
    def _record_path_result(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                           action: int, path: List[Tuple[int, int]], 
                           is_dead_end: bool, is_goal: bool):
        """経路の結果を記録"""
        memory = PathMemory(
            from_pos=from_pos,
            to_pos=to_pos,
            action=action,
            path_length=len(path),
            is_dead_end=is_dead_end,
            is_goal=is_goal
        )
        
        # 開始ノードに記憶を追加
        from_node = self._get_or_create_node(from_pos)
        from_node.path_memories.append(memory)
        
        # その行動の価値を更新
        current_value = from_node.action_values.get(action, 0.0)
        from_node.action_values[action] = current_value + memory.value
        
        # 終点ノードの属性も更新
        to_node = self._get_or_create_node(to_pos)
        to_node.is_dead_end = is_dead_end
        to_node.is_goal = is_goal
        
        if is_dead_end:
            print(f"💀 経路記録: {from_pos} --{['上','右','下','左'][action]}--> {to_pos} (行き止まり, 長さ{len(path)})")
        elif is_goal:
            print(f"🎯 経路記録: {from_pos} --{['上','右','下','左'][action]}--> {to_pos} (ゴール!, 長さ{len(path)})")
            
    def _propagate_values(self):
        """価値をネットワーク全体に伝播"""
        # BFSで価値を伝播
        changed = True
        iterations = 0
        
        while changed and iterations < 10:
            changed = False
            iterations += 1
            
            for node in self.nodes.values():
                # 隣接ノードから価値を集める
                for action, neighbor_pos in node.neighbors.items():
                    if neighbor_pos in self.nodes:
                        neighbor = self.nodes[neighbor_pos]
                        
                        # 隣接ノードの最大価値を取得
                        if neighbor.is_goal:
                            neighbor_value = 10.0
                        elif neighbor.is_dead_end:
                            neighbor_value = -10.0
                        else:
                            neighbor_value = max(neighbor.action_values.values()) if neighbor.action_values else 0.0
                        
                        # 減衰させて伝播
                        propagated_value = neighbor_value * 0.8
                        
                        # 現在の価値と比較して更新
                        old_value = node.action_values.get(action, 0.0)
                        if abs(propagated_value - old_value) > 0.1:
                            node.action_values[action] = propagated_value
                            changed = True
                            
    def decide_action(self, obs, maze) -> int:
        """メッセージパッシングに基づいて行動を決定"""
        current_pos = obs.position
        current_node = self._get_or_create_node(current_pos)
        current_node.visit_count += 1
        self.visited_count[current_pos] += 1
        
        # ノードの属性を更新
        current_node.is_junction = obs.is_junction
        current_node.is_dead_end = obs.is_dead_end
        current_node.is_goal = obs.is_goal
        
        # 隣接ノードを記録
        for action in obs.possible_moves:
            dx, dy = maze.ACTIONS[action]
            neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
            current_node.neighbors[action] = neighbor_pos
            
        # ゴール発見
        if obs.is_goal and not self.goal_pos:
            self.goal_pos = current_pos
            print(f"🎯 ゴール発見！位置: {current_pos}")
            if self.path_start_pos and self.path_start_action is not None:
                self._record_path_result(
                    self.path_start_pos, current_pos, self.path_start_action,
                    self.current_path, is_dead_end=False, is_goal=True
                )
                
        # 行き止まり到達
        if obs.is_dead_end and self.path_start_pos and self.path_start_action is not None:
            self._record_path_result(
                self.path_start_pos, current_pos, self.path_start_action,
                self.current_path, is_dead_end=True, is_goal=False
            )
            
        # 重要な地点（分岐点、開始位置、前の経路の終点）での経路管理
        if obs.is_junction or current_node.is_start or \
           (self.path_start_pos and len(obs.possible_moves) >= 2 and current_pos != self.path_start_pos):
            # 新しい経路の開始
            self.path_start_pos = current_pos
            self.current_path = [current_pos]
        else:
            # 経路の継続
            if current_pos not in self.current_path:
                self.current_path.append(current_pos)
                
        # 価値を伝播
        if len(self.nodes) > 1:
            self._propagate_values()
            
        # 各行動の評価
        action_scores = {}
        
        for action in obs.possible_moves:
            # 記録された価値
            recorded_value = current_node.get_action_value(action)
            
            # 訪問回数によるペナルティ
            neighbor_pos = current_node.neighbors[action]
            visit_penalty = self.visited_count[neighbor_pos] * 0.5
            
            # 情報利得（未探索ボーナス）
            if neighbor_pos not in self.nodes:
                ig_bonus = 3.0
            else:
                neighbor_visit = self.nodes[neighbor_pos].visit_count
                ig_bonus = 1.0 / (neighbor_visit + 1)
                
            # 最終スコア
            score = recorded_value - visit_penalty + self.config.k_ig * ig_bonus
            action_scores[action] = score
            
        # デバッグ情報（重要な地点のみ）
        if current_node.visit_count <= 2 or obs.is_junction or len(obs.possible_moves) == 1:
            print(f"\n位置{current_pos}での行動評価 (訪問{current_node.visit_count}回目):")
            for a in obs.possible_moves:
                dir_name = ['上','右','下','左'][a]
                score = action_scores[a]
                recorded = current_node.get_action_value(a)
                print(f"  {dir_name}: スコア={score:.2f} (記録値={recorded:.2f})")
                
        # 最高スコアの行動を選択
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        if obs.is_junction or current_node.is_start:
            self.path_start_action = best_action
            
        return best_action
        
    def visualize_knowledge_graph(self, filename='improved_message_passing.png'):
        """知識グラフの可視化"""
        if not self.nodes:
            return
            
        G = nx.DiGraph()
        
        # ノードを追加
        for pos, node in self.nodes.items():
            label = f"{pos}\nV:{node.visit_count}"
            if node.is_goal:
                G.add_node(pos, label=label, color='gold', size=1000)
            elif node.is_dead_end:
                G.add_node(pos, label=label, color='red', size=700)
            elif node.is_junction:
                G.add_node(pos, label=label, color='lightblue', size=800)
            else:
                G.add_node(pos, label=label, color='lightgray', size=500)
                
        # エッジを追加（価値で色分け）
        for pos, node in self.nodes.items():
            for action, neighbor_pos in node.neighbors.items():
                if neighbor_pos in self.nodes:
                    value = node.get_action_value(action)
                    if value > 0:
                        G.add_edge(pos, neighbor_pos, 
                                 weight=value, color='green', 
                                 label=f"{['↑','→','↓','←'][action]}")
                    elif value < 0:
                        G.add_edge(pos, neighbor_pos, 
                                 weight=-value, color='red',
                                 label=f"{['↑','→','↓','←'][action]}")
                    else:
                        G.add_edge(pos, neighbor_pos, 
                                 weight=0.1, color='gray',
                                 label=f"{['↑','→','↓','←'][action]}")
                        
        # レイアウトと描画
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # ノードの描画
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=node_sizes, alpha=0.8)
        
        # エッジの描画
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                             width=edge_weights, alpha=0.6, 
                             arrows=True, arrowsize=20)
        
        # ラベルの描画
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Improved Message Passing Knowledge Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()


def demonstrate_improved_message_passing():
    """改良版メッセージパッシングのデモ"""
    print("改良版メッセージパッシングナビゲーション")
    print("=" * 60)
    print("特徴：")
    print("- 経路単位での記憶（開始点→終点）")
    print("- 行き止まり情報の価値伝播")
    print("- 訪問回数によるペナルティ")
    print("- 効率的な探索と回避")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    config.k_ig = 1.0  # 探索ボーナスを調整
    
    # 複数試行でテスト
    n_trials = 3
    results = []
    
    for trial in range(n_trials):
        print(f"\n試行 {trial + 1}/{n_trials}")
        print("-" * 40)
        
        np.random.seed(trial + 42)
        maze = SimpleMaze(size=(10, 10), maze_type='dfs')
        navigator = ImprovedMessagePassingNavigator(config)
        
        # 開始ノードをマーク
        start_node = navigator._get_or_create_node(maze.start_pos)
        start_node.is_start = True
        
        print(f"迷路: {maze.size}")
        print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
        
        obs = maze.reset()
        steps = 0
        
        for _ in range(500):
            action = navigator.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            steps += 1
            
            if done and maze.agent_pos == maze.goal_pos:
                print(f"\n✅ ゴール到達！ステップ数: {steps}")
                results.append({
                    'success': True,
                    'steps': steps,
                    'nodes': len(navigator.nodes),
                    'dead_ends_found': sum(1 for n in navigator.nodes.values() if n.is_dead_end)
                })
                break
        else:
            print(f"\n❌ タイムアウト（{steps}ステップ）")
            results.append({
                'success': False,
                'steps': steps,
                'nodes': len(navigator.nodes),
                'dead_ends_found': sum(1 for n in navigator.nodes.values() if n.is_dead_end)
            })
            
        # 最後の試行の知識グラフを可視化
        if trial == n_trials - 1:
            navigator.visualize_knowledge_graph()
            
        # 統計情報
        print(f"\n探索統計:")
        print(f"  訪問ノード数: {len(navigator.nodes)}")
        print(f"  発見した行き止まり: {results[-1]['dead_ends_found']}")
        
        # 価値の高い経路を表示
        valuable_paths = []
        for node in navigator.nodes.values():
            for mem in node.path_memories:
                if mem.is_goal:
                    valuable_paths.append((mem, "ゴール"))
                elif mem.is_dead_end:
                    valuable_paths.append((mem, "行き止まり"))
                    
        if valuable_paths:
            print(f"\n重要な経路記憶:")
            for mem, type_str in valuable_paths[:5]:
                print(f"  {mem.from_pos} → {mem.to_pos} ({type_str}, 長さ{mem.path_length})")
                
    # 全体の統計
    print("\n" + "=" * 60)
    print("全試行の結果:")
    success_count = sum(1 for r in results if r['success'])
    success_results = [r for r in results if r['success']]
    
    print(f"成功率: {success_count}/{n_trials} ({success_count/n_trials*100:.0f}%)")
    if success_results:
        avg_steps = np.mean([r['steps'] for r in success_results])
        avg_nodes = np.mean([r['nodes'] for r in success_results])
        avg_dead_ends = np.mean([r['dead_ends_found'] for r in success_results])
        print(f"平均ステップ数（成功時）: {avg_steps:.1f}")
        print(f"平均探索ノード数: {avg_nodes:.1f}")
        print(f"平均発見行き止まり数: {avg_dead_ends:.1f}")
        
    print("\n" + "=" * 60)
    print("✨ メッセージパッシングの利点:")
    print("✨ 行き止まり情報が経路全体に伝播")
    print("✨ 一度の失敗から効率的に学習")
    print("✨ 分散的な知識共有で賢い探索")


if __name__ == "__main__":
    demonstrate_improved_message_passing()