#!/usr/bin/env python3
"""メッセージパッシングgeDIGの完全ビジュアライゼーション"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import networkx as nx

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class PathMemory:
    """経路の記憶"""
    from_junction: Tuple[int, int]
    to_destination: Tuple[int, int]
    via_action: int
    is_dead_end: bool
    is_goal: bool
    path_length: int
    value: float
    path: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class PositionNode:
    """位置ノード"""
    position: Tuple[int, int]
    is_junction: bool = False
    is_dead_end: bool = False
    is_goal: bool = False
    is_start: bool = False
    neighbors: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    action_values: Dict[int, float] = field(default_factory=dict)
    visit_count: int = 0
    path_memories: List[PathMemory] = field(default_factory=list)
    
    def get_action_value(self, action: int) -> float:
        """行動の価値を取得"""
        return self.action_values.get(action, 0.0)


class VisualizedMessagePassingNavigator:
    """ビジュアライゼーション付きメッセージパッシングナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.nodes: Dict[Tuple[int, int], PositionNode] = {}
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.current_path: List[Tuple[int, int]] = []
        self.path_start_pos: Optional[Tuple[int, int]] = None
        self.path_start_action: Optional[int] = None
        self.visited_count: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # ビジュアライゼーション用
        self.history = []  # 各ステップの状態を記録
        
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
            from_junction=from_pos,
            to_destination=to_pos,
            via_action=action,
            path_length=len(path),
            is_dead_end=is_dead_end,
            is_goal=is_goal,
            value=10.0 / (len(path) + 1) if is_goal else (-10.0 if is_dead_end else 0.0),
            path=path.copy()
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
            print(f"💀 経路記録: {from_pos} --{['上','右','下','左'][action]}--> {to_pos} (行き止まり)")
        elif is_goal:
            print(f"🎯 経路記録: {from_pos} --{['上','右','下','左'][action]}--> {to_pos} (ゴール!)")
            
    def _propagate_values(self):
        """価値をネットワーク全体に伝播"""
        # 簡易的な価値伝播（3回の反復）
        for _ in range(3):
            for node in self.nodes.values():
                for action, neighbor_pos in node.neighbors.items():
                    if neighbor_pos in self.nodes:
                        neighbor = self.nodes[neighbor_pos]
                        
                        # 隣接ノードの価値を取得
                        if neighbor.is_goal:
                            neighbor_value = 10.0
                        elif neighbor.is_dead_end:
                            neighbor_value = -10.0
                        else:
                            max_value = max(neighbor.action_values.values()) if neighbor.action_values else 0.0
                            neighbor_value = max_value
                        
                        # 減衰させて伝播
                        propagated_value = neighbor_value * 0.8
                        
                        # 現在の価値と比較して更新
                        old_value = node.action_values.get(action, 0.0)
                        if abs(propagated_value) > abs(old_value):
                            node.action_values[action] = propagated_value
                            
    def decide_action(self, obs, maze) -> int:
        """行動を決定"""
        current_pos = obs.position
        current_node = self._get_or_create_node(current_pos)
        current_node.visit_count += 1
        self.visited_count[current_pos] += 1
        
        # ノード属性更新
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
            
        # 重要な地点での経路管理
        if obs.is_junction or current_node.visit_count == 1 or obs.is_dead_end:
            self.path_start_pos = current_pos
            self.current_path = [current_pos]
        else:
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
            
            # 情報利得
            if neighbor_pos not in self.nodes:
                ig_bonus = 3.0
            else:
                neighbor_visit = self.nodes[neighbor_pos].visit_count
                ig_bonus = 1.0 / (neighbor_visit + 1)
                
            # 最終スコア
            score = recorded_value - visit_penalty + self.config.k_ig * ig_bonus
            action_scores[action] = score
            
        # 最高スコアの行動を選択
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            
            if current_pos == self.path_start_pos:
                self.path_start_action = best_action
                
            # 現在の状態を記録
            self.history.append({
                'position': current_pos,
                'action': best_action,
                'nodes': {pos: {
                    'type': 'goal' if n.is_goal else 'dead_end' if n.is_dead_end else 'junction' if n.is_junction else 'normal',
                    'visit_count': n.visit_count,
                    'action_values': n.action_values.copy(),
                    'neighbors': n.neighbors.copy()
                } for pos, n in self.nodes.items()},
                'path_memories': [(m.from_junction, m.to_destination, m.is_dead_end, m.is_goal) 
                                 for node in self.nodes.values() for m in node.path_memories]
            })
                
            return best_action
        else:
            return np.random.choice([0, 1, 2, 3])


def create_visualization(maze, navigator, max_steps=500):
    """リアルタイムビジュアライゼーションを作成"""
    
    # 図の設定
    fig = plt.figure(figsize=(20, 10))
    
    # 迷路表示用
    ax_maze = plt.subplot(121)
    ax_maze.set_title('Maze Navigation', fontsize=16)
    
    # グラフ表示用
    ax_graph = plt.subplot(122)
    ax_graph.set_title('Knowledge Graph Construction', fontsize=16)
    
    # 迷路の初期描画
    def draw_maze(ax, maze_obj, agent_pos=None):
        ax.clear()
        ax.set_xlim(-0.5, maze_obj.width - 0.5)
        ax.set_ylim(-0.5, maze_obj.height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # グリッド描画
        for i in range(maze_obj.height):
            for j in range(maze_obj.width):
                if maze_obj.grid[i, j] == 1:  # 壁
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, facecolor='black')
                    ax.add_patch(rect)
                elif (i, j) == maze_obj.start_pos:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, facecolor='lightgreen')
                    ax.add_patch(rect)
                    ax.text(j, i, 'S', ha='center', va='center', fontsize=12, fontweight='bold')
                elif (i, j) == maze_obj.goal_pos:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, facecolor='lightcoral')
                    ax.add_patch(rect)
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=12, fontweight='bold')
                    
        # エージェント描画
        if agent_pos:
            circle = Circle((agent_pos[1], agent_pos[0]), 0.3, facecolor='blue', edgecolor='darkblue')
            ax.add_patch(circle)
            
        ax.set_xticks(range(maze_obj.width))
        ax.set_yticks(range(maze_obj.height))
        ax.grid(True, alpha=0.3)
        
    # ナビゲーション実行
    obs = maze.reset()
    navigator._get_or_create_node(maze.start_pos).is_start = True
    
    steps = 0
    done = False
    
    def animate(frame):
        nonlocal obs, done, steps
        
        if done or steps >= max_steps:
            return
            
        # 行動決定と実行
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        steps += 1
        
        # 迷路の更新
        draw_maze(ax_maze, maze, maze.agent_pos)
        
        # 訪問した位置を薄く表示
        for pos, count in navigator.visited_count.items():
            if count > 0 and pos != maze.agent_pos:
                alpha = min(0.5, count * 0.1)
                circle = Circle((pos[1], pos[0]), 0.2, facecolor='lightblue', alpha=alpha)
                ax_maze.add_patch(circle)
                
        # 現在の経路を表示
        if len(navigator.current_path) > 1:
            path = navigator.current_path
            for i in range(len(path) - 1):
                ax_maze.plot([path[i][1], path[i+1][1]], 
                           [path[i][0], path[i+1][0]], 
                           'b-', alpha=0.5, linewidth=2)
                
        # グラフの更新
        ax_graph.clear()
        
        if navigator.nodes:
            G = nx.DiGraph()
            
            # ノードを追加
            pos_dict = {}
            for pos, node in navigator.nodes.items():
                G.add_node(pos)
                pos_dict[pos] = (pos[1], -pos[0])  # 座標変換
                
            # エッジを追加（隣接関係）
            for pos, node in navigator.nodes.items():
                for action, neighbor_pos in node.neighbors.items():
                    if neighbor_pos in navigator.nodes:
                        G.add_edge(pos, neighbor_pos)
                        
            # レイアウト
            if len(G.nodes()) > 0:
                # ノードの色
                node_colors = []
                for pos in G.nodes():
                    node = navigator.nodes[pos]
                    if node.is_goal:
                        node_colors.append('gold')
                    elif node.is_dead_end:
                        node_colors.append('red')
                    elif node.is_junction:
                        node_colors.append('lightblue')
                    elif node.is_start:
                        node_colors.append('lightgreen')
                    else:
                        node_colors.append('lightgray')
                        
                # ノードサイズ（訪問回数に応じて）
                node_sizes = [300 + navigator.nodes[pos].visit_count * 100 for pos in G.nodes()]
                
                # 描画
                nx.draw_networkx_nodes(G, pos_dict, node_color=node_colors, 
                                     node_size=node_sizes, alpha=0.8, ax=ax_graph)
                
                # エッジの色（価値に応じて）
                for pos, node in navigator.nodes.items():
                    for action, neighbor_pos in node.neighbors.items():
                        if neighbor_pos in navigator.nodes:
                            value = node.get_action_value(action)
                            if value > 0:
                                ax_graph.annotate('', xy=pos_dict[neighbor_pos], 
                                                xytext=pos_dict[pos],
                                                arrowprops=dict(arrowstyle='->', 
                                                              color='green', 
                                                              alpha=min(1.0, value/10),
                                                              lw=2))
                            elif value < 0:
                                ax_graph.annotate('', xy=pos_dict[neighbor_pos], 
                                                xytext=pos_dict[pos],
                                                arrowprops=dict(arrowstyle='->', 
                                                              color='red', 
                                                              alpha=min(1.0, -value/10),
                                                              lw=2))
                                
                # ノードラベル
                labels = {}
                for pos in G.nodes():
                    node = navigator.nodes[pos]
                    if node.is_goal:
                        labels[pos] = 'G'
                    elif node.is_dead_end:
                        labels[pos] = 'X'
                    elif node.is_junction:
                        labels[pos] = 'J'
                    elif node.is_start:
                        labels[pos] = 'S'
                    else:
                        labels[pos] = f'{node.visit_count}'
                        
                nx.draw_networkx_labels(G, pos_dict, labels, font_size=10, ax=ax_graph)
                
        ax_graph.set_title(f'Knowledge Graph (Step {steps})', fontsize=16)
        ax_graph.axis('off')
        
        # 成功/失敗の表示
        if done:
            if maze.agent_pos == maze.goal_pos:
                fig.suptitle(f'SUCCESS! Goal reached in {steps} steps', fontsize=20, color='green')
            else:
                fig.suptitle(f'Failed after {steps} steps', fontsize=20, color='red')
                
        plt.tight_layout()
        
    # アニメーション作成
    anim = animation.FuncAnimation(fig, animate, frames=max_steps, 
                                 interval=100, repeat=False)
    
    return fig, anim


def run_visualization_experiment():
    """ビジュアライゼーション実験を実行"""
    print("メッセージパッシングgeDIGビジュアライゼーション実験")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    config.k_ig = 1.0
    
    # 適度なサイズの迷路で実験
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    navigator = VisualizedMessagePassingNavigator(config)
    
    print(f"迷路サイズ: {maze.size}")
    print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
    print("-" * 40)
    
    # ビジュアライゼーション作成
    fig, anim = create_visualization(maze, navigator, max_steps=200)
    
    # GIFとして保存
    print("\nGIFを作成中...")
    anim.save('message_passing_visualization.gif', writer='pillow', fps=5)
    print("✅ message_passing_visualization.gif として保存しました")
    
    # 静止画も保存
    plt.savefig('message_passing_final_state.png', dpi=150, bbox_inches='tight')
    print("✅ message_passing_final_state.png として最終状態を保存しました")
    
    plt.show()


if __name__ == "__main__":
    run_visualization_experiment()