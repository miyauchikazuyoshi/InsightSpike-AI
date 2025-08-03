#!/usr/bin/env python3
"""軽量版メッセージパッシングビジュアライゼーション"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


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
    dead_end_actions: Set[int] = field(default_factory=set)
    goal_path_actions: Set[int] = field(default_factory=set)


class LightweightNavigator:
    """軽量版ナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.nodes: Dict[Tuple[int, int], PositionNode] = {}
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.current_path: List[Tuple[int, int]] = []
        self.path_start_pos: Optional[Tuple[int, int]] = None
        self.path_start_action: Optional[int] = None
        
    def _get_or_create_node(self, pos: Tuple[int, int]) -> PositionNode:
        if pos not in self.nodes:
            self.nodes[pos] = PositionNode(position=pos)
        return self.nodes[pos]
        
    def _propagate_dead_end(self, path: List[Tuple[int, int]], start_action: int):
        """行き止まり情報を伝播"""
        if len(path) < 2:
            return
            
        # 開始点に行き止まり情報を記録
        start_node = self._get_or_create_node(path[0])
        start_node.dead_end_actions.add(start_action)
        start_node.action_values[start_action] = -10.0
        
    def _propagate_goal_path(self, path: List[Tuple[int, int]], start_action: int):
        """ゴール経路情報を伝播"""
        if len(path) < 2:
            return
            
        # 開始点にゴール経路情報を記録
        start_node = self._get_or_create_node(path[0])
        start_node.goal_path_actions.add(start_action)
        start_node.action_values[start_action] = 10.0
        
    def decide_action(self, obs, maze) -> int:
        """行動を決定"""
        current_pos = obs.position
        current_node = self._get_or_create_node(current_pos)
        current_node.visit_count += 1
        
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
            if self.path_start_pos and self.path_start_action is not None:
                self._propagate_goal_path(self.current_path, self.path_start_action)
                
        # 行き止まり到達
        if obs.is_dead_end and self.path_start_pos and self.path_start_action is not None:
            self._propagate_dead_end(self.current_path, self.path_start_action)
            
        # 経路管理
        if obs.is_junction or current_node.visit_count == 1 or obs.is_dead_end:
            self.path_start_pos = current_pos
            self.current_path = [current_pos]
        else:
            self.current_path.append(current_pos)
            
        # 行動選択
        action_scores = {}
        for action in obs.possible_moves:
            score = current_node.action_values.get(action, 0.0)
            
            # 未探索ボーナス
            neighbor_pos = current_node.neighbors[action]
            if neighbor_pos not in self.nodes:
                score += 2.0
                
            action_scores[action] = score
            
        # 最高スコアの行動を選択
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            if current_pos == self.path_start_pos:
                self.path_start_action = best_action
            return best_action
        else:
            return np.random.choice(obs.possible_moves)


def create_snapshot_visualization(maze, navigator, steps=50):
    """指定ステップ数ごとのスナップショットを作成"""
    
    # 実行
    obs = maze.reset()
    navigator._get_or_create_node(maze.start_pos).is_start = True
    
    snapshots = []
    snapshot_steps = [10, 25, 50, 100, 150]  # スナップショットを取るステップ
    current_step = 0
    
    for step in range(max(snapshot_steps) + 1):
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        current_step += 1
        
        # スナップショット作成
        if current_step in snapshot_steps or (done and maze.agent_pos == maze.goal_pos):
            snapshots.append({
                'step': current_step,
                'agent_pos': maze.agent_pos,
                'nodes': dict(navigator.nodes),
                'goal_found': navigator.goal_pos is not None,
                'success': done and maze.agent_pos == maze.goal_pos
            })
            
        if done:
            break
            
    # 可視化
    n_snapshots = len(snapshots)
    fig, axes = plt.subplots(2, n_snapshots, figsize=(5*n_snapshots, 10))
    
    if n_snapshots == 1:
        axes = axes.reshape(2, 1)
    
    for idx, snapshot in enumerate(snapshots):
        # 上段：迷路
        ax_maze = axes[0, idx]
        ax_maze.set_title(f'Step {snapshot["step"]}', fontsize=12)
        draw_maze_state(ax_maze, maze, snapshot['agent_pos'], snapshot['nodes'])
        
        # 下段：グラフ
        ax_graph = axes[1, idx]
        draw_graph_state(ax_graph, snapshot['nodes'], snapshot['goal_found'])
        
        if snapshot['success']:
            ax_maze.set_title(f'Step {snapshot["step"]} - SUCCESS!', fontsize=12, color='green')
            
    plt.tight_layout()
    plt.savefig('message_passing_snapshots.png', dpi=150, bbox_inches='tight')
    print("✅ message_passing_snapshots.png として保存しました")
    
    return snapshots


def draw_maze_state(ax, maze, agent_pos, nodes):
    """迷路の状態を描画"""
    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(-0.5, maze.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # グリッド描画
    for i in range(maze.height):
        for j in range(maze.width):
            if maze.grid[i, j] == 1:  # 壁
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, facecolor='black')
                ax.add_patch(rect)
            elif (i, j) == maze.start_pos:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, facecolor='lightgreen', alpha=0.5)
                ax.add_patch(rect)
                ax.text(j, i, 'S', ha='center', va='center', fontsize=10, fontweight='bold')
            elif (i, j) == maze.goal_pos:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, facecolor='lightcoral', alpha=0.5)
                ax.add_patch(rect)
                ax.text(j, i, 'G', ha='center', va='center', fontsize=10, fontweight='bold')
                
    # 訪問したノードを表示
    for pos, node in nodes.items():
        if node.visit_count > 0:
            if node.is_dead_end:
                color = 'red'
                marker = 'x'
            elif node.is_junction:
                color = 'blue'
                marker = 'o'
            else:
                color = 'gray'
                marker = '.'
            ax.plot(pos[1], pos[0], marker=marker, color=color, markersize=8, alpha=0.6)
            
    # エージェント
    if agent_pos:
        circle = patches.Circle((agent_pos[1], agent_pos[0]), 0.3, facecolor='darkblue')
        ax.add_patch(circle)
        
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)


def draw_graph_state(ax, nodes, goal_found):
    """グラフの状態を描画"""
    if not nodes:
        return
        
    G = nx.DiGraph()
    
    # ノードを追加
    pos_dict = {}
    for pos, node in nodes.items():
        G.add_node(pos)
        # グラフレイアウト用の座標
        pos_dict[pos] = (pos[1], -pos[0])
        
    # エッジを追加
    edge_colors = []
    edge_widths = []
    
    for pos, node in nodes.items():
        for action, neighbor_pos in node.neighbors.items():
            if neighbor_pos in nodes:
                G.add_edge(pos, neighbor_pos)
                
                # エッジの色と太さ
                value = node.action_values.get(action, 0.0)
                if value > 0:
                    edge_colors.append('green')
                    edge_widths.append(2)
                elif value < 0:
                    edge_colors.append('red')
                    edge_widths.append(2)
                else:
                    edge_colors.append('gray')
                    edge_widths.append(1)
                    
    # ノードの色
    node_colors = []
    for pos in G.nodes():
        node = nodes[pos]
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
            
    # 描画
    if len(G.nodes()) > 0:
        nx.draw(G, pos_dict, node_color=node_colors, edge_color=edge_colors,
                width=edge_widths, node_size=300, with_labels=False, ax=ax, 
                arrows=True, arrowsize=10)
                
        # ラベル
        labels = {}
        for pos in G.nodes():
            node = nodes[pos]
            if node.is_start:
                labels[pos] = 'S'
            elif node.is_goal:
                labels[pos] = 'G'
            elif node.is_dead_end:
                labels[pos] = 'X'
            elif node.is_junction:
                labels[pos] = 'J'
                
        nx.draw_networkx_labels(G, pos_dict, labels, font_size=8, ax=ax)
        
    ax.set_title('Knowledge Graph', fontsize=10)
    ax.axis('off')


def run_lightweight_experiment():
    """軽量版実験を実行"""
    print("メッセージパッシングgeDIG軽量版ビジュアライゼーション")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    config.k_ig = 1.0
    
    # 複数の迷路でテスト
    for seed in [42, 100, 200]:
        print(f"\n実験 {seed//100 + 1}")
        print("-" * 40)
        
        np.random.seed(seed)
        maze = SimpleMaze(size=(12, 12), maze_type='dfs')
        navigator = LightweightNavigator(config)
        
        print(f"迷路サイズ: {maze.size}")
        print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
        
        # スナップショット作成
        snapshots = create_snapshot_visualization(maze, navigator)
        
        # 結果表示
        final_snapshot = snapshots[-1]
        if final_snapshot['success']:
            print(f"✅ 成功！ステップ数: {final_snapshot['step']}")
        else:
            print(f"❌ 失敗（{final_snapshot['step']}ステップ後）")
            
        print(f"訪問ノード数: {len(final_snapshot['nodes'])}")
        dead_ends = sum(1 for n in final_snapshot['nodes'].values() if n.is_dead_end)
        junctions = sum(1 for n in final_snapshot['nodes'].values() if n.is_junction)
        print(f"発見した行き止まり: {dead_ends}")
        print(f"発見した分岐点: {junctions}")
        
        # 個別の画像として保存
        plt.savefig(f'experiment_{seed//100 + 1}_snapshots.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    print("\n" + "=" * 60)
    print("✨ グラフ構築の様子をスナップショットで確認できます")
    print("✨ 行き止まり（赤）の情報が分岐点に伝播される様子が分かります")
    print("✨ ゴール発見後は緑のエッジで最適経路が示されます")


if __name__ == "__main__":
    run_lightweight_experiment()