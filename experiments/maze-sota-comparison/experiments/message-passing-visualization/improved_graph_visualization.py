#!/usr/bin/env python3
"""改良版グラフビジュアライゼーション：エピソード記憶の色分け"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class Episode:
    """エピソード記憶"""
    query: str
    context: Tuple[int, int]
    action: int
    result: Tuple[int, int]
    value: float
    episode_type: str  # 'wall_collision', 'goal_query', 'action_guidance', 'successful_move'


@dataclass
class EpisodeNode:
    """エピソードノード"""
    position: Tuple[int, int]
    episodes: List[Episode] = field(default_factory=list)
    is_junction: bool = False
    is_dead_end: bool = False
    is_goal: bool = False
    is_start: bool = False
    neighbors: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    visit_count: int = 0


class EpisodeGraphNavigator:
    """エピソード記憶グラフナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.nodes: Dict[Tuple[int, int], EpisodeNode] = {}
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.episodes: List[Episode] = []
        
    def _get_or_create_node(self, pos: Tuple[int, int]) -> EpisodeNode:
        if pos not in self.nodes:
            self.nodes[pos] = EpisodeNode(position=pos)
        return self.nodes[pos]
        
    def _create_episode(self, query: str, context: Tuple[int, int], 
                       action: int, result: Tuple[int, int], 
                       success: bool, hit_wall: bool, found_goal: bool) -> Episode:
        """エピソードを作成"""
        # エピソードタイプを判定
        if query.startswith("ゴール"):
            episode_type = "goal_query"
            value = 10.0 if found_goal else 1.0
        elif hit_wall:
            episode_type = "wall_collision"
            value = -10.0
        elif context != result and success:
            episode_type = "successful_move"
            value = 5.0
        else:
            episode_type = "action_guidance"
            value = 0.0
            
        return Episode(
            query=query,
            context=context,
            action=action,
            result=result,
            value=value,
            episode_type=episode_type
        )
        
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
            # ゴール到達エピソードを作成
            episode = self._create_episode(
                "ゴール座標を目指せ",
                current_pos,
                -1,  # 到達済み
                current_pos,
                True,
                False,
                True
            )
            self.episodes.append(episode)
            current_node.episodes.append(episode)
            
        # 各行動の評価
        action_scores = {}
        
        # デモ用：最初の分岐点では意図的に間違った方向を選ぶ
        if current_node.is_junction and current_node.visit_count == 1:
            # 分岐点での最初の訪問時は、ランダムに選ぶ（行き止まりに行く可能性を上げる）
            return np.random.choice(obs.possible_moves)
        
        for action in obs.possible_moves:
            score = 0.0
            
            # エピソード記憶から評価
            relevant_episodes = [e for e in self.episodes 
                               if e.context == current_pos and e.action == action]
            
            for episode in relevant_episodes:
                if episode.episode_type == "wall_collision":
                    score -= 10.0
                elif episode.episode_type == "successful_move":
                    score += 5.0
                elif episode.episode_type == "goal_query" and episode.value > 0:
                    score += 10.0
                    
            # 未探索ボーナス（デモ用に弱める）
            neighbor_pos = current_node.neighbors[action]
            if neighbor_pos not in self.nodes:
                score += 1.0  # 3.0から1.0に減らす
                
            action_scores[action] = score
            
        # 最適行動を選択
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            return best_action
        else:
            return np.random.choice([0, 1, 2, 3])
            
    def update_after_action(self, old_pos: Tuple[int, int], action: int,
                           new_pos: Tuple[int, int], obs, hit_wall: bool):
        """行動後の更新"""
        # エピソードを作成
        query = "ゴール座標を目指せ" if not self.goal_pos else f"位置{old_pos}から次の行動は？"
        
        episode = self._create_episode(
            query=query,
            context=old_pos,
            action=action,
            result=new_pos,
            success=(old_pos != new_pos),
            hit_wall=hit_wall,
            found_goal=obs.is_goal
        )
        
        # エピソードを記録
        self.episodes.append(episode)
        old_node = self._get_or_create_node(old_pos)
        old_node.episodes.append(episode)
        
        # 行き止まりの場合、メッセージパッシング
        if obs.is_dead_end:
            # 分岐点まで遡って伝播
            for node in self.nodes.values():
                if node.is_junction:
                    # この分岐点から行き止まりへの経路があるか確認
                    guidance_episode = Episode(
                        query=f"分岐点{node.position}での行動指針",
                        context=node.position,
                        action=action,
                        result=new_pos,
                        value=-5.0,
                        episode_type="action_guidance"
                    )
                    node.episodes.append(guidance_episode)
                    self.episodes.append(guidance_episode)


def create_episode_graph_visualization(maze, navigator, max_steps=200):
    """エピソードグラフのビジュアライゼーション"""
    
    # ナビゲーション実行
    obs = maze.reset()
    navigator._get_or_create_node(maze.start_pos).is_start = True
    
    for step in range(max_steps):
        old_pos = obs.position
        action = navigator.decide_action(obs, maze)
        
        # 壁衝突チェック
        dx, dy = maze.ACTIONS[action]
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        hit_wall = (new_pos[0] < 0 or new_pos[0] >= maze.height or 
                   new_pos[1] < 0 or new_pos[1] >= maze.width or
                   maze.grid[new_pos[0], new_pos[1]] == 1)
        
        obs, reward, done, info = maze.step(action)
        navigator.update_after_action(old_pos, action, obs.position, obs, hit_wall)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"✅ ゴール到達！ステップ数: {step + 1}")
            break
            
    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左側：迷路
    ax1.set_title('Maze Navigation', fontsize=16)
    draw_maze_with_episodes(ax1, maze, navigator)
    
    # 右側：エピソードグラフ
    ax2.set_title('Episode Memory Graph', fontsize=16)
    draw_episode_graph(ax2, navigator)
    
    plt.tight_layout()
    return fig


def draw_maze_with_episodes(ax, maze, navigator):
    """エピソード情報付き迷路を描画"""
    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(-0.5, maze.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # 迷路描画
    for i in range(maze.height):
        for j in range(maze.width):
            if maze.grid[i, j] == 1:  # 壁
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor='black')
                ax.add_patch(rect)
            elif (i, j) == maze.start_pos:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor='lightgreen', alpha=0.5)
                ax.add_patch(rect)
                ax.text(j, i, 'S', ha='center', va='center', fontsize=12, fontweight='bold')
            elif (i, j) == maze.goal_pos:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor='yellow', alpha=0.5)
                ax.add_patch(rect)
                ax.text(j, i, 'G', ha='center', va='center', fontsize=12, fontweight='bold')
                
    # エピソードの軌跡を描画
    for episode in navigator.episodes:
        if episode.episode_type == "successful_move":
            # 成功した移動を青い線で
            ax.plot([episode.context[1], episode.result[1]], 
                   [episode.context[0], episode.result[0]], 
                   'b-', alpha=0.5, linewidth=2)
        elif episode.episode_type == "wall_collision":
            # 壁衝突を赤い×で
            ax.plot(episode.context[1], episode.context[0], 'rx', markersize=10)
            
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)


def draw_episode_graph(ax, navigator):
    """エピソード記憶グラフを描画"""
    G = nx.DiGraph()
    
    # ノードを追加（エピソードごと）
    node_colors = []
    node_sizes = []
    pos_dict = {}
    labels = {}
    
    # エピソードごとにノードを作成
    for idx, episode in enumerate(navigator.episodes):
        node_id = f"E{idx}"
        G.add_node(node_id)
        
        # 色分け
        if episode.episode_type == "wall_collision":
            node_colors.append('red')
            node_sizes.append(800)
        elif episode.episode_type == "goal_query":
            node_colors.append('yellow')
            node_sizes.append(1000)
        elif episode.episode_type == "action_guidance":
            node_colors.append('white')
            node_sizes.append(600)
        elif episode.episode_type == "successful_move":
            node_colors.append('lightblue')
            node_sizes.append(700)
        else:
            node_colors.append('gray')
            node_sizes.append(500)
            
        # ラベル
        action_str = ['↑', '→', '↓', '←'][episode.action] if episode.action >= 0 else 'G'
        labels[node_id] = f"{episode.context}\n{action_str}"
        
    # エッジを追加（エピソード間の関連性）
    for i, e1 in enumerate(navigator.episodes):
        for j, e2 in enumerate(navigator.episodes):
            if i != j:
                # 結果位置が次のコンテキストになる場合
                if e1.result == e2.context:
                    G.add_edge(f"E{i}", f"E{j}")
                # メッセージパッシングの関連
                elif e1.episode_type == "action_guidance" and e2.context == e1.context:
                    G.add_edge(f"E{i}", f"E{j}", style='dashed')
                    
    # レイアウト
    if len(G.nodes()) > 0:
        # 階層的レイアウト
        pos_dict = nx.spring_layout(G, k=3, iterations=50)
        
        # ノード描画
        nx.draw_networkx_nodes(G, pos_dict, node_color=node_colors, 
                             node_size=node_sizes, alpha=0.8, ax=ax)
        
        # エッジ描画
        solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') != 'dashed']
        dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dashed']
        
        nx.draw_networkx_edges(G, pos_dict, edgelist=solid_edges, 
                             edge_color='gray', alpha=0.5, ax=ax)
        nx.draw_networkx_edges(G, pos_dict, edgelist=dashed_edges, 
                             edge_color='orange', alpha=0.5, style='dashed', ax=ax)
        
        # ラベル描画
        nx.draw_networkx_labels(G, pos_dict, labels, font_size=8, ax=ax)
        
    # 凡例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=12, label='壁衝突エピソード'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                   markersize=12, label='ゴールクエリ'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                   markeredgecolor='black', markersize=12, label='行動指針（メッセージパッシング）'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=12, label='成功した移動'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.axis('off')


def run_improved_visualization():
    """改良版ビジュアライゼーションを実行"""
    print("エピソード記憶グラフビジュアライゼーション")
    print("=" * 60)
    print("色分け:")
    print("- 赤: 壁衝突エピソード")
    print("- 黄: ゴールクエリ")
    print("- 白: 行動指針（メッセージパッシング）")
    print("- 青: 成功した移動")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    config.k_ig = 1.0
    
    # 3つの異なる迷路で実験
    for seed in [42, 100, 200]:
        print(f"\n実験 {seed//50}:")
        print("-" * 40)
        
        np.random.seed(seed)
        maze = SimpleMaze(size=(10, 10), maze_type='dfs')
        navigator = EpisodeGraphNavigator(config)
        
        print(f"迷路サイズ: {maze.size}")
        print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
        
        # ビジュアライゼーション作成
        fig = create_episode_graph_visualization(maze, navigator, max_steps=200)
        
        # 保存
        filename = f'episode_graph_visualization_{seed//50}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ {filename} として保存しました")
        
        # エピソード統計
        print(f"\nエピソード統計:")
        episode_types = defaultdict(int)
        for e in navigator.episodes:
            episode_types[e.episode_type] += 1
            
        for etype, count in episode_types.items():
            print(f"  {etype}: {count}個")
            
        plt.close(fig)
        
    print("\n" + "=" * 60)
    print("✨ エピソード記憶の色分けビジュアライゼーション完成！")
    print("✨ メッセージパッシングによる行動指針の伝播が可視化されました")


if __name__ == "__main__":
    run_improved_visualization()