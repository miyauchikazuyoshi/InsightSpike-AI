#!/usr/bin/env python3
"""geDIG理論に基づいたナビゲーター（ΔGED×ΔIG実装）"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import math

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class GedigEpisodeNode:
    """geDIG理論に基づくエピソードノード"""
    # エピソード情報
    episode_type: str  # "goal_info" or "movement"
    content: Dict      # エピソードの内容
    vector: np.ndarray # ベクトル表現
    
    # geDIG関連の値
    ged_delta: float = 0.0    # ΔGED（グラフ編集距離の変化）
    ig_delta: float = 0.0     # ΔIG（情報利得）
    gedig_value: float = 0.0  # ΔGED × ΔIG
    
    # グラフ構造
    node_id: int = -1
    connected_episodes: List['GedigEpisodeNode'] = field(default_factory=list)
    
    # 状態空間情報（情報利得計算用）
    position_entropy: float = 0.0
    action_entropy: float = 0.0
    
    def __str__(self):
        if self.episode_type == "goal_info":
            return f"Goal: {self.content['position']}"
        else:
            from_pos = self.content['from']
            to_pos = self.content['to']
            result = self.content['result']
            action_str = ['↑', '→', '↓', '←'][self.content['action']]
            return f"Move[{self.node_id}]: {from_pos}{action_str}{to_pos}({result}) GED:{self.ged_delta:.3f} IG:{self.ig_delta:.3f}"


class GedigBasedNavigator:
    """geDIG理論に基づくナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: List[GedigEpisodeNode] = []
        self.episode_counter = 0
        self.current_path: List[GedigEpisodeNode] = []
        self.visited_positions: Set[Tuple[int, int]] = set()
        
        # グラフ構造
        self.episode_graph = nx.DiGraph()
        
        # 位置-行動の統計情報（情報利得計算用）
        self.position_action_counts = defaultdict(lambda: defaultdict(int))
        self.position_visit_counts = defaultdict(int)
        self.action_success_counts = defaultdict(lambda: defaultdict(int))
        
    def calculate_ged_delta(self, new_episode: GedigEpisodeNode) -> float:
        """新しいエピソードによるグラフ編集距離の変化を計算"""
        if len(self.episode_graph.nodes()) == 0:
            # 最初のノードは最大の新規性
            return 1.0
            
        # 新しいエピソードがもたらす構造的新規性を評価
        # 1. 新しい位置からの移動か？
        if new_episode.episode_type == "movement":
            from_pos = new_episode.content['from']
            to_pos = new_episode.content['to']
            
            # 既存のエピソードと比較
            structural_novelty = 1.0
            
            for episode in self.episodes:
                if episode.episode_type == "movement":
                    # 同じ位置からの移動が既にある場合
                    if episode.content['from'] == from_pos:
                        structural_novelty *= 0.8
                        
                        # 同じ方向への移動がある場合
                        if episode.content['action'] == new_episode.content['action']:
                            structural_novelty *= 0.5
                            
                            # 同じ結果の場合
                            if episode.content['result'] == new_episode.content['result']:
                                structural_novelty *= 0.3
                                
            # グラフの連結性への影響
            connectivity_impact = 0.0
            
            # 新しい位置への到達
            if to_pos not in self.visited_positions:
                connectivity_impact += 0.3
                
            # 新しいパスの形成
            if self.current_path:
                # 既存パスからの分岐
                connectivity_impact += 0.2
                
            return structural_novelty + connectivity_impact
            
        return 0.1  # goal_info等
        
    def calculate_ig_delta(self, new_episode: GedigEpisodeNode) -> float:
        """新しいエピソードによる情報利得を計算"""
        if new_episode.episode_type != "movement":
            return 0.1
            
        from_pos = new_episode.content['from']
        action = new_episode.content['action']
        result = new_episode.content['result']
        
        # 事前エントロピー：この位置からの行動の不確実性
        prior_entropy = self._calculate_position_entropy(from_pos)
        
        # 事後エントロピー：新しい情報を得た後の不確実性
        # 行動結果を記録
        self.position_action_counts[from_pos][action] += 1
        self.position_visit_counts[from_pos] += 1
        self.action_success_counts[from_pos][action] += (1 if result == "成功" else 0)
        
        posterior_entropy = self._calculate_position_entropy(from_pos)
        
        # 情報利得 = エントロピーの減少
        ig = prior_entropy - posterior_entropy
        
        # 結果の意外性によるボーナス
        if result == "行き止まり":
            # 行き止まりの発見は価値が高い
            ig += 0.3
        elif result == "壁" and self.position_action_counts[from_pos][action] == 1:
            # 初めて壁を発見
            ig += 0.2
            
        return max(0.0, ig)
        
    def _calculate_position_entropy(self, position: Tuple[int, int]) -> float:
        """特定位置での行動選択のエントロピーを計算"""
        if self.position_visit_counts[position] == 0:
            # 未訪問位置は最大エントロピー
            return math.log(4)  # 4方向の一様分布
            
        # 各行動の成功確率を推定
        action_probs = []
        for action in range(4):
            count = self.position_action_counts[position][action]
            if count > 0:
                success_rate = self.action_success_counts[position][action] / count
                action_probs.append(success_rate)
            else:
                # 未試行の行動は0.5の確率を仮定
                action_probs.append(0.5)
                
        # エントロピー計算
        entropy = 0.0
        for p in action_probs:
            if p > 0 and p < 1:
                entropy -= p * math.log(p) + (1-p) * math.log(1-p)
                
        return entropy
        
    def add_goal_info(self, goal_pos: Tuple[int, int]) -> GedigEpisodeNode:
        """ゴール情報エピソードを追加"""
        content = {"position": goal_pos}
        vector = np.array([goal_pos[0], goal_pos[1], 100.0])
        
        episode = GedigEpisodeNode(
            episode_type="goal_info",
            content=content,
            vector=vector,
            node_id=self.episode_counter
        )
        
        # geDIG値の計算
        episode.ged_delta = 1.0  # ゴール情報は常に新規性が高い
        episode.ig_delta = 1.0   # ゴール情報は情報価値が高い
        episode.gedig_value = episode.ged_delta * episode.ig_delta
        
        self.episodes.append(episode)
        self.episode_graph.add_node(episode.node_id, episode=episode)
        self.episode_counter += 1
        
        print(f"📍 ゴール情報追加: {goal_pos} (geDIG値: {episode.gedig_value:.3f})")
        return episode
        
    def add_movement_episode(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                           action: int, result: str) -> GedigEpisodeNode:
        """移動エピソードを追加（geDIG理論に基づく価値計算）"""
        content = {
            "from": from_pos,
            "to": to_pos,
            "action": action,
            "result": result
        }
        
        # 移動をベクトル化（拡張版）
        vector = np.array([
            from_pos[0], 
            from_pos[1],
            to_pos[0],
            to_pos[1],
            1.0 if result == "成功" else -1.0,
            float(action),  # 行動も含める
            len(self.visited_positions)  # 探索の進行度
        ])
        
        episode = GedigEpisodeNode(
            episode_type="movement",
            content=content,
            vector=vector,
            node_id=self.episode_counter
        )
        
        # geDIG値の計算
        episode.ged_delta = self.calculate_ged_delta(episode)
        episode.ig_delta = self.calculate_ig_delta(episode)
        episode.gedig_value = episode.ged_delta * episode.ig_delta
        
        self.episodes.append(episode)
        self.episode_graph.add_node(episode.node_id, episode=episode)
        self.episode_counter += 1
        
        # グラフ構造の更新
        if self.current_path and result == "成功":
            last_episode = self.current_path[-1]
            self.episode_graph.add_edge(last_episode.node_id, episode.node_id)
            self.current_path.append(episode)
        elif result != "成功":
            self.current_path = []
        else:
            self.current_path = [episode]
        
        # デバッグ出力
        action_str = ['↑', '→', '↓', '←'][action]
        print(f"   {from_pos} {action_str} {to_pos}: {result}")
        print(f"   ΔGED: {episode.ged_delta:.3f}, ΔIG: {episode.ig_delta:.3f}, geDIG: {episode.gedig_value:.3f}")
        
        return episode
        
    def decide_action(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """geDIG値に基づいて次の行動を決定"""
        self.visited_positions.add(current_pos)
        
        print(f"\n🤔 クエリ: 現在位置{current_pos}での最適行動は？")
        
        # 各行動の期待geDIG値を計算
        action_values = {}
        
        for action in possible_actions:
            # 過去の経験から期待値を推定
            expected_ged = 1.0  # デフォルト値
            expected_ig = 0.5
            
            # 同じ位置・行動の過去エピソードを検索
            for episode in self.episodes:
                if (episode.episode_type == "movement" and 
                    episode.content['from'] == current_pos and 
                    episode.content['action'] == action):
                    # 過去の経験から学習
                    expected_ged *= 0.7  # 既知の行動は新規性が低い
                    expected_ig = episode.ig_delta * 0.8
                    
            # 未試行ボーナス（UCB的アプローチ）
            trial_count = self.position_action_counts[current_pos][action]
            if trial_count == 0:
                exploration_bonus = 2.0
            else:
                exploration_bonus = math.sqrt(2 * math.log(self.position_visit_counts[current_pos] + 1) / trial_count)
                
            # 期待geDIG値 + 探索ボーナス
            action_values[action] = expected_ged * expected_ig + exploration_bonus
            
        # 最高値の行動を選択（ε-greedy的な要素も追加可能）
        best_action = max(action_values.items(), key=lambda x: x[1])[0]
        action_str = ['↑', '→', '↓', '←'][best_action]
        print(f"   決定: {action_str} (期待値: {action_values[best_action]:.3f})")
        
        return best_action
        
    def propagate_gedig_gradient(self, end_episode: GedigEpisodeNode, gradient_type: str):
        """geDIG勾配をグラフ上で伝播"""
        print(f"\n📊 geDIG勾配伝播: {gradient_type}")
        
        # PageRank的なアプローチで価値を伝播
        if gradient_type == "goal_path":
            # ゴール到達パスは正の勾配
            base_gradient = 1.0
        else:
            # 行き止まりは負の勾配（ただし情報価値はある）
            base_gradient = -0.5
            
        # 逆方向にトラバース
        visited = set()
        queue = [(end_episode, base_gradient)]
        
        while queue:
            current_episode, gradient = queue.pop(0)
            
            if current_episode.node_id in visited:
                continue
                
            visited.add(current_episode.node_id)
            
            # geDIG値を勾配で更新
            learning_rate = 0.1
            current_episode.gedig_value += learning_rate * gradient
            
            print(f"   Episode {current_episode.node_id}: 勾配 {gradient:.3f} → geDIG値 {current_episode.gedig_value:.3f}")
            
            # 前のエピソードに減衰した勾配を伝播
            for pred_id in self.episode_graph.predecessors(current_episode.node_id):
                pred_episode = self.episodes[pred_id]
                decayed_gradient = gradient * 0.9  # 減衰率
                queue.append((pred_episode, decayed_gradient))


def visualize_gedig_graph(navigator: 'GedigBasedNavigator'):
    """geDIG値を含むグラフを可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 左側：geDIG値のヒートマップ
    ax1.set_title("geDIG Values Heatmap", fontsize=14)
    ax1.set_aspect('equal')
    
    # 位置ごとの最大geDIG値を集計
    position_gedig = defaultdict(float)
    for episode in navigator.episodes:
        if episode.episode_type == "movement":
            pos = episode.content['to']
            position_gedig[pos] = max(position_gedig[pos], episode.gedig_value)
            
    # ヒートマップ描画
    if position_gedig:
        positions = list(position_gedig.keys())
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        values = [position_gedig[p] for p in positions]
        
        scatter = ax1.scatter(x_coords, y_coords, c=values, cmap='coolwarm', 
                            s=300, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, ax=ax1, label='geDIG value')
        
    # 移動軌跡
    for episode in navigator.episodes:
        if episode.episode_type == "movement" and episode.content['result'] == "成功":
            from_pos = episode.content['from']
            to_pos = episode.content['to']
            ax1.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                    'b-', alpha=0.3, linewidth=1)
            
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.3)
    
    # 右側：エピソードグラフ（geDIG値で色分け）
    ax2.set_title("Episode Graph (colored by geDIG)", fontsize=14)
    
    G = navigator.episode_graph
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # ノードの色（geDIG値に基づく）
        node_colors = []
        node_sizes = []
        for node_id in G.nodes():
            episode = navigator.episodes[node_id]
            # geDIG値を色に変換
            gedig_normalized = min(max(episode.gedig_value, -1), 1)  # -1〜1に正規化
            color_value = (gedig_normalized + 1) / 2  # 0〜1に変換
            node_colors.append(plt.cm.coolwarm(color_value))
            node_sizes.append(300 + abs(episode.gedig_value) * 200)
            
        # 描画
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=node_sizes, ax=ax2)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                             alpha=0.5, arrows=True, ax=ax2)
        
        # ラベル（geDIG値を表示）
        labels = {}
        for node_id in G.nodes():
            episode = navigator.episodes[node_id]
            if episode.episode_type == "goal_info":
                labels[node_id] = "Goal"
            else:
                labels[node_id] = f"{node_id}\n{episode.gedig_value:.2f}"
                
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax2)
        
    plt.tight_layout()
    return fig


def run_gedig_experiment():
    """geDIG理論に基づく実験を実行"""
    print("geDIG理論ベースのナビゲーター実験")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    navigator = GedigBasedNavigator(config)
    
    # 迷路生成
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    
    print(f"迷路サイズ: {maze.size}")
    print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
    print("-" * 60)
    
    # 1. ゴール情報エピソードを追加
    navigator.add_goal_info(maze.goal_pos)
    
    # 2. メインループ
    obs = maze.reset()
    steps = 0
    max_steps = 100
    
    while steps < max_steps:
        current_pos = obs.position
        
        # 行動決定
        action = navigator.decide_action(current_pos, obs.possible_moves)
        
        # 行動実行
        old_pos = current_pos
        obs, reward, done, info = maze.step(action)
        new_pos = obs.position
        steps += 1
        
        # 移動結果の判定
        if old_pos == new_pos:
            result = "壁"
        elif obs.is_dead_end:
            result = "行き止まり"
        else:
            result = "成功"
            
        # エピソード形成
        episode = navigator.add_movement_episode(old_pos, new_pos, action, result)
        
        # geDIG勾配伝播
        if obs.is_dead_end:
            navigator.propagate_gedig_gradient(episode, "dead_end")
            
        # ゴール到達
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            navigator.propagate_gedig_gradient(episode, "goal_path")
            break
            
    else:
        print(f"\n❌ タイムアウト（{max_steps}ステップ）")
        
    # 統計表示
    print("\n統計情報:")
    print(f"  総エピソード数: {len(navigator.episodes)}")
    print(f"  訪問位置数: {len(navigator.visited_positions)}")
    
    # geDIG値の統計
    gedig_values = [e.gedig_value for e in navigator.episodes if e.episode_type == "movement"]
    if gedig_values:
        print(f"  平均geDIG値: {np.mean(gedig_values):.3f}")
        print(f"  最大geDIG値: {np.max(gedig_values):.3f}")
        print(f"  最小geDIG値: {np.min(gedig_values):.3f}")
    
    # グラフ可視化
    fig = visualize_gedig_graph(navigator)
    fig.savefig('gedig_graph_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ gedig_graph_visualization.png として保存しました")


if __name__ == "__main__":
    run_gedig_experiment()