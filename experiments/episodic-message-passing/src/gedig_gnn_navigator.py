#!/usr/bin/env python3
"""PyTorchGeometricを使ったGNNベースのgeDIGナビゲーター"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import math

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


class EpisodeGNN(torch.nn.Module):
    """エピソードグラフ用のGNN"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch=None):
        # 1st layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # 2nd layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # 3rd layer
        x = self.conv3(x, edge_index)
        
        if batch is not None:
            # グラフレベルの出力
            x = global_mean_pool(x, batch)
            
        return x


@dataclass
class GNNEpisodeNode:
    """GNN用のエピソードノード"""
    episode_type: str
    content: Dict
    vector: np.ndarray
    
    # geDIG関連
    ged_delta: float = 0.0
    ig_delta: float = 0.0
    gedig_value: float = 0.0
    gnn_value: float = 0.0  # GNNによる予測値
    
    # グラフ構造
    node_id: int = -1
    connected_episodes: List[int] = field(default_factory=list)  # IDのリスト
    
    # 位置情報（高速アクセス用）
    position: Optional[Tuple[int, int]] = None
    
    def to_tensor(self) -> torch.Tensor:
        """ノード特徴量をテンソルに変換"""
        # ベクトル + geDIG関連の値を特徴量とする
        features = np.concatenate([
            self.vector,
            [self.ged_delta, self.ig_delta, self.gedig_value, self.gnn_value]
        ])
        return torch.tensor(features, dtype=torch.float32)


class GedigGNNNavigator:
    """GNNベースのgeDIGナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: List[GNNEpisodeNode] = []
        self.episode_counter = 0
        self.current_path: List[int] = []  # エピソードIDのリスト
        self.visited_positions: Set[Tuple[int, int]] = set()
        
        # 位置ベースのインデックス（高速検索用）
        self.position_episodes: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # 統計情報
        self.position_action_counts = defaultdict(lambda: defaultdict(int))
        self.position_visit_counts = defaultdict(int)
        self.action_success_counts = defaultdict(lambda: defaultdict(int))
        
        # GNN
        self.gnn = EpisodeGNN(input_dim=11, hidden_dim=64, output_dim=1)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.001)
        
    def _build_graph_data(self) -> Data:
        """現在のエピソードグラフをPyG Dataオブジェクトに変換"""
        if not self.episodes:
            return None
            
        # ノード特徴量
        node_features = torch.stack([ep.to_tensor() for ep in self.episodes])
        
        # エッジリスト構築
        edge_list = []
        for i, episode in enumerate(self.episodes):
            for j in episode.connected_episodes:
                edge_list.append([i, j])
                
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # エッジがない場合は空のテンソル
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 自己ループを追加（GATのため）
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(self.episodes))
        
        return Data(x=node_features, edge_index=edge_index)
        
    def _update_gnn_values(self):
        """GNNを使ってエピソードの価値を更新"""
        if len(self.episodes) < 5:  # 少なすぎる場合はスキップ
            return
            
        # グラフデータ構築
        data = self._build_graph_data()
        if data is None:
            return
            
        # GNN推論
        self.gnn.eval()
        with torch.no_grad():
            values = self.gnn(data.x, data.edge_index)
            
        # 値を各エピソードに反映
        for i, episode in enumerate(self.episodes):
            episode.gnn_value = values[i].item()
            
    def calculate_ged_delta(self, new_episode: GNNEpisodeNode) -> float:
        """グラフ編集距離の変化を計算（改良版）"""
        if len(self.episodes) == 0:
            return 1.0
            
        structural_novelty = 1.0
        
        if new_episode.episode_type == "movement":
            from_pos = new_episode.content['from']
            to_pos = new_episode.content['to']
            action = new_episode.content['action']
            
            # 同じ位置からの既存エピソードをチェック
            for ep_id in self.position_episodes.get(from_pos, []):
                existing = self.episodes[ep_id]
                if existing.episode_type == "movement":
                    structural_novelty *= 0.8
                    
                    if existing.content['action'] == action:
                        structural_novelty *= 0.5
                        
                        if existing.content['result'] == new_episode.content['result']:
                            structural_novelty *= 0.3
                            
            # グラフ構造への影響を評価
            connectivity_impact = 0.0
            
            # 新しい位置への到達
            if to_pos not in self.visited_positions:
                connectivity_impact += 0.3
                
            # 新しいパスの形成（既存ノードからの分岐数）
            if self.current_path:
                last_episode_id = self.current_path[-1]
                branch_count = len(self.episodes[last_episode_id].connected_episodes)
                connectivity_impact += 0.2 / (branch_count + 1)
                
            # グラフの密度への影響
            if len(self.episodes) > 0:
                current_density = sum(len(ep.connected_episodes) for ep in self.episodes) / len(self.episodes)
                connectivity_impact += 0.1 * (1 - min(current_density / 4, 1))  # 密度が低いほど価値が高い
                
        return structural_novelty + connectivity_impact
        
    def calculate_ig_delta(self, new_episode: GNNEpisodeNode) -> float:
        """情報利得を計算（改良版）"""
        if new_episode.episode_type != "movement":
            return 0.1
            
        from_pos = new_episode.content['from']
        action = new_episode.content['action']
        result = new_episode.content['result']
        
        # 事前エントロピー
        prior_entropy = self._calculate_position_entropy(from_pos)
        
        # 統計更新
        self.position_action_counts[from_pos][action] += 1
        self.position_visit_counts[from_pos] += 1
        self.action_success_counts[from_pos][action] += (1 if result == "成功" else 0)
        
        # 事後エントロピー
        posterior_entropy = self._calculate_position_entropy(from_pos)
        
        # 基本的な情報利得
        ig = prior_entropy - posterior_entropy
        
        # 結果の意外性ボーナス
        if result == "行き止まり":
            ig += 0.3
            # 周囲の行き止まり密度も考慮
            nearby_dead_ends = sum(1 for ep in self.episodes 
                                 if ep.episode_type == "movement" 
                                 and ep.content['result'] == "行き止まり"
                                 and abs(ep.content['from'][0] - from_pos[0]) <= 2
                                 and abs(ep.content['from'][1] - from_pos[1]) <= 2)
            ig += 0.1 * (1 / (nearby_dead_ends + 1))
            
        elif result == "壁" and self.position_action_counts[from_pos][action] == 1:
            ig += 0.2
            
        return max(0.0, ig)
        
    def _calculate_position_entropy(self, position: Tuple[int, int]) -> float:
        """位置でのエントロピー計算"""
        if self.position_visit_counts[position] == 0:
            return math.log(4)
            
        action_probs = []
        for action in range(4):
            count = self.position_action_counts[position][action]
            if count > 0:
                success_rate = self.action_success_counts[position][action] / count
                action_probs.append(success_rate)
            else:
                action_probs.append(0.5)
                
        entropy = 0.0
        for p in action_probs:
            if 0 < p < 1:
                entropy -= p * math.log(p) + (1-p) * math.log(1-p)
                
        return entropy
        
    def add_movement_episode(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                           action: int, result: str) -> GNNEpisodeNode:
        """移動エピソードを追加"""
        content = {
            "from": from_pos,
            "to": to_pos,
            "action": action,
            "result": result
        }
        
        # 拡張ベクトル表現
        vector = np.array([
            from_pos[0], from_pos[1],
            to_pos[0], to_pos[1],
            1.0 if result == "成功" else -1.0,
            float(action),
            len(self.visited_positions) / 100.0  # 正規化された探索進行度
        ])
        
        episode = GNNEpisodeNode(
            episode_type="movement",
            content=content,
            vector=vector,
            node_id=self.episode_counter,
            position=from_pos
        )
        
        # geDIG値計算
        episode.ged_delta = self.calculate_ged_delta(episode)
        episode.ig_delta = self.calculate_ig_delta(episode)
        episode.gedig_value = episode.ged_delta * episode.ig_delta
        
        # エピソード追加
        self.episodes.append(episode)
        self.position_episodes[from_pos].append(self.episode_counter)
        
        # グラフ構造更新
        if self.current_path and result == "成功":
            last_episode_id = self.current_path[-1]
            self.episodes[last_episode_id].connected_episodes.append(self.episode_counter)
            self.current_path.append(self.episode_counter)
        elif result != "成功":
            self.current_path = []
        else:
            self.current_path = [self.episode_counter]
            
        self.episode_counter += 1
        
        # GNN値の更新（バッチ処理のため、一定間隔で実行）
        if self.episode_counter % 10 == 0:
            self._update_gnn_values()
            
        # デバッグ出力
        action_str = ['↑', '→', '↓', '←'][action]
        print(f"   {from_pos} {action_str} {to_pos}: {result}")
        print(f"   ΔGED: {episode.ged_delta:.3f}, ΔIG: {episode.ig_delta:.3f}, "
              f"geDIG: {episode.gedig_value:.3f}, GNN: {episode.gnn_value:.3f}")
        
        return episode
        
    def decide_action(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """GNNとgeDIG値を組み合わせて行動決定"""
        self.visited_positions.add(current_pos)
        
        print(f"\n🤔 クエリ: 現在位置{current_pos}での最適行動は？")
        
        # 最新のGNN値を取得
        self._update_gnn_values()
        
        action_values = {}
        
        for action in possible_actions:
            # 基本的な期待値
            expected_ged = 1.0
            expected_ig = 0.5
            gnn_estimate = 0.0
            
            # 同じ位置・行動の過去エピソードから学習
            for ep_id in self.position_episodes.get(current_pos, []):
                episode = self.episodes[ep_id]
                if (episode.episode_type == "movement" and 
                    episode.content['action'] == action):
                    expected_ged *= 0.7
                    expected_ig = episode.ig_delta * 0.8
                    gnn_estimate = episode.gnn_value  # GNNの推定値を利用
                    
            # UCB的な探索ボーナス
            trial_count = self.position_action_counts[current_pos][action]
            if trial_count == 0:
                exploration_bonus = 2.0
            else:
                exploration_bonus = math.sqrt(2 * math.log(self.position_visit_counts[current_pos] + 1) / trial_count)
                
            # 統合値：geDIG期待値 + GNN推定 + 探索ボーナス
            action_values[action] = (expected_ged * expected_ig + gnn_estimate * 0.5 + exploration_bonus)
            
        # ε-greedy的な要素を追加（探索と活用のバランス）
        epsilon = 0.1
        if np.random.random() < epsilon:
            best_action = np.random.choice(possible_actions)
            print(f"   ランダム探索: {['↑', '→', '↓', '←'][best_action]}")
        else:
            best_action = max(action_values.items(), key=lambda x: x[1])[0]
            action_str = ['↑', '→', '↓', '←'][best_action]
            print(f"   決定: {action_str} (価値: {action_values[best_action]:.3f})")
            
        return best_action
        
    def train_gnn_on_episode_batch(self, target_episodes: List[int], target_values: List[float]):
        """エピソードバッチでGNNを訓練"""
        if len(self.episodes) < 10:
            return
            
        data = self._build_graph_data()
        if data is None:
            return
            
        self.gnn.train()
        self.optimizer.zero_grad()
        
        # 予測
        predictions = self.gnn(data.x, data.edge_index)
        
        # ターゲットテンソル作成
        targets = torch.zeros_like(predictions)
        for ep_id, value in zip(target_episodes, target_values):
            targets[ep_id] = value
            
        # 損失計算（MSE）
        loss = F.mse_loss(predictions[target_episodes], targets[target_episodes])
        
        # バックプロパゲーション
        loss.backward()
        self.optimizer.step()
        
        print(f"   GNN訓練損失: {loss.item():.4f}")
        
    def propagate_gedig_gradient_with_gnn(self, end_episode_id: int, gradient_type: str):
        """GNNを使った勾配伝播と学習"""
        print(f"\n📊 GNN強化geDIG勾配伝播: {gradient_type}")
        
        # 基本勾配設定
        base_gradient = 1.0 if gradient_type == "goal_path" else -0.5
        
        # 逆方向トラバースで影響を受けるエピソードを収集
        visited = set()
        queue = [(end_episode_id, base_gradient)]
        training_episodes = []
        training_values = []
        
        while queue:
            current_id, gradient = queue.pop(0)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            current_episode = self.episodes[current_id]
            
            # geDIG値を勾配で更新
            learning_rate = 0.1
            current_episode.gedig_value += learning_rate * gradient
            
            # 訓練データとして記録
            training_episodes.append(current_id)
            training_values.append(current_episode.gedig_value)
            
            print(f"   Episode {current_id}: 勾配 {gradient:.3f} → geDIG値 {current_episode.gedig_value:.3f}")
            
            # 前のエピソードを探す
            for i, ep in enumerate(self.episodes):
                if current_id in ep.connected_episodes:
                    decayed_gradient = gradient * 0.9
                    queue.append((i, decayed_gradient))
                    
        # GNNを訓練
        if training_episodes:
            self.train_gnn_on_episode_batch(training_episodes, training_values)


def visualize_gnn_graph(navigator: 'GedigGNNNavigator'):
    """GNN強化されたグラフを可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. geDIG値ヒートマップ
    ax = axes[0, 0]
    ax.set_title("geDIG Values Heatmap", fontsize=14)
    ax.set_aspect('equal')
    
    position_gedig = defaultdict(float)
    for episode in navigator.episodes:
        if episode.episode_type == "movement":
            pos = episode.content['to']
            position_gedig[pos] = max(position_gedig[pos], episode.gedig_value)
            
    if position_gedig:
        positions = list(position_gedig.keys())
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        values = [position_gedig[p] for p in positions]
        scatter = ax.scatter(x_coords, y_coords, c=values, cmap='coolwarm', 
                           s=300, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='geDIG value')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    
    # 2. GNN予測値ヒートマップ
    ax = axes[0, 1]
    ax.set_title("GNN Predicted Values", fontsize=14)
    ax.set_aspect('equal')
    
    position_gnn = defaultdict(float)
    for episode in navigator.episodes:
        if episode.episode_type == "movement":
            pos = episode.content['to']
            position_gnn[pos] = max(position_gnn[pos], episode.gnn_value)
            
    if position_gnn:
        positions = list(position_gnn.keys())
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        values = [position_gnn[p] for p in positions]
        scatter = ax.scatter(x_coords, y_coords, c=values, cmap='viridis', 
                           s=300, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='GNN value')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    
    # 3. エピソードグラフ（geDIG値）
    ax = axes[1, 0]
    ax.set_title("Episode Graph (geDIG)", fontsize=14)
    
    G = nx.DiGraph()
    for episode in navigator.episodes:
        G.add_node(episode.node_id)
    for episode in navigator.episodes:
        for connected_id in episode.connected_episodes:
            G.add_edge(episode.node_id, connected_id)
            
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=2, iterations=50)
        node_colors = []
        for node_id in G.nodes():
            episode = navigator.episodes[node_id]
            gedig_normalized = min(max(episode.gedig_value, -1), 1)
            color_value = (gedig_normalized + 1) / 2
            node_colors.append(plt.cm.coolwarm(color_value))
        nx.draw(G, pos, node_color=node_colors, node_size=300,
               with_labels=True, ax=ax, arrows=True, edge_color='gray', alpha=0.7)
               
    # 4. エピソードグラフ（GNN値）
    ax = axes[1, 1]
    ax.set_title("Episode Graph (GNN)", fontsize=14)
    
    if len(G.nodes()) > 0:
        node_colors = []
        for node_id in G.nodes():
            episode = navigator.episodes[node_id]
            gnn_normalized = min(max(episode.gnn_value, -1), 1)
            color_value = (gnn_normalized + 1) / 2
            node_colors.append(plt.cm.viridis(color_value))
        nx.draw(G, pos, node_color=node_colors, node_size=300,
               with_labels=True, ax=ax, arrows=True, edge_color='gray', alpha=0.7)
    
    plt.tight_layout()
    return fig


def run_gnn_experiment():
    """GNNベースのgeDIG実験を実行"""
    print("GNNベースgeDIGナビゲーター実験")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    navigator = GedigGNNNavigator(config)
    
    # 迷路生成
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    
    print(f"迷路サイズ: {maze.size}")
    print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
    print("-" * 60)
    
    # ゴール情報エピソードを追加
    goal_episode = GNNEpisodeNode(
        episode_type="goal_info",
        content={"position": maze.goal_pos},
        vector=np.array([maze.goal_pos[0], maze.goal_pos[1], 100.0, 0, 0, 0, 0]),
        ged_delta=1.0,
        ig_delta=1.0,
        gedig_value=1.0,
        node_id=navigator.episode_counter,
        position=maze.goal_pos
    )
    navigator.episodes.append(goal_episode)
    navigator.episode_counter += 1
    print(f"📍 ゴール情報追加: {maze.goal_pos}")
    
    # メインループ
    obs = maze.reset()
    steps = 0
    max_steps = 150  # GNN学習のため少し長めに
    
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
        
        # 勾配伝播
        if obs.is_dead_end:
            navigator.propagate_gedig_gradient_with_gnn(episode.node_id, "dead_end")
            
        # ゴール到達
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            navigator.propagate_gedig_gradient_with_gnn(episode.node_id, "goal_path")
            break
            
    else:
        print(f"\n❌ タイムアウト（{max_steps}ステップ）")
        
    # 最終的なGNN更新
    navigator._update_gnn_values()
    
    # 統計表示
    print("\n統計情報:")
    print(f"  総エピソード数: {len(navigator.episodes)}")
    print(f"  訪問位置数: {len(navigator.visited_positions)}")
    
    # geDIG値とGNN値の相関
    gedig_values = []
    gnn_values = []
    for ep in navigator.episodes:
        if ep.episode_type == "movement":
            gedig_values.append(ep.gedig_value)
            gnn_values.append(ep.gnn_value)
            
    if gedig_values:
        correlation = np.corrcoef(gedig_values, gnn_values)[0, 1] if len(gedig_values) > 1 else 0
        print(f"  geDIG-GNN相関: {correlation:.3f}")
        print(f"  平均geDIG値: {np.mean(gedig_values):.3f}")
        print(f"  平均GNN値: {np.mean(gnn_values):.3f}")
    
    # グラフ可視化
    fig = visualize_gnn_graph(navigator)
    fig.savefig('gedig_gnn_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ gedig_gnn_visualization.png として保存しました")


if __name__ == "__main__":
    run_gnn_experiment()