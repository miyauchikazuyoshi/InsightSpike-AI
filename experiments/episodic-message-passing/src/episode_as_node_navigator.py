#!/usr/bin/env python3
"""エピソード自体をノードとして扱うナビゲーター"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class EpisodeNode:
    """エピソードノード（エピソード自体がノード）"""
    # エピソード情報
    episode_type: str  # "goal_info" or "movement"
    content: Dict      # エピソードの内容
    vector: np.ndarray # ベクトル表現
    value: float = 0.0 # エピソードの価値
    
    # グラフ構造
    node_id: int = -1
    connected_episodes: List['EpisodeNode'] = None
    
    def __post_init__(self):
        if self.connected_episodes is None:
            self.connected_episodes = []
    
    def __str__(self):
        if self.episode_type == "goal_info":
            return f"Goal: {self.content['position']}"
        else:
            from_pos = self.content['from']
            to_pos = self.content['to']
            result = self.content['result']
            action_str = ['↑', '→', '↓', '←'][self.content['action']]
            return f"Move[{self.node_id}]: {from_pos}{action_str}{to_pos}({result})"


class EpisodeAsNodeNavigator:
    """エピソードをノードとして扱うナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: List[EpisodeNode] = []
        self.episode_counter = 0
        self.current_path: List[EpisodeNode] = []
        self.visited_positions: Set[Tuple[int, int]] = set()
        
    def add_goal_info(self, goal_pos: Tuple[int, int]) -> EpisodeNode:
        """ゴール情報エピソードを追加"""
        content = {"position": goal_pos}
        vector = np.array([goal_pos[0], goal_pos[1], 100.0])  # ゴールは特別な値
        
        episode = EpisodeNode(
            episode_type="goal_info",
            content=content,
            vector=vector,
            value=10.0,
            node_id=self.episode_counter
        )
        self.episodes.append(episode)
        self.episode_counter += 1
        print(f"📍 ゴール情報追加: {goal_pos}")
        return episode
        
    def add_movement_episode(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                           action: int, result: str) -> EpisodeNode:
        """移動エピソードを追加"""
        content = {
            "from": from_pos,
            "to": to_pos,
            "action": action,
            "result": result
        }
        
        # 移動をベクトル化
        vector = np.array([
            from_pos[0], 
            from_pos[1],
            to_pos[0],
            to_pos[1],
            1.0 if result == "成功" else -1.0
        ])
        
        # 価値の設定
        if result == "成功":
            value = 1.0
        elif result == "壁":
            value = -5.0
        else:  # 行き止まり
            value = -10.0
            
        episode = EpisodeNode(
            episode_type="movement",
            content=content,
            vector=vector,
            value=value,
            node_id=self.episode_counter
        )
        self.episodes.append(episode)
        self.episode_counter += 1
        
        # 経路上のエピソードを接続
        if self.current_path and result == "成功":
            last_episode = self.current_path[-1]
            last_episode.connected_episodes.append(episode)
            self.current_path.append(episode)
        elif result != "成功":
            # 失敗したらパスをリセット
            self.current_path = []
        else:
            # 新しいパスの開始
            self.current_path = [episode]
        
        # デバッグ出力
        action_str = ['↑', '→', '↓', '←'][action]
        print(f"   {from_pos} {action_str} {to_pos}: {result} (Episode {episode.node_id})")
        
        return episode
        
    def find_similar_episodes(self, current_pos: Tuple[int, int]) -> List[Tuple[EpisodeNode, float]]:
        """現在位置から実行可能な移動エピソードを検索"""
        similarities = []
        
        for episode in self.episodes:
            if episode.episode_type == "movement" and episode.content['from'] == current_pos:
                # 同じ位置からの移動エピソード
                # 価値に基づいてスコアを計算
                score = episode.value
                
                # 訪問済みの場所への移動は避ける
                if episode.content['to'] in self.visited_positions:
                    score -= 2.0
                    
                similarities.append((episode, score))
                
        # スコアでソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
        
    def decide_action(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """エピソード記憶に基づいて次の行動を決定"""
        self.visited_positions.add(current_pos)
        
        # クエリ表示
        print(f"\n🤔 クエリ: ゴールに辿り着くために、現在位置{current_pos}で次に移動すべき方向は？")
        
        # 類似エピソード検索
        similar_episodes = self.find_similar_episodes(current_pos)
        
        # 各行動のスコアを計算
        action_scores = {}
        tried_actions = set()
        
        for episode, score in similar_episodes:
            action = episode.content['action']
            if action in possible_actions:
                tried_actions.add(action)
                # 既存のスコアと比較して最大値を取る
                if action not in action_scores or score > action_scores[action]:
                    action_scores[action] = score
                    
        # 未試行の行動に探索ボーナス
        for action in possible_actions:
            if action not in tried_actions:
                action_scores[action] = 3.0  # 探索ボーナス
                
        # 最高スコアの行動を選択
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            action_str = ['↑', '→', '↓', '←'][best_action]
            print(f"   決定: {action_str} (スコア: {action_scores[best_action]:.2f})")
            return best_action
        else:
            # ランダムに選択
            return np.random.choice(possible_actions)
            
    def propagate_messages(self, end_episode: EpisodeNode, message_type: str):
        """エピソードグラフを遡ってメッセージを伝播"""
        print(f"\n📨 メッセージパッシング: {message_type}")
        
        # 幅優先探索で接続されたエピソードを遡る
        visited = set()
        queue = [(end_episode, 0)]
        
        while queue:
            current_episode, distance = queue.pop(0)
            
            if current_episode.node_id in visited:
                continue
                
            visited.add(current_episode.node_id)
            
            # メッセージに応じて価値を更新
            if message_type == "dead_end":
                # 距離に応じて減衰
                penalty = -5.0 / (distance + 1)
                current_episode.value = min(current_episode.value + penalty, -5.0)
                print(f"   Episode {current_episode.node_id}: {current_episode} → 価値: {current_episode.value:.2f}")
                
            elif message_type == "goal_path":
                # 距離に応じて減衰
                reward = 5.0 / (distance + 1)
                current_episode.value = max(current_episode.value + reward, 5.0)
                print(f"   Episode {current_episode.node_id}: {current_episode} → 価値: {current_episode.value:.2f}")
                
            # 前のエピソードを探す（このエピソードに接続しているもの）
            for other_episode in self.episodes:
                if current_episode in other_episode.connected_episodes:
                    queue.append((other_episode, distance + 1))


def visualize_episode_graph(navigator: 'EpisodeAsNodeNavigator'):
    """エピソードグラフを可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左側：移動の軌跡
    ax1.set_title("Movement Trajectories", fontsize=14)
    ax1.set_aspect('equal')
    
    # 移動エピソードを矢印で表示
    for episode in navigator.episodes:
        if episode.episode_type == "movement":
            from_pos = episode.content['from']
            to_pos = episode.content['to']
            
            # 色分け
            if episode.value > 0:
                color = 'blue'
                alpha = 0.7
            elif episode.value < -5:
                color = 'red'
                alpha = 0.7
            else:
                color = 'gray'
                alpha = 0.3
                
            # 矢印を描画
            ax1.annotate('', xy=to_pos, xytext=from_pos,
                        arrowprops=dict(arrowstyle='->', color=color, 
                                      alpha=alpha, lw=2))
            
    # ゴール位置
    goal_episodes = [e for e in navigator.episodes if e.episode_type == "goal_info"]
    if goal_episodes:
        goal_pos = goal_episodes[0].content['position']
        ax1.plot(goal_pos[0], goal_pos[1], 'go', markersize=15, label='Goal')
        
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右側：エピソードグラフ
    ax2.set_title("Episode Graph", fontsize=14)
    
    # NetworkXグラフを構築
    G = nx.DiGraph()
    
    # ノードを追加
    for episode in navigator.episodes:
        G.add_node(episode.node_id, episode=episode)
        
    # エッジを追加
    for episode in navigator.episodes:
        for connected in episode.connected_episodes:
            G.add_edge(episode.node_id, connected.node_id)
            
    # レイアウト
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # ノードの色
        node_colors = []
        for node_id in G.nodes():
            episode = navigator.episodes[node_id]
            if episode.episode_type == "goal_info":
                node_colors.append('yellow')
            elif episode.value > 0:
                node_colors.append('lightblue')
            elif episode.value < -5:
                node_colors.append('red')
            else:
                node_colors.append('lightgray')
                
        # 描画
        nx.draw(G, pos, node_color=node_colors, node_size=500,
                with_labels=True, ax=ax2, arrows=True,
                edge_color='gray', alpha=0.7)
                
        # ラベル
        labels = {}
        for node_id in G.nodes():
            episode = navigator.episodes[node_id]
            if episode.episode_type == "goal_info":
                labels[node_id] = "Goal"
            else:
                action_str = ['↑', '→', '↓', '←'][episode.content['action']]
                labels[node_id] = f"{action_str}\n{episode.content['result'][:1]}"
                
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax2)
        
    plt.tight_layout()
    return fig


def run_experiment():
    """実験を実行"""
    print("エピソードをノードとするナビゲーター実験")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    navigator = EpisodeAsNodeNavigator(config)
    
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
        
        # メッセージパッシング
        if obs.is_dead_end:
            # 行き止まり情報を伝播
            navigator.propagate_messages(episode, "dead_end")
            
        # ゴール到達
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            # ゴール経路情報を伝播
            navigator.propagate_messages(episode, "goal_path")
            break
            
    else:
        print(f"\n❌ タイムアウト（{max_steps}ステップ）")
        
    # 統計表示
    print("\n統計情報:")
    print(f"  総エピソード数: {len(navigator.episodes)}")
    print(f"  訪問位置数: {len(navigator.visited_positions)}")
    movement_episodes = [e for e in navigator.episodes if e.episode_type == "movement"]
    print(f"  移動エピソード数: {len(movement_episodes)}")
    
    # 結果別集計
    results = {"成功": 0, "壁": 0, "行き止まり": 0}
    for e in movement_episodes:
        results[e.content['result']] += 1
    print(f"  結果内訳: {results}")
    
    # グラフ可視化
    fig = visualize_episode_graph(navigator)
    fig.savefig('episode_graph_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ episode_graph_visualization.png として保存しました")


if __name__ == "__main__":
    run_experiment()