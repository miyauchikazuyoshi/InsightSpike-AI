#!/usr/bin/env python3
"""geDIG Episode Navigator with Frontier Exploration: 未探索領域を優先的に探索"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class EpisodeNode:
    """エピソードノード（エピソード自体がグラフのノード）"""
    node_id: int
    episode_type: str  # "goal_info" or "movement"
    content: Dict      # エピソードの内容
    vector: np.ndarray # ベクトル表現
    value: float = 0.0 # エピソードの価値
    
    # グラフ構造（他のエピソードへの接続）
    connected_to: List[int] = field(default_factory=list)  # 接続先エピソードID
    connected_from: List[int] = field(default_factory=list)  # 接続元エピソードID
    
    # メッセージパッシング
    messages: Dict[str, float] = field(default_factory=dict)  # 受信したメッセージ
    
    # 異物なエントロピー（行き止まりへの近さ）
    anomalous_entropy: float = 0.0
    
    def __str__(self):
        if self.episode_type == "goal_info":
            return f"Goal[{self.node_id}]: {self.content['position']}"
        else:
            from_pos = self.content['from']
            to_pos = self.content['to']
            result = self.content['result']
            action_str = ['↑', '→', '↓', '←'][self.content['action']]
            return f"Move[{self.node_id}]: {from_pos}{action_str}{to_pos}({result})"


class GeDIGFrontierNavigator:
    """geDIG方式のエピソードナビゲーター（フロンティア探索版）"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: Dict[int, EpisodeNode] = {}  # ID -> EpisodeNode
        self.episode_counter = 0
        self.current_path_episodes: List[int] = []  # 現在の経路のエピソードID列
        
        # 既知情報: (位置, 行動) -> 結果
        self.known_transitions: Dict[Tuple[Tuple[int, int], int], str] = {}
        
        # 探索済み位置
        self.explored_positions: Set[Tuple[int, int]] = set()
        
        # 各位置から未探索方向の数
        self.unexplored_actions: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        
    def add_goal_info(self, goal_pos: Tuple[int, int]) -> int:
        """ゴール情報エピソードを追加"""
        content = {"position": goal_pos}
        vector = np.array([goal_pos[0], goal_pos[1], 100.0])  # ゴールは特別な値
        
        episode = EpisodeNode(
            node_id=self.episode_counter,
            episode_type="goal_info",
            content=content,
            vector=vector,
            value=100.0
        )
        self.episodes[self.episode_counter] = episode
        self.episode_counter += 1
        print(f"📍 ゴール情報追加: {goal_pos} (Episode {episode.node_id})")
        return episode.node_id
        
    def add_movement_episode(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                           action: int, result: str) -> int:
        """移動エピソードを追加"""
        content = {
            "from": from_pos,
            "to": to_pos,
            "action": action,
            "result": result
        }
        
        # 移動をベクトル化（位置情報 + 結果）
        vector = np.array([
            from_pos[0], from_pos[1],
            to_pos[0], to_pos[1],
            1.0 if result == "成功" else -1.0
        ])
        
        # 価値の初期設定と異物なエントロピー
        if result == "成功":
            value = 1.0
            anomalous_entropy = 0.0
            # 探索済み位置に追加
            self.explored_positions.add(from_pos)
            self.explored_positions.add(to_pos)
        elif result == "壁":
            value = -5.0
            anomalous_entropy = 0.5  # 壁も軽度の異物
        else:  # 行き止まり
            value = -10.0
            anomalous_entropy = 1.0  # 最大の異物なエントロピー
            
        episode = EpisodeNode(
            node_id=self.episode_counter,
            episode_type="movement",
            content=content,
            vector=vector,
            value=value,
            anomalous_entropy=anomalous_entropy
        )
        self.episodes[self.episode_counter] = episode
        self.episode_counter += 1
        
        # 成功した移動の場合、経路上の前のエピソードと接続
        if result == "成功" and self.current_path_episodes:
            prev_episode_id = self.current_path_episodes[-1]
            self.episodes[prev_episode_id].connected_to.append(episode.node_id)
            episode.connected_from.append(prev_episode_id)
            self.current_path_episodes.append(episode.node_id)
        elif result == "成功":
            # 新しい経路の開始
            self.current_path_episodes = [episode.node_id]
        else:
            # 失敗したらパスをリセット
            self.current_path_episodes = []
        
        # 既知情報として記録
        self.known_transitions[(from_pos, action)] = result
        
        # デバッグ出力
        action_str = ['↑', '→', '↓', '←'][action]
        print(f"   {from_pos} {action_str} {to_pos}: {result} (Episode {episode.node_id})")
        
        return episode.node_id
        
    def find_path_to_frontier(self, current_pos: Tuple[int, int]) -> Optional[int]:
        """未探索領域への最短経路を見つける"""
        # 各位置の未探索方向を更新
        for ep_id, episode in self.episodes.items():
            if episode.episode_type == "movement" and episode.content['result'] == "成功":
                from_pos = episode.content['from']
                action = episode.content['action']
                
                # この方向は探索済み
                if from_pos in self.unexplored_actions:
                    self.unexplored_actions[from_pos].discard(action)
                    
        # 現在位置から最も近い未探索位置を見つける（BFS）
        visited = set()
        queue = deque([(current_pos, [])])  # (位置, 経路)
        
        while queue:
            pos, path = queue.popleft()
            
            if pos in visited:
                continue
                
            visited.add(pos)
            
            # この位置に未探索方向があるか
            if pos != current_pos:  # 現在位置以外で
                # この位置から4方向チェック
                for action in range(4):
                    if (pos, action) not in self.known_transitions:
                        # 未探索方向を発見！
                        if path:
                            # 最初の移動方向を返す
                            return path[0]
                        else:
                            return action
                            
            # 隣接位置を探索
            for action in range(4):
                if (pos, action) in self.known_transitions:
                    result = self.known_transitions[(pos, action)]
                    if result == "成功":
                        # 次の位置を計算
                        dx, dy = 0, 0
                        if action == 0: dx = -1  # 上
                        elif action == 1: dy = 1  # 右
                        elif action == 2: dx = 1  # 下
                        elif action == 3: dy = -1  # 左
                        next_pos = (pos[0] + dx, pos[1] + dy)
                        
                        if next_pos not in visited:
                            new_path = path + [action] if pos == current_pos else path
                            queue.append((next_pos, new_path))
                            
        return None  # 未探索領域が見つからない
        
    def calculate_information_gain(self, episode_combination: List[int], current_pos: Tuple[int, int]) -> float:
        """エピソードの組み合わせから情報利得（IG）を計算"""
        # 異物なエントロピーの平均値を計算
        total_anomaly = 0.0
        relevant_episodes = 0
        
        for ep_id in episode_combination:
            episode = self.episodes[ep_id]
            if episode.episode_type == "movement" and episode.content['from'] == current_pos:
                total_anomaly += episode.anomalous_entropy
                relevant_episodes += 1
                
        if relevant_episodes == 0:
            return 0.0
            
        avg_anomaly = total_anomaly / relevant_episodes
        
        # IG = -(異物なエントロピー)
        # 異物なエントロピーが低いほどIGが高い
        return -avg_anomaly
        
    def find_similar_episodes(self, current_pos: Tuple[int, int]) -> List[Tuple[int, float]]:
        """現在位置に関連するエピソードをIGを考慮して検索"""
        episode_scores = []
        
        # 同じ位置からの移動エピソードを重視
        for ep_id, episode in self.episodes.items():
            if episode.episode_type == "movement" and episode.content['from'] == current_pos:
                # スコア = 基本価値 - 異物なエントロピー
                score = episode.value - episode.anomalous_entropy * 10.0
                episode_scores.append((ep_id, score))
                
        # ゴール情報も考慮
        for ep_id, episode in self.episodes.items():
            if episode.episode_type == "goal_info":
                goal_pos = episode.content['position']
                distance = abs(goal_pos[0] - current_pos[0]) + abs(goal_pos[1] - current_pos[1])
                score = 5.0 / (distance + 1)  # ゴールに近いほど高スコア
                episode_scores.append((ep_id, score))
                    
        # スコアでソート
        episode_scores.sort(key=lambda x: x[1], reverse=True)
        return episode_scores
        
    def propagate_anomalous_entropy(self, source_episode_id: int):
        """異物なエントロピーを経路に沿って伝播"""
        source_episode = self.episodes[source_episode_id]
        if source_episode.anomalous_entropy == 0:
            return  # 異物でない場合は伝播しない
            
        print(f"\n🔴 異物なエントロピー伝播: Episode {source_episode_id} (entropy={source_episode.anomalous_entropy:.2f})")
        
        # BFSで接続されたエピソードに異物なエントロピーを伝播
        visited = set()
        queue = [(source_episode_id, 0, source_episode.anomalous_entropy)]
        
        while queue:
            current_id, distance, current_entropy = queue.pop(0)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            current_episode = self.episodes[current_id]
            
            # 異物なエントロピーを更新（最大値を取る）
            if distance > 0:  # ソース自身はスキップ
                old_entropy = current_episode.anomalous_entropy
                current_episode.anomalous_entropy = max(current_episode.anomalous_entropy, current_entropy)
                if current_episode.anomalous_entropy > old_entropy:
                    print(f"   Episode {current_id}: {current_episode} ← 異物エントロピー更新 ({old_entropy:.2f} → {current_episode.anomalous_entropy:.2f})")
            
            # 距離に応じて減衰
            next_entropy = current_entropy * 0.8  # 異物なエントロピーはゆっくり減衰
            
            # 接続元のエピソードに伝播
            for prev_id in current_episode.connected_from:
                if prev_id not in visited and next_entropy > 0.1:
                    queue.append((prev_id, distance + 1, next_entropy))
                    
    def decide_action(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """クエリ処理：類似度検索とメッセージパッシングに基づく行動決定"""
        # クエリ表示
        print(f"\n🤔 クエリ: ゴールに辿り着くために、現在位置{current_pos}で次に移動すべき方向は？")
        
        # 現在位置を探索済みに追加
        self.explored_positions.add(current_pos)
        
        # 既知の行動を除外
        unexplored_actions = []
        for action in possible_actions:
            if (current_pos, action) not in self.known_transitions:
                unexplored_actions.append(action)
        
        if unexplored_actions:
            print(f"   未探索の方向: {[['↑', '→', '↓', '←'][a] for a in unexplored_actions]}")
            # 未探索の方向がある場合は優先
            return np.random.choice(unexplored_actions)
            
        # 全て既知の場合、未探索領域への経路を探す
        print(f"   全ての方向が既知")
        frontier_direction = self.find_path_to_frontier(current_pos)
        
        if frontier_direction is not None:
            print(f"   未探索領域への方向: {['↑', '→', '↓', '←'][frontier_direction]}")
            return frontier_direction
            
        # 未探索領域が見つからない場合、異物なエントロピーを考慮
        print(f"   未探索領域なし - 異物なエントロピーで判断")
        
        # 類似エピソード検索
        similar_episodes = self.find_similar_episodes(current_pos)
        
        # 各行動のスコアを計算
        action_scores = {}
        action_anomaly = defaultdict(list)
        
        for ep_id, score in similar_episodes:
            episode = self.episodes[ep_id]
            if episode.episode_type == "movement" and episode.content['from'] == current_pos:
                action = episode.content['action']
                if action in possible_actions:
                    # 異物なエントロピーを記録
                    action_anomaly[action].append(episode.anomalous_entropy)
                    if action not in action_scores:
                        action_scores[action] = score
                        
        # ゴールの位置を取得
        goal_pos = None
        for episode in self.episodes.values():
            if episode.episode_type == "goal_info":
                goal_pos = episode.content['position']
                break
                
        # 成功した行動をフィルタ
        for action in possible_actions:
            if (current_pos, action) in self.known_transitions:
                result = self.known_transitions[(current_pos, action)]
                if result == "成功":
                    base_score = 1.0
                    
                    # ゴール方向にボーナス
                    if goal_pos:
                        dx = goal_pos[0] - current_pos[0]
                        dy = goal_pos[1] - current_pos[1]
                        
                        if (action == 0 and dx < 0) or (action == 2 and dx > 0):  # 上下
                            base_score += abs(dx)
                        elif (action == 1 and dy > 0) or (action == 3 and dy < 0):  # 左右
                            base_score += abs(dy)
                            
                    action_scores[action] = base_score
                elif result == "壁":
                    action_scores[action] = -5.0
                else:  # 行き止まり
                    action_scores[action] = -10.0
                    
        # 各行動の最終スコアを計算
        action_final_scores = {}
        for action in possible_actions:
            if action in action_anomaly and action_anomaly[action]:
                avg_anomaly = sum(action_anomaly[action]) / len(action_anomaly[action])
                # スコア - 異物なエントロピーの影響
                base_score = action_scores.get(action, 0)
                action_final_scores[action] = base_score - avg_anomaly * 10.0
                print(f"     {['↑', '→', '↓', '←'][action]}: 異物エントロピー={avg_anomaly:.2f}, スコア={action_final_scores[action]:.2f}")
            else:
                action_final_scores[action] = action_scores.get(action, 0)
                
        if action_final_scores:
            best_action = max(action_final_scores.items(), key=lambda x: x[1])[0]
            action_str = ['↑', '→', '↓', '←'][best_action]
            print(f"   決定: {action_str} (最終スコア: {action_final_scores[best_action]:.2f})")
            return best_action
        else:
            # ランダムに選択
            return np.random.choice(possible_actions)


def visualize_episode_graph(navigator: 'GeDIGFrontierNavigator', save_path: str = 'gedig_frontier_graph.png'):
    """エピソードグラフを可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左側：移動の軌跡と探索状況
    ax1.set_title("Movement Trajectories & Exploration Status", fontsize=14)
    ax1.set_aspect('equal')
    
    # 探索済み位置を背景に表示
    for pos in navigator.explored_positions:
        rect = plt.Rectangle((pos[1]-0.4, pos[0]-0.4), 0.8, 0.8, 
                            facecolor='lightgreen', alpha=0.3)
        ax1.add_patch(rect)
    
    # 移動エピソードを矢印で表示
    for ep_id, episode in navigator.episodes.items():
        if episode.episode_type == "movement":
            from_pos = episode.content['from']
            to_pos = episode.content['to']
            
            # 色分け（異物なエントロピーで色分け）
            if episode.anomalous_entropy > 0.8:
                color = 'red'  # 高い異物エントロピー
                alpha = 0.8
            elif episode.anomalous_entropy > 0.3:
                color = 'orange'  # 中程度の異物エントロピー
                alpha = 0.6
            elif episode.anomalous_entropy > 0:
                color = 'yellow'  # 低い異物エントロピー
                alpha = 0.5
            else:
                color = 'blue'  # 正常
                alpha = 0.7
                
            # 矢印を描画
            if from_pos != to_pos:  # 移動成功
                ax1.annotate('', xy=(to_pos[1], to_pos[0]), 
                            xytext=(from_pos[1], from_pos[0]),
                            arrowprops=dict(arrowstyle='->', color=color, 
                                          alpha=alpha, lw=2))
            else:  # 壁衝突
                ax1.plot(from_pos[1], from_pos[0], 'x', color=color, 
                        markersize=8, alpha=alpha)
            
    # ゴール位置
    goal_episodes = [e for e in navigator.episodes.values() if e.episode_type == "goal_info"]
    if goal_episodes:
        goal_pos = goal_episodes[0].content['position']
        ax1.plot(goal_pos[1], goal_pos[0], 'go', markersize=15, label='Goal')
        
    ax1.set_xlabel("Y")
    ax1.set_ylabel("X")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Y軸を反転（上が小さい値）
    
    # 右側：エピソードグラフ
    ax2.set_title("Episode Graph (Episodes as Nodes)", fontsize=14)
    
    # NetworkXグラフを構築
    G = nx.DiGraph()
    
    # ノードを追加
    for ep_id, episode in navigator.episodes.items():
        G.add_node(ep_id, episode=episode)
        
    # エッジを追加
    for ep_id, episode in navigator.episodes.items():
        for connected_id in episode.connected_to:
            G.add_edge(ep_id, connected_id)
            
    # レイアウト
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # ノードの色
        node_colors = []
        for node_id in G.nodes():
            episode = navigator.episodes[node_id]
            if episode.episode_type == "goal_info":
                node_colors.append('yellow')
            else:
                if episode.anomalous_entropy > 0.8:
                    node_colors.append('red')  # 高異物エントロピー
                elif episode.anomalous_entropy > 0.3:
                    node_colors.append('orange')  # 中異物エントロピー
                elif episode.anomalous_entropy > 0:
                    node_colors.append('yellow')  # 低異物エントロピー
                else:
                    node_colors.append('lightblue')  # 正常
                
        # 描画
        nx.draw(G, pos, node_color=node_colors, node_size=800,
                with_labels=True, ax=ax2, arrows=True,
                edge_color='gray', alpha=0.7, font_size=10)
                
        # ラベル
        labels = {}
        for node_id in G.nodes():
            episode = navigator.episodes[node_id]
            if episode.episode_type == "goal_info":
                labels[node_id] = f"G\n{episode.content['position']}"
            else:
                action_str = ['↑', '→', '↓', '←'][episode.content['action']]
                result_str = episode.content['result'][:1]
                labels[node_id] = f"{node_id}\n{action_str}{result_str}"
                
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax2)
        
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ {save_path} として保存しました")
    return fig


def run_experiment():
    """実験を実行"""
    print("geDIG フロンティア探索ナビゲーター実験")
    print("=" * 60)
    print("未探索領域を優先的に探索し、異物なエントロピーで行き止まりを回避")
    print("-" * 60)
    
    config = MazeNavigatorConfig()
    navigator = GeDIGFrontierNavigator(config)
    
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
    max_steps = 200
    
    while steps < max_steps:
        current_pos = obs.position
        
        # 行動決定（クエリ処理）
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
        episode_id = navigator.add_movement_episode(old_pos, new_pos, action, result)
        
        # 異物なエントロピーの伝播
        if result == "行き止まり" or result == "壁":
            navigator.propagate_anomalous_entropy(episode_id)
            
        # ゴール到達
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            break
            
    else:
        print(f"\n❌ タイムアウト（{max_steps}ステップ）")
        
    # 統計表示
    print("\n統計情報:")
    print(f"  総エピソード数: {len(navigator.episodes)}")
    movement_episodes = [e for e in navigator.episodes.values() if e.episode_type == "movement"]
    print(f"  移動エピソード数: {len(movement_episodes)}")
    
    # 結果別集計
    results = {"成功": 0, "壁": 0, "行き止まり": 0}
    for e in movement_episodes:
        results[e.content['result']] += 1
    print(f"  結果内訳: {results}")
    
    # 異物なエントロピーを持つエピソード数
    anomalous_episodes = [e for e in navigator.episodes.values() if e.anomalous_entropy > 0]
    print(f"  異物エントロピー付きエピソード: {len(anomalous_episodes)}")
    
    # 異物エントロピーの分布
    if anomalous_episodes:
        avg_anomaly = sum(e.anomalous_entropy for e in anomalous_episodes) / len(anomalous_episodes)
        print(f"  平均異物エントロピー: {avg_anomaly:.3f}")
        
    print(f"  探索済み位置数: {len(navigator.explored_positions)}")
    
    # グラフ可視化
    visualize_episode_graph(navigator)


if __name__ == "__main__":
    run_experiment()