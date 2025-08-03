#!/usr/bin/env python3
"""geDIG Episode Navigator: エピソードをノードとして扱うメッセージパッシング実装"""

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
    
    # 訪問回数（頻繁な訪問も異物として扱う）
    visit_count: int = 0
    
    def __str__(self):
        if self.episode_type == "goal_info":
            return f"Goal[{self.node_id}]: {self.content['position']}"
        else:
            from_pos = self.content['from']
            to_pos = self.content['to']
            result = self.content['result']
            action_str = ['↑', '→', '↓', '←'][self.content['action']]
            return f"Move[{self.node_id}]: {from_pos}{action_str}{to_pos}({result})"


class GeDIGEpisodeNavigator:
    """geDIG方式のエピソードナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: Dict[int, EpisodeNode] = {}  # ID -> EpisodeNode
        self.episode_counter = 0
        self.current_path_episodes: List[int] = []  # 現在の経路のエピソードID列
        # 既知情報: (位置, 行動) -> 結果
        self.known_transitions: Dict[Tuple[Tuple[int, int], int], str] = {}
        # 最近訪問した位置（循環回避用）
        self.recent_positions: deque = deque(maxlen=5)
        
    def add_goal_info(self, goal_pos: Tuple[int, int]) -> int:
        """ゴール情報エピソードを追加"""
        content = {
            "position": goal_pos,
            "description": "ゴール地点",
            "reward": 100.0
        }
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
                           action: int, result: str, found_goal: bool = False) -> int:
        """移動エピソードを追加"""
        content = {
            "from": from_pos,
            "to": to_pos,
            "action": action,
            "result": result,
            "found_goal": found_goal  # ゴールに到達したかどうか
        }
        
        # 移動をベクトル化（位置情報 + 結果）
        vector = np.array([
            from_pos[0], from_pos[1],
            to_pos[0], to_pos[1],
            1.0 if result == "成功" else -1.0
        ])
        
        # 同じ移動の既存エピソードを探して訪問回数を更新
        for ep_id, episode in self.episodes.items():
            if (episode.episode_type == "movement" and 
                episode.content['from'] == from_pos and 
                episode.content['action'] == action):
                episode.visit_count += 1
                # 訪問回数が多いほど異物なエントロピーを増加
                if episode.visit_count > 2:  # 3回以上訪問したら異物として扱い始める
                    episode.anomalous_entropy = min(1.0, 0.2 * (episode.visit_count - 2))
                    print(f"   🔄 繰り返し訪問検出: {from_pos} → {to_pos} (訪問回数: {episode.visit_count}, 異物エントロピー: {episode.anomalous_entropy:.2f})")
        
        # 価値の初期設定と異物なエントロピー
        if result == "成功":
            if found_goal:
                value = 100.0  # ゴール到達は最高価値
                anomalous_entropy = 0.0
            else:
                value = 1.0
                anomalous_entropy = 0.0
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
            anomalous_entropy=anomalous_entropy,
            visit_count=1  # 初回訪問
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
        
    def calculate_graph_edit_distance(self, episode1: EpisodeNode, episode2: EpisodeNode) -> float:
        """2つのエピソード間のグラフ編集距離を計算"""
        # ノードの接続数の差
        out_diff = abs(len(episode1.connected_to) - len(episode2.connected_to))
        in_diff = abs(len(episode1.connected_from) - len(episode2.connected_from))
        
        # 共通接続の計算
        common_out = len(set(episode1.connected_to) & set(episode2.connected_to))
        common_in = len(set(episode1.connected_from) & set(episode2.connected_from))
        
        # GED = 接続の差 - 共通接続（共通が多いほどGEDが小さい）
        ged = (out_diff + in_diff) - (common_out + common_in) * 0.5
        return max(0, ged)
    
    def calculate_connection_entropy(self, episode: EpisodeNode) -> float:
        """エピソードの接続パターンのエントロピーを計算（偏差ベース）"""
        # 接続先エピソードの訪問回数分布
        visit_counts = []
        for connected_id in episode.connected_to:
            if connected_id in self.episodes:
                visit_counts.append(self.episodes[connected_id].visit_count)
        
        if not visit_counts:
            return 0.0
            
        # 訪問回数の偏差を計算
        mean_visits = np.mean(visit_counts)
        variance = np.var(visit_counts)
        
        # 偏差が大きい = エントロピーが低い = 確実性が高い
        # 正規化: variance / (mean^2 + 1) で0-1の範囲に
        if mean_visits > 0:
            normalized_variance = variance / (mean_visits ** 2 + 1)
            entropy = 1.0 - min(1.0, normalized_variance)  # 偏差の逆関数
        else:
            entropy = 1.0
            
        return entropy
    
    def calculate_gedig(self, query_episode: EpisodeNode, candidate_episode: EpisodeNode) -> float:
        """geDIG = GED - IG を計算"""
        # Graph Edit Distance
        ged = self.calculate_graph_edit_distance(query_episode, candidate_episode)
        
        # Information Gain (エントロピーベース)
        # 高エントロピー（均等・不確実） = 高IG（新しい情報）
        # 低エントロピー（偏り・確実） = 低IG（既知の情報）
        entropy = self.calculate_connection_entropy(candidate_episode)
        ig = entropy  # エントロピーがそのままIG
        
        # 訪問回数による追加ペナルティ（IGを減少させる）
        if candidate_episode.visit_count > 2:
            visit_penalty = (candidate_episode.visit_count - 2) * 0.5
            ig = max(0, ig - visit_penalty)  # IGが負にならないように
        
        # geDIG = GED - IG
        # 小さいほど良い（似ているが新しい情報を持つ）
        return ged - ig
        
    def find_topk_episodes_by_gedig(self, current_pos: Tuple[int, int], k: int = 7) -> List[Tuple[int, float]]:
        """geDIGが最小となるトップKエピソードを検索"""
        # クエリエピソードを作成（現在位置の仮想エピソード）
        # 現在位置から/への既存接続を収集
        query_connected_to = []
        query_connected_from = []
        
        for ep_id, episode in self.episodes.items():
            if episode.episode_type == "movement":
                if episode.content['from'] == current_pos and episode.content['result'] == "成功":
                    query_connected_to.append(ep_id)
                elif episode.content['to'] == current_pos and episode.content['result'] == "成功":
                    query_connected_from.append(ep_id)
        
        query_episode = EpisodeNode(
            node_id=-1,
            episode_type="query",
            content={"position": current_pos, "seeking_goal": True},
            vector=np.array([current_pos[0], current_pos[1], 0]),
            connected_to=query_connected_to,
            connected_from=query_connected_from
        )
        
        # 現在位置から移動可能なエピソードを収集
        gedig_scores = []
        
        for ep_id, episode in self.episodes.items():
            # 移動エピソード
            if episode.episode_type == "movement":
                # 現在位置からの移動
                if episode.content['from'] == current_pos:
                    gedig = self.calculate_gedig(query_episode, episode)
                    # ゴール発見エピソードは価値が高いので優先
                    if episode.content.get('found_goal', False):
                        gedig -= 2.0  # ゴール発見ボーナス
                    gedig_scores.append((ep_id, gedig))
                # 現在位置への移動（逆方向も考慮）
                elif episode.content['to'] == current_pos:
                    gedig = self.calculate_gedig(query_episode, episode) * 1.2  # 逆方向はペナルティ
                    gedig_scores.append((ep_id, gedig))
            
            # ゴール関連エピソードも類似度で評価
            elif episode.episode_type == "goal_info":
                # 位置的な近さを評価
                goal_pos = episode.content['position']
                dist = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
                # 近いほど関連性が高い（geDIGが小さい）
                gedig = dist * 0.5  # 距離に応じたスコア
                gedig_scores.append((ep_id, gedig))
        
        # geDIGでソート（昇順：小さいほど良い）
        gedig_scores.sort(key=lambda x: x[1])
        
        # デバッグ出力
        print(f"\n📊 Top-{k} エピソード (geDIG昇順):")
        for i, (ep_id, gedig) in enumerate(gedig_scores[:k]):
            episode = self.episodes[ep_id]
            print(f"   {i+1}. Episode {ep_id}: geDIG={gedig:.2f}, {episode}")
        
        return gedig_scores[:k]
        
    def propagate_anomalous_entropy(self, source_episode_id: int):
        """異物なエントロピーを経路に沿って伝播"""
        source_episode = self.episodes[source_episode_id]
        if source_episode.anomalous_entropy == 0:
            return  # 異物でない場合は伝播しない
            
        # 繰り返し訪問の場合と行き止まりの場合で異なるメッセージ
        if source_episode.visit_count > 2:
            print(f"\n🔄 繰り返し訪問による異物エントロピー伝播: Episode {source_episode_id} (訪問回数={source_episode.visit_count}, entropy={source_episode.anomalous_entropy:.2f})")
        else:
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
        """クエリ処理：geDIG最小のトップKエピソードとメッセージパッシングに基づく行動決定"""
        # クエリ表示
        print(f"\n🤔 クエリ: ゴールに辿り着くために、現在位置{current_pos}で次に移動すべき方向は？")
        
        # 現在位置を記録
        self.recent_positions.append(current_pos)
        
        # ループ検出: 同じ位置が3回以上出現したら強制的にランダム選択
        position_count = self.recent_positions.count(current_pos)
        if position_count >= 3:
            print(f"   🔄 ループ検出! {current_pos}が{position_count}回出現 - ランダム探索に切り替え")
            valid_actions = []
            for action in possible_actions:
                if (current_pos, action) in self.known_transitions:
                    if self.known_transitions[(current_pos, action)] != "壁":
                        valid_actions.append(action)
                else:
                    valid_actions.append(action)  # 未探索は含める
            
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.choice(possible_actions)
        
        # 既知の行動を除外
        unexplored_actions = []
        for action in possible_actions:
            if (current_pos, action) not in self.known_transitions:
                unexplored_actions.append(action)
        
        if unexplored_actions:
            print(f"   未探索の方向: {[['↑', '→', '↓', '←'][a] for a in unexplored_actions]}")
        
        # geDIGベースでトップKエピソードを検索
        topk_episodes = self.find_topk_episodes_by_gedig(current_pos, k=7)
        
        # メッセージパッシング：トップKエピソードから各行動のスコアを集計
        action_scores = defaultdict(float)
        action_episodes = defaultdict(list)
        
        print("\n💬 メッセージパッシング:")
        for ep_id, gedig in topk_episodes:
            episode = self.episodes[ep_id]
            
            if episode.episode_type == "movement" and episode.content['from'] == current_pos:
                action = episode.content['action']
                if action in possible_actions:
                    # メッセージ = 1 / (|geDIG| + 1) で正規化（負の値を避ける）
                    message = 1.0 / (abs(gedig) + 1.0)
                    
                    # 成功/失敗で重み付け
                    if episode.content['result'] == "成功":
                        # 訪問回数が多いほどペナルティ
                        if episode.visit_count > 5:
                            message *= -2.0  # 6回以上は強い負のメッセージ
                        elif episode.visit_count > 3:
                            message *= 0.1  # 4-5回は大幅減
                        else:
                            message *= 1.0
                    elif episode.content['result'] == "壁":
                        message *= -0.5
                    else:  # 行き止まり
                        message *= -1.0
                    
                    action_scores[action] += message
                    action_episodes[action].append((ep_id, message))
                    print(f"   Action {['↑', '→', '↓', '←'][action]}: Episode {ep_id} → message={message:.3f}")
            
                    
        # ゴール関連エピソードがtopKに入っていればメッセージとして活用
        for ep_id, gedig in topk_episodes:
            episode = self.episodes[ep_id]
            if episode.episode_type == "goal_info":
                # ゴールエピソードからのメッセージ：「この位置を目指せ」
                goal_pos = episode.content['position']
                print(f"   💎 ゴールエピソード検出: {goal_pos} (geDIG={gedig:.2f})")
                
                # ゴールに近づく方向に正のメッセージ
                dx = goal_pos[0] - current_pos[0]
                dy = goal_pos[1] - current_pos[1]
                
                message_strength = 1.0 / (gedig + 1.0)
                
                if dx > 0 and 2 in possible_actions:  # 下
                    action_scores[2] = action_scores.get(2, 0) + message_strength
                elif dx < 0 and 0 in possible_actions:  # 上
                    action_scores[0] = action_scores.get(0, 0) + message_strength
                    
                if dy > 0 and 1 in possible_actions:  # 右
                    action_scores[1] = action_scores.get(1, 0) + message_strength
                elif dy < 0 and 3 in possible_actions:  # 左
                    action_scores[3] = action_scores.get(3, 0) + message_strength
        
        # 未探索の行動を優先
        if unexplored_actions:
            print("   未探索行動にボーナス付与")
            for action in unexplored_actions:
                action_scores[action] = action_scores.get(action, 0) + 10.0  # 探索ボーナスを強化
                
        # 全ての可能な行動に最小スコアを確保（デッドロック回避）
        for action in possible_actions:
            if action not in action_scores:
                action_scores[action] = 0.1  # 最小スコア
        
        # 行動決定：スコアが最も高い行動を選択
        if action_scores:
            # 最終スコアの集計
            print("\n📊 最終スコア:")
            for action in possible_actions:
                if action in action_scores:
                    print(f"   {['↑', '→', '↓', '←'][action]}: {action_scores[action]:.3f}")
            
            # 最高スコアが負の場合、ランダム探索に切り替え
            best_score = max(action_scores.values())
            if best_score < 0:
                print("\n⚠️ 全ての方向が負のスコア - ランダム探索に切り替え")
                # ただし、壁の方向は除外
                valid_actions = []
                for action in possible_actions:
                    if (current_pos, action) in self.known_transitions:
                        if self.known_transitions[(current_pos, action)] != "壁":
                            valid_actions.append(action)
                    else:
                        valid_actions.append(action)  # 未探索は含める
                
                if valid_actions:
                    return np.random.choice(valid_actions)
                else:
                    return np.random.choice(possible_actions)
            
            # 最高スコアの行動を選択
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            action_str = ['↑', '→', '↓', '←'][best_action]
            print(f"\n✅ 決定: {action_str}")
            
            # 参考エピソードを表示
            if best_action in action_episodes and action_episodes[best_action]:
                print("   参考エピソード:")
                for ep_id, msg in action_episodes[best_action][:3]:  # 上位3つまで
                    episode = self.episodes[ep_id]
                    print(f"     Episode {ep_id}: {episode}")
                    
            return best_action
        else:
            # スコアがない場合はランダム
            print("   ⚠️ 有効なスコアなし - ランダム選択")
            return np.random.choice(possible_actions)


def visualize_episode_graph(navigator: 'GeDIGEpisodeNavigator', maze: 'SimpleMaze' = None, 
                          save_path: str = 'gedig_episode_graph.png'):
    """エピソードグラフを可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左側：迷路と探索経路
    ax1.set_title("Maze Exploration Path", fontsize=14)
    ax1.set_aspect('equal')
    
    # グリッドベースの迷路表示
    if maze:
        # 背景グリッド
        for i in range(maze.size[0] + 1):
            ax1.axhline(y=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(maze.size[1] + 1):
            ax1.axvline(x=j - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        # 外枠
        ax1.axhline(y=-0.5, color='black', linewidth=2)
        ax1.axhline(y=maze.size[0]-0.5, color='black', linewidth=2)
        ax1.axvline(x=-0.5, color='black', linewidth=2)
        ax1.axvline(x=maze.size[1]-0.5, color='black', linewidth=2)
    
    # 訪問回数を記録するヒートマップ用データ
    visit_counts = defaultdict(int)
    
    # 探索経路を時系列順に描画
    movement_episodes = [(ep_id, ep) for ep_id, ep in navigator.episodes.items() 
                        if ep.episode_type == "movement" and ep.content['result'] == "成功"]
    
    # 経路を順番に描画（薄い色から濃い色へ）
    for idx, (ep_id, episode) in enumerate(movement_episodes):
        from_pos = episode.content['from']
        to_pos = episode.content['to']
        
        # 訪問回数を記録
        visit_counts[from_pos] += 1
        visit_counts[to_pos] += 1
        
        # 時系列による色の変化（古い→新しい：青→赤）
        time_ratio = idx / max(len(movement_episodes) - 1, 1)
        
        # 色分け
        if episode.content.get('found_goal', False):
            color = 'gold'  # ゴール発見経路
            alpha = 1.0
            linewidth = 4
        else:
            # 青から赤へのグラデーション
            color = plt.cm.coolwarm(time_ratio)
            alpha = 0.3 + 0.4 * time_ratio  # 新しいほど濃く
            linewidth = 1 + episode.visit_count * 0.3
        
        # 矢印を描画（座標系を修正）
        dx = to_pos[1] - from_pos[1]
        dy = to_pos[0] - from_pos[0]
        ax1.arrow(from_pos[1], from_pos[0], dx * 0.8, dy * 0.8,
                 head_width=0.15, head_length=0.1, fc=color, ec=color,
                 alpha=alpha, linewidth=linewidth)
    
    # 訪問回数のヒートマップ
    for pos, count in visit_counts.items():
        if count > 2:  # 3回以上訪問した場所
            size = min(100 + count * 20, 300)
            ax1.scatter(pos[1], pos[0], s=size, c='red', alpha=0.3, marker='o')
    
    # スタートとゴール
    if maze:
        ax1.plot(maze.start_pos[1], maze.start_pos[0], 'bs', markersize=15, 
                label='Start', zorder=10)
    
    goal_episodes = [e for e in navigator.episodes.values() if e.episode_type == "goal_info"]
    if goal_episodes:
        goal_pos = goal_episodes[0].content['position']
        ax1.plot(goal_pos[1], goal_pos[0], 'g*', markersize=25, 
                label='Goal', zorder=10)
    
    # 壁の位置を検出して表示（失敗した移動から推測）
    wall_positions = set()
    for ep in navigator.episodes.values():
        if ep.episode_type == "movement" and ep.content['result'] == "壁":
            from_pos = ep.content['from']
            action = ep.content['action']
            # 壁の位置を推定
            if action == 0:  # 上
                wall_pos = (from_pos[0] - 0.5, from_pos[1])
            elif action == 1:  # 右
                wall_pos = (from_pos[0], from_pos[1] + 0.5)
            elif action == 2:  # 下
                wall_pos = (from_pos[0] + 0.5, from_pos[1])
            else:  # 左
                wall_pos = (from_pos[0], from_pos[1] - 0.5)
            wall_positions.add((wall_pos, action))
    
    # 壁を描画
    for (wall_pos, direction) in wall_positions:
        if direction in [0, 2]:  # 横壁
            ax1.plot([wall_pos[1] - 0.4, wall_pos[1] + 0.4], 
                    [wall_pos[0], wall_pos[0]], 'k-', linewidth=3)
        else:  # 縦壁
            ax1.plot([wall_pos[1], wall_pos[1]], 
                    [wall_pos[0] - 0.4, wall_pos[0] + 0.4], 'k-', linewidth=3)
    
    ax1.set_xlim(-0.5, maze.size[1] - 0.5 if maze else 9.5)
    ax1.set_ylim(maze.size[0] - 0.5 if maze else 9.5, -0.5)
    ax1.set_xlabel("Y")
    ax1.set_ylabel("X")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
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
    print("geDIG エピソードナビゲーター実験")
    print("=" * 60)
    print("エピソードをノードとして扱い、メッセージパッシングで失敗情報を伝播")
    print("-" * 60)
    
    config = MazeNavigatorConfig()
    navigator = GeDIGEpisodeNavigator(config)
    
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
    max_steps = 500  # タイムアウトなしで探索を続ける
    
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
            
        # ゴール到達チェック
        found_goal = (new_pos == maze.goal_pos and result == "成功")
        
        # エピソード形成
        episode_id = navigator.add_movement_episode(old_pos, new_pos, action, result, found_goal)
        
        # 異物なエントロピーの伝播
        if result == "行き止まり" or result == "壁":
            navigator.propagate_anomalous_entropy(episode_id)
            
        # 繰り返し訪問による異物エントロピーの伝播もチェック
        for ep_id, episode in navigator.episodes.items():
            if (episode.episode_type == "movement" and 
                episode.content['from'] == old_pos and 
                episode.content['action'] == action and
                episode.visit_count > 2 and
                episode.anomalous_entropy > 0):
                navigator.propagate_anomalous_entropy(ep_id)
            
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
    
    # グラフ可視化
    visualize_episode_graph(navigator, maze)


if __name__ == "__main__":
    run_experiment()