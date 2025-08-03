#!/usr/bin/env python3
"""統合型geDIGナビゲーター：エピソード記憶＋メッセージパッシング"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class Episode:
    """エピソード：クエリ・コンテキスト・行動・結果・価値"""
    query: str  # "位置(x,y)でゴール(gx,gy)に向かうには？"
    context: Dict[str, Any]  # 状況情報
    action: int  # 選択した行動
    result: Dict[str, Any]  # 結果情報
    value: float  # エピソードの価値
    timestamp: int  # 時刻


@dataclass
class PathMessage:
    """経路メッセージ：分岐点からの経路情報"""
    from_junction: Tuple[int, int]
    to_destination: Tuple[int, int]
    via_action: int
    is_dead_end: bool
    is_goal: bool
    path_length: int
    value: float


@dataclass
class PositionNode:
    """位置ノード：エピソード記憶とメッセージを持つ"""
    position: Tuple[int, int]
    episodes: List[Episode] = field(default_factory=list)
    incoming_messages: List[PathMessage] = field(default_factory=list)
    outgoing_messages: List[PathMessage] = field(default_factory=list)
    is_junction: bool = False
    is_dead_end: bool = False
    is_goal: bool = False
    visit_count: int = 0
    possible_actions: Set[int] = field(default_factory=set)
    
    def get_action_value(self, action: int) -> float:
        """特定の行動の総合価値を計算"""
        # エピソード記憶からの価値
        episode_values = [ep.value for ep in self.episodes if ep.action == action]
        episode_value = np.mean(episode_values) if episode_values else 0.0
        
        # メッセージからの価値
        message_values = [msg.value for msg in self.outgoing_messages if msg.via_action == action]
        message_value = np.mean(message_values) if message_values else 0.0
        
        # 総合価値（エピソード記憶とメッセージの融合）
        return episode_value + message_value * 0.5


class IntegratedGeDIGNavigator:
    """統合型geDIGナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.nodes: Dict[Tuple[int, int], PositionNode] = {}
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.time_step = 0
        
        # 経路追跡
        self.current_path: List[Tuple[int, int]] = []
        self.path_start_pos: Optional[Tuple[int, int]] = None
        self.path_start_action: Optional[int] = None
        
    def _get_or_create_node(self, pos: Tuple[int, int]) -> PositionNode:
        """ノードを取得または作成"""
        if pos not in self.nodes:
            self.nodes[pos] = PositionNode(position=pos)
        return self.nodes[pos]
        
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """マンハッタン距離"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def _structural_similarity(self, context1: Dict, context2: Dict) -> float:
        """構造的類似度の計算"""
        # 位置の近さ
        if 'position' in context1 and 'position' in context2:
            pos_dist = self._manhattan_distance(context1['position'], context2['position'])
            pos_sim = 1.0 / (1.0 + pos_dist)
        else:
            pos_sim = 0.0
            
        # 可能な行動の類似度
        if 'possible_actions' in context1 and 'possible_actions' in context2:
            actions1 = set(context1['possible_actions'])
            actions2 = set(context2['possible_actions'])
            if actions1 and actions2:
                action_sim = len(actions1 & actions2) / len(actions1 | actions2)
            else:
                action_sim = 0.0
        else:
            action_sim = 0.0
            
        # ゴールまでの距離の類似度
        if 'goal_distance' in context1 and 'goal_distance' in context2:
            dist_diff = abs(context1['goal_distance'] - context2['goal_distance'])
            dist_sim = 1.0 / (1.0 + dist_diff)
        else:
            dist_sim = 0.0
            
        # 総合類似度
        return (pos_sim + action_sim + dist_sim) / 3.0
        
    def create_episode(self, obs, action: int, new_obs, maze) -> Episode:
        """エピソードを作成"""
        # クエリ生成
        if self.goal_pos:
            query = f"位置{obs.position}でゴール{self.goal_pos}に向かうには？"
        else:
            query = f"位置{obs.position}から探索を進めるには？"
            
        # コンテキスト
        context = {
            'position': obs.position,
            'possible_actions': obs.possible_moves,
            'is_junction': obs.is_junction,
            'goal_known': self.goal_pos is not None,
            'goal_distance': self._manhattan_distance(obs.position, self.goal_pos) if self.goal_pos else 0,
            'time': self.time_step
        }
        
        # 結果
        success = obs.position != new_obs.position
        goal_progress = 0.0
        if self.goal_pos and success:
            dist_before = self._manhattan_distance(obs.position, self.goal_pos)
            dist_after = self._manhattan_distance(new_obs.position, self.goal_pos)
            goal_progress = dist_before - dist_after
            
        result = {
            'new_position': new_obs.position,
            'success': success,
            'goal_progress': goal_progress,
            'found_goal': new_obs.is_goal,
            'hit_dead_end': new_obs.is_dead_end
        }
        
        # 価値計算
        if new_obs.is_goal:
            value = 10.0
        elif new_obs.is_dead_end:
            value = -5.0
        elif success:
            value = 1.0 + goal_progress
        else:
            value = -1.0
            
        return Episode(
            query=query,
            context=context,
            action=action,
            result=result,
            value=value,
            timestamp=self.time_step
        )
        
    def _propagate_path_result(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                              action: int, path_length: int, is_dead_end: bool, is_goal: bool):
        """経路結果をメッセージとして伝播"""
        # メッセージ作成
        if is_goal:
            value = 10.0 / (path_length + 1)
        elif is_dead_end:
            value = -10.0
        else:
            value = 0.0
            
        message = PathMessage(
            from_junction=from_pos,
            to_destination=to_pos,
            via_action=action,
            is_dead_end=is_dead_end,
            is_goal=is_goal,
            path_length=path_length,
            value=value
        )
        
        # 開始ノードに発信メッセージ追加
        from_node = self._get_or_create_node(from_pos)
        from_node.outgoing_messages.append(message)
        
        # 終点ノードに受信メッセージ追加
        to_node = self._get_or_create_node(to_pos)
        to_node.incoming_messages.append(message)
        
        if is_dead_end:
            print(f"💀 メッセージ: {from_pos} --{['上','右','下','左'][action]}--> {to_pos} (行き止まり)")
        elif is_goal:
            print(f"🎯 メッセージ: {from_pos} --{['上','右','下','左'][action]}--> {to_pos} (ゴール!)")
            
    def query_action(self, obs, maze) -> int:
        """現在の状況に最適な行動をクエリ"""
        current_node = self._get_or_create_node(obs.position)
        
        # 現在のクエリ
        if self.goal_pos:
            current_query = f"位置{obs.position}でゴール{self.goal_pos}に向かうには？"
        else:
            current_query = f"位置{obs.position}から探索を進めるには？"
            
        # 関連エピソードを検索
        all_episodes = []
        for node in self.nodes.values():
            all_episodes.extend(node.episodes)
            
        # 類似度でソート
        current_context = {
            'position': obs.position,
            'possible_actions': obs.possible_moves,
            'goal_distance': self._manhattan_distance(obs.position, self.goal_pos) if self.goal_pos else 0
        }
        
        similar_episodes = []
        for ep in all_episodes:
            similarity = self._structural_similarity(current_context, ep.context)
            if similarity > 0.3:  # 閾値
                similar_episodes.append((ep, similarity))
                
        similar_episodes.sort(key=lambda x: x[1], reverse=True)
        
        # 各行動のgeDIG評価
        action_scores = {}
        
        for action in obs.possible_moves:
            # エピソード記憶からの評価
            episode_scores = []
            for ep, sim in similar_episodes[:10]:  # 上位10件
                if ep.action == action:
                    episode_scores.append(ep.value * sim)
                    
            # メッセージからの評価
            message_value = current_node.get_action_value(action)
            
            # 情報利得
            action_count = sum(1 for ep in current_node.episodes if ep.action == action)
            ig = 1.0 / (action_count + 1)
            
            # geDIG目的関数
            if episode_scores:
                ged_value = np.mean(episode_scores) + message_value
            else:
                ged_value = message_value
                
            score = self.config.w_ged * ged_value - self.config.k_ig * ig
            action_scores[action] = score
            
        # 最適行動を選択
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            
            # デバッグ情報
            if current_node.visit_count <= 2 or obs.is_junction:
                print(f"\n位置{obs.position}での意思決定:")
                for a in obs.possible_moves:
                    print(f"  {['上','右','下','左'][a]}: {action_scores[a]:.2f}")
                print(f"  → 選択: {['上','右','下','左'][best_action]}")
                
            return best_action
        else:
            return np.random.choice(obs.possible_moves)
            
    def decide_action(self, obs, maze) -> int:
        """観測から行動を決定"""
        current_node = self._get_or_create_node(obs.position)
        current_node.visit_count += 1
        current_node.possible_actions.update(obs.possible_moves)
        
        # ノード属性更新
        current_node.is_junction = obs.is_junction
        current_node.is_dead_end = obs.is_dead_end
        current_node.is_goal = obs.is_goal
        
        # ゴール発見
        if obs.is_goal and not self.goal_pos:
            self.goal_pos = obs.position
            print(f"🎯 ゴール発見！位置: {self.goal_pos}")
            
        # 経路管理
        if obs.is_junction or current_node.visit_count == 1:
            # 前の経路を終了
            if self.path_start_pos and self.path_start_action is not None:
                path_length = len(self.current_path)
                if obs.is_dead_end:
                    self._propagate_path_result(
                        self.path_start_pos, obs.position, self.path_start_action,
                        path_length, is_dead_end=True, is_goal=False
                    )
                elif obs.is_goal:
                    self._propagate_path_result(
                        self.path_start_pos, obs.position, self.path_start_action,
                        path_length, is_dead_end=False, is_goal=True
                    )
                    
            # 新しい経路を開始
            self.path_start_pos = obs.position
            self.current_path = [obs.position]
        else:
            self.current_path.append(obs.position)
            
        # 行動をクエリ
        action = self.query_action(obs, maze)
        
        if obs.is_junction or current_node.visit_count == 1:
            self.path_start_action = action
            
        return action
        
    def update_after_action(self, old_obs, action: int, new_obs, maze):
        """行動後の更新"""
        # エピソード作成
        episode = self.create_episode(old_obs, action, new_obs, maze)
        
        # ノードに追加
        node = self._get_or_create_node(old_obs.position)
        node.episodes.append(episode)
        
        self.time_step += 1
        
    def visualize_integrated_knowledge(self, filename='integrated_knowledge.png'):
        """統合知識の可視化"""
        if not self.nodes:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # エピソード記憶グラフ
        G1 = nx.DiGraph()
        for pos, node in self.nodes.items():
            if node.episodes:
                avg_value = np.mean([ep.value for ep in node.episodes])
                G1.add_node(pos, value=avg_value, visits=node.visit_count)
                
        for pos, node in self.nodes.items():
            for ep in node.episodes:
                if ep.result['success'] and ep.result['new_position'] in self.nodes:
                    G1.add_edge(pos, ep.result['new_position'], 
                              action=ep.action, value=ep.value)
                    
        # メッセージパッシンググラフ
        G2 = nx.DiGraph()
        for pos, node in self.nodes.items():
            label = f"{pos}"
            if node.is_goal:
                G2.add_node(pos, label=label, color='gold')
            elif node.is_dead_end:
                G2.add_node(pos, label=label, color='red')
            elif node.is_junction:
                G2.add_node(pos, label=label, color='lightblue')
            else:
                G2.add_node(pos, label=label, color='lightgray')
                
        for node in self.nodes.values():
            for msg in node.outgoing_messages:
                if msg.to_destination in self.nodes:
                    if msg.value > 0:
                        G2.add_edge(node.position, msg.to_destination,
                                  color='green', weight=msg.value)
                    else:
                        G2.add_edge(node.position, msg.to_destination,
                                  color='red', weight=-msg.value)
                        
        # 描画
        if G1.nodes():
            pos1 = nx.spring_layout(G1, k=2)
            nx.draw(G1, pos1, ax=ax1, with_labels=True, node_size=300, 
                   font_size=8, arrows=True)
            ax1.set_title("エピソード記憶ネットワーク")
            
        if G2.nodes():
            pos2 = nx.spring_layout(G2, k=2)
            node_colors = [G2.nodes[n].get('color', 'gray') for n in G2.nodes()]
            nx.draw(G2, pos2, ax=ax2, node_color=node_colors,
                   with_labels=True, node_size=400, font_size=8, arrows=True)
            ax2.set_title("メッセージパッシングネットワーク")
            
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()


def demonstrate_integrated_gedig():
    """統合型geDIGナビゲーターのデモ"""
    print("統合型geDIGナビゲーター：エピソード記憶＋メッセージパッシング")
    print("=" * 70)
    print("特徴：")
    print("- エピソード記憶による経験の蓄積と活用")
    print("- メッセージパッシングによる知識の伝播")
    print("- 構造的類似性に基づく類推")
    print("- geDIG目的関数による探索と活用のバランス")
    print("=" * 70)
    
    config = MazeNavigatorConfig()
    config.w_ged = 1.0
    config.k_ig = 2.0
    
    # 複数試行で性能評価
    n_trials = 5
    results = []
    
    for trial in range(n_trials):
        print(f"\n試行 {trial + 1}/{n_trials}")
        print("-" * 40)
        
        np.random.seed(trial + 100)
        maze = SimpleMaze(size=(12, 12), maze_type='dfs')
        navigator = IntegratedGeDIGNavigator(config)
        
        print(f"迷路: {maze.size}")
        print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
        
        obs = maze.reset()
        steps = 0
        
        for _ in range(500):
            old_obs = obs
            action = navigator.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            navigator.update_after_action(old_obs, action, obs, maze)
            steps += 1
            
            if done and maze.agent_pos == maze.goal_pos:
                print(f"\n✅ ゴール到達！ステップ数: {steps}")
                results.append({
                    'success': True,
                    'steps': steps,
                    'nodes': len(navigator.nodes),
                    'episodes': sum(len(n.episodes) for n in navigator.nodes.values()),
                    'messages': sum(len(n.outgoing_messages) for n in navigator.nodes.values())
                })
                break
        else:
            print(f"\n❌ タイムアウト（{steps}ステップ）")
            results.append({
                'success': False,
                'steps': steps,
                'nodes': len(navigator.nodes),
                'episodes': sum(len(n.episodes) for n in navigator.nodes.values()),
                'messages': sum(len(n.outgoing_messages) for n in navigator.nodes.values())
            })
            
        # 最後の試行の知識を可視化
        if trial == n_trials - 1:
            navigator.visualize_integrated_knowledge()
            
        print(f"\n統計:")
        print(f"  訪問ノード数: {results[-1]['nodes']}")
        print(f"  総エピソード数: {results[-1]['episodes']}")
        print(f"  メッセージ数: {results[-1]['messages']}")
        
    # 全体統計
    print("\n" + "=" * 70)
    print("全試行の結果:")
    success_count = sum(1 for r in results if r['success'])
    success_results = [r for r in results if r['success']]
    
    print(f"成功率: {success_count}/{n_trials} ({success_count/n_trials*100:.0f}%)")
    
    if success_results:
        print(f"平均ステップ数（成功時）: {np.mean([r['steps'] for r in success_results]):.1f}")
        print(f"平均エピソード数: {np.mean([r['episodes'] for r in success_results]):.1f}")
        print(f"平均メッセージ数: {np.mean([r['messages'] for r in success_results]):.1f}")
        
    print("\n" + "=" * 70)
    print("✨ 統合型geDIGの特徴:")
    print("✨ エピソード記憶が類似状況での意思決定を支援")
    print("✨ メッセージパッシングが行き止まり情報を伝播")
    print("✨ 構造的類似性により未知の状況でも適切に行動")
    print("✨ これがInsightSpike AIの本質！")


if __name__ == "__main__":
    demonstrate_integrated_gedig()