#!/usr/bin/env python3
"""自律的なエピソード記憶に基づくgeDIGナビゲーター"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class MovementEpisode:
    """移動エピソード：位置Aから位置Bへの移動試行の記録"""
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    action: int  # 0:上, 1:右, 2:下, 3:左
    success: bool
    distance_to_goal: float  # ゴールまでの距離変化
    timestamp: int


@dataclass 
class PositionQuery:
    """位置クエリ：特定の位置で次に進むべき方向を問う"""
    current_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    context: str = "ゴールに到達するために次に進むべき方向は？"


class AutonomousGeDIGNavigator:
    """エピソード記憶から自律的に行動を決定するナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episode_memory: List[MovementEpisode] = []
        self.position_history: List[Tuple[int, int]] = []
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.time_step = 0
        
    def add_movement_episode(self, from_pos: Tuple[int, int], 
                           to_pos: Tuple[int, int], 
                           action: int, 
                           success: bool):
        """移動エピソードを記憶に追加"""
        if self.goal_pos:
            # ゴールまでの距離変化を計算
            dist_before = self._manhattan_distance(from_pos, self.goal_pos)
            dist_after = self._manhattan_distance(to_pos, self.goal_pos)
            distance_change = dist_before - dist_after  # 正の値ならゴールに近づいた
        else:
            distance_change = 0.0
            
        episode = MovementEpisode(
            from_pos=from_pos,
            to_pos=to_pos,
            action=action,
            success=success,
            distance_to_goal=distance_change,
            timestamp=self.time_step
        )
        
        self.episode_memory.append(episode)
        self.time_step += 1
        
    def query_next_action(self, current_pos: Tuple[int, int]) -> int:
        """現在位置でのクエリ：次に進むべき方向は？"""
        
        # 関連するエピソードを検索
        relevant_episodes = self._find_relevant_episodes(current_pos)
        
        if not relevant_episodes:
            # 記憶にない場合は探索（ランダム）
            return np.random.randint(0, 4)
            
        # geDIG的な評価：構造的類似性と情報利得のバランス
        best_action = self._evaluate_actions_gedig(current_pos, relevant_episodes)
        
        return best_action
        
    def _find_relevant_episodes(self, pos: Tuple[int, int]) -> List[MovementEpisode]:
        """現在位置に関連するエピソードを検索"""
        relevant = []
        
        for episode in self.episode_memory:
            # 同じ位置からの移動
            if episode.from_pos == pos:
                relevant.append(episode)
            # 近い位置からの移動（構造的類似性）
            elif self._manhattan_distance(episode.from_pos, pos) <= 2:
                relevant.append(episode)
                
        return relevant
        
    def _evaluate_actions_gedig(self, pos: Tuple[int, int], 
                               episodes: List[MovementEpisode]) -> int:
        """geDIG的な行動評価"""
        action_scores = {}
        
        for action in range(4):
            # この行動に関するエピソードを収集
            action_episodes = [e for e in episodes if e.action == action]
            
            if not action_episodes:
                # 未探索の行動は高い情報利得
                action_scores[action] = self.config.k_ig * 2.0
                continue
                
            # GED（グラフ編集距離）的な評価
            # 成功したエピソードほど価値が高い
            success_rate = sum(1 for e in action_episodes if e.success) / len(action_episodes)
            
            # ゴールへの接近度
            avg_goal_progress = np.mean([e.distance_to_goal for e in action_episodes])
            
            # 時間的減衰（新しい記憶ほど重要）
            recency_weight = np.mean([
                np.exp(-(self.time_step - e.timestamp) * 0.1) 
                for e in action_episodes
            ])
            
            # geDIG評価関数
            ged_score = success_rate * avg_goal_progress * recency_weight
            ig_score = 1.0 / (len(action_episodes) + 1)  # 試行回数が少ないほど情報利得大
            
            action_scores[action] = (
                self.config.w_ged * ged_score - 
                self.config.k_ig * ig_score
            )
            
        # 最高スコアの行動を選択
        return max(action_scores.items(), key=lambda x: x[1])[0]
        
    def _manhattan_distance(self, pos1: Tuple[int, int], 
                           pos2: Tuple[int, int]) -> float:
        """マンハッタン距離"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def decide_action(self, obs, maze) -> int:
        """観測に基づいて行動を決定（インターフェース互換性のため）"""
        current_pos = obs.position
        
        # ゴール位置を記録
        if obs.is_goal:
            self.goal_pos = current_pos
            
        # エピソード記憶に基づいて行動を決定
        action = self.query_next_action(current_pos)
        
        # 可能な行動のみを選択
        if action not in obs.possible_moves:
            # 壁にぶつかることも記憶として重要
            self.add_movement_episode(
                from_pos=current_pos,
                to_pos=current_pos,  # 移動できなかった
                action=action,
                success=False
            )
            # 別の行動を選択
            if obs.possible_moves:
                action = np.random.choice(obs.possible_moves)
            else:
                action = 0
                
        return action
        
    def update_after_action(self, old_pos: Tuple[int, int], 
                           new_pos: Tuple[int, int], 
                           action: int):
        """行動後の更新"""
        success = old_pos != new_pos  # 移動できたかどうか
        self.add_movement_episode(old_pos, new_pos, action, success)
        
    def explain_decision(self, pos: Tuple[int, int]) -> str:
        """意思決定の説明（解釈可能性）"""
        episodes = self._find_relevant_episodes(pos)
        
        explanation = f"位置{pos}での意思決定:\n"
        explanation += f"関連エピソード数: {len(episodes)}\n"
        
        for action in range(4):
            action_name = ['上', '右', '下', '左'][action]
            action_eps = [e for e in episodes if e.action == action]
            
            if action_eps:
                success_rate = sum(1 for e in action_eps if e.success) / len(action_eps)
                explanation += f"  {action_name}: 成功率{success_rate:.1%}, 試行{len(action_eps)}回\n"
            else:
                explanation += f"  {action_name}: 未探索\n"
                
        return explanation


def demonstrate_autonomous_navigation():
    """自律的ナビゲーションのデモンストレーション"""
    print("自律的エピソード記憶によるgeDIGナビゲーション")
    print("=" * 60)
    
    config = MazeNavigatorConfig(
        ged_weight=1.0,
        ig_weight=2.0,
        temperature=1.0,
        exploration_epsilon=0.0
    )
    
    # 迷路を作成
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    
    # ナビゲーター作成
    navigator = AutonomousGeDIGNavigator(config)
    
    # ナビゲーション実行
    obs = maze.reset()
    path = [obs.position]
    
    print(f"スタート: {maze.start_pos}")
    print(f"ゴール: {maze.goal_pos}")
    print("-" * 40)
    
    for step in range(200):
        old_pos = obs.position
        
        # 行動決定（エピソード記憶から自律的に）
        action = navigator.decide_action(obs, maze)
        
        # 行動実行
        obs, reward, done, info = maze.step(action)
        new_pos = obs.position
        
        # エピソード記憶を更新
        navigator.update_after_action(old_pos, new_pos, action)
        
        path.append(new_pos)
        
        # 重要な時点での説明
        if step % 20 == 0 or obs.is_junction or obs.is_dead_end:
            print(f"\nステップ {step}: 位置{old_pos}")
            print(navigator.explain_decision(old_pos))
            
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n🎉 ゴール到達！ステップ数: {step + 1}")
            break
            
    # 最終統計
    print("\n" + "=" * 60)
    print("エピソード記憶統計:")
    print(f"総エピソード数: {len(navigator.episode_memory)}")
    
    success_episodes = [e for e in navigator.episode_memory if e.success]
    print(f"成功エピソード: {len(success_episodes)}")
    print(f"失敗エピソード: {len(navigator.episode_memory) - len(success_episodes)}")
    
    # 位置ごとの学習状況
    position_visits = {}
    for episode in navigator.episode_memory:
        pos = episode.from_pos
        if pos not in position_visits:
            position_visits[pos] = 0
        position_visits[pos] += 1
        
    print(f"訪問位置数: {len(position_visits)}")
    print(f"平均訪問回数: {np.mean(list(position_visits.values())):.1f}")
    
    print("\n重要な洞察:")
    print("- エピソード記憶が自律的に行動を決定")
    print("- 失敗も重要な記憶として活用")
    print("- 構造的類似性により未知の状況でも対応可能")
    print("- geDIG的評価により効率的な探索を実現")


if __name__ == "__main__":
    demonstrate_autonomous_navigation()