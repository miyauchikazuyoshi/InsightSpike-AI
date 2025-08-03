#!/usr/bin/env python3
"""真のエピソード記憶によるgeDIGナビゲーション"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class Episode:
    """エピソード：状況-行動-結果の記録"""
    query: str  # "位置(x,y)でゴールに向かうには？"
    context: Dict  # 位置、ゴールまでの距離など
    action: int  # 選択した行動
    result: Dict  # 結果（新位置、成功/失敗、ゴールへの接近度）
    value: float  # このエピソードの価値


class TrueEpisodicGeDIGNavigator:
    """エピソード記憶から自律的に行動を決定"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: List[Episode] = []
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.current_pos: Optional[Tuple[int, int]] = None
        self.time_step = 0
        
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """マンハッタン距離"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def create_episode(self, old_pos: Tuple[int, int], action: int, 
                      new_pos: Tuple[int, int], possible_actions: List[int]):
        """エピソードを作成して記憶"""
        # クエリの生成
        if self.goal_pos:
            query = f"位置{old_pos}からゴール{self.goal_pos}に向かうには？"
            goal_dist_before = self._manhattan_distance(old_pos, self.goal_pos)
            goal_dist_after = self._manhattan_distance(new_pos, self.goal_pos)
        else:
            query = f"位置{old_pos}から探索を進めるには？"
            goal_dist_before = 0
            goal_dist_after = 0
            
        # コンテキスト
        context = {
            'position': old_pos,
            'goal_known': self.goal_pos is not None,
            'possible_actions': possible_actions,
            'time': self.time_step
        }
        
        # 結果
        success = old_pos != new_pos
        result = {
            'new_position': new_pos,
            'success': success,
            'goal_progress': goal_dist_before - goal_dist_after if self.goal_pos else (1 if success else 0)
        }
        
        # 価値の計算（成功度 × ゴール接近度）
        value = result['goal_progress'] if success else -0.5
        
        episode = Episode(
            query=query,
            context=context,
            action=action,
            result=result,
            value=value
        )
        
        self.episodes.append(episode)
        self.time_step += 1
        
    def query_action(self, position: Tuple[int, int], 
                    possible_actions: List[int]) -> int:
        """現在状況に基づいて行動を決定"""
        
        # 現在のクエリ
        if self.goal_pos:
            current_query = f"位置{position}からゴール{self.goal_pos}に向かうには？"
        else:
            current_query = f"位置{position}から探索を進めるには？"
            
        # 関連エピソードを検索（geDIG的な類似度計算）
        relevant_episodes = self._find_relevant_episodes(position, current_query)
        
        if not relevant_episodes:
            # 未知の状況では探索（高い情報利得）
            return np.random.choice(possible_actions)
            
        # 各行動の評価
        action_scores = defaultdict(list)
        
        for episode in relevant_episodes:
            # 構造的類似度（同じ位置からのエピソードは高い重み）
            if episode.context['position'] == position:
                similarity = 1.0
            else:
                dist = self._manhattan_distance(episode.context['position'], position)
                similarity = 1.0 / (1.0 + dist)
                
            # エピソードの価値を行動ごとに集計
            action_scores[episode.action].append(episode.value * similarity)
            
        # 各行動のgeDIG評価
        best_action = None
        best_score = float('-inf')
        
        for action in possible_actions:
            if action in action_scores:
                # 既知の行動：平均価値
                avg_value = np.mean(action_scores[action])
                # 情報利得は試行回数に反比例
                ig = 1.0 / (len(action_scores[action]) + 1)
            else:
                # 未知の行動：高い情報利得
                avg_value = 0.0
                ig = 2.0
                
            # geDIG目的関数
            score = self.config.w_ged * avg_value - self.config.k_ig * ig
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action if best_action is not None else np.random.choice(possible_actions)
        
    def _find_relevant_episodes(self, position: Tuple[int, int], 
                               query: str) -> List[Episode]:
        """関連するエピソードを検索"""
        relevant = []
        
        for episode in self.episodes:
            # 同じ位置のエピソード
            if episode.context['position'] == position:
                relevant.append(episode)
            # 近い位置のエピソード（構造的類似性）
            elif self._manhattan_distance(episode.context['position'], position) <= 2:
                relevant.append(episode)
                
        # 時間的に新しいものを優先
        relevant.sort(key=lambda e: e.context['time'], reverse=True)
        
        return relevant[:20]  # 最新20件まで
        
    def decide_action(self, obs, maze) -> int:
        """観測から行動を決定"""
        self.current_pos = obs.position
        
        # ゴール発見
        if obs.is_goal and not self.goal_pos:
            self.goal_pos = obs.position
            print(f"🎯 ゴール発見！位置: {self.goal_pos}")
            
        return self.query_action(obs.position, obs.possible_moves)
        
    def learn_from_experience(self, old_pos: Tuple[int, int], 
                            action: int, new_pos: Tuple[int, int], 
                            possible_actions: List[int]):
        """経験から学習（エピソード記憶に追加）"""
        self.create_episode(old_pos, action, new_pos, possible_actions)


def demonstrate_true_episodic_gedig():
    """真のエピソード記憶geDIGのデモンストレーション"""
    print("エピソード記憶による自律的geDIGナビゲーション")
    print("=" * 60)
    print("重要な概念：")
    print("- エピソード = クエリ（状況での問い）+ 行動 + 結果")
    print("- 類似状況のエピソードから行動を決定")
    print("- geDIG目的関数で探索と活用のバランス")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    
    # テスト実行
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    navigator = TrueEpisodicGeDIGNavigator(config)
    
    print(f"\n迷路: {maze.size}")
    print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
    print("-" * 40)
    
    obs = maze.reset()
    path = [obs.position]
    
    for step in range(200):
        old_pos = obs.position
        possible_actions = obs.possible_moves.copy()
        
        # エピソード記憶から行動を決定
        action = navigator.decide_action(obs, maze)
        
        # 行動実行
        obs, reward, done, info = maze.step(action)
        new_pos = obs.position
        
        # 経験から学習
        navigator.learn_from_experience(old_pos, action, new_pos, possible_actions)
        
        path.append(new_pos)
        
        # 重要な時点での状況
        if step % 20 == 0:
            print(f"ステップ {step}: 位置{old_pos} → {new_pos}")
            print(f"  エピソード数: {len(navigator.episodes)}")
            
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {step + 1}")
            break
            
    # エピソード記憶の分析
    print("\n" + "=" * 60)
    print("エピソード記憶の分析:")
    print(f"総エピソード数: {len(navigator.episodes)}")
    
    # 価値の高いエピソード
    valuable_episodes = sorted(navigator.episodes, key=lambda e: e.value, reverse=True)[:5]
    print("\n価値の高いエピソード（上位5件）:")
    for i, ep in enumerate(valuable_episodes):
        print(f"{i+1}. {ep.query}")
        print(f"   行動: {['上', '右', '下', '左'][ep.action]}, 価値: {ep.value:.2f}")
        
    # 位置ごとの訪問回数
    position_counts = defaultdict(int)
    for ep in navigator.episodes:
        position_counts[ep.context['position']] += 1
        
    print(f"\n訪問位置数: {len(position_counts)}")
    print(f"平均訪問回数: {np.mean(list(position_counts.values())):.1f}")
    
    # 経路の効率性
    if maze.agent_pos == maze.goal_pos:
        optimal_dist = abs(maze.start_pos[0] - maze.goal_pos[0]) + abs(maze.start_pos[1] - maze.goal_pos[1])
        actual_dist = len(path) - 1
        efficiency = optimal_dist / actual_dist if actual_dist > 0 else 0
        print(f"\n経路効率: {efficiency:.1%} (最適{optimal_dist}歩 / 実際{actual_dist}歩)")
    
    print("\n" + "=" * 60)
    print("まとめ：")
    print("✨ エピソード記憶が自律的に次の行動を決定")
    print("✨ 過去の経験から類似状況での最適行動を学習")
    print("✨ geDIG目的関数により探索と活用のバランスを実現")
    print("✨ これが本来のgeDIGの姿！")


if __name__ == "__main__":
    demonstrate_true_episodic_gedig()