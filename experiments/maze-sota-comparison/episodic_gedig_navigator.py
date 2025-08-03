#!/usr/bin/env python3
"""エピソード記憶に基づく自律的geDIGナビゲーター（改良版）"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class NavigationEpisode:
    """ナビゲーションエピソード：位置での行動とその結果"""
    position: Tuple[int, int]
    action: int
    result_position: Tuple[int, int]
    success: bool
    goal_distance_before: float
    goal_distance_after: float
    timestamp: int
    
    @property
    def goal_progress(self) -> float:
        """ゴールへの接近度（正の値が良い）"""
        return self.goal_distance_before - self.goal_distance_after


@dataclass
class PositionMemory:
    """位置ごとの記憶"""
    position: Tuple[int, int]
    episodes: List[NavigationEpisode] = field(default_factory=list)
    visits: int = 0
    last_visit: int = 0
    
    def get_action_statistics(self, action: int) -> Dict:
        """特定の行動の統計を取得"""
        action_episodes = [e for e in self.episodes if e.action == action]
        if not action_episodes:
            return {'count': 0, 'success_rate': 0.0, 'avg_progress': 0.0}
            
        success_count = sum(1 for e in action_episodes if e.success)
        avg_progress = np.mean([e.goal_progress for e in action_episodes])
        
        return {
            'count': len(action_episodes),
            'success_rate': success_count / len(action_episodes),
            'avg_progress': avg_progress
        }


class EpisodicGeDIGNavigator:
    """エピソード記憶で自律的に行動を決定するナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.position_memories: Dict[Tuple[int, int], PositionMemory] = {}
        self.goal_position: Optional[Tuple[int, int]] = None
        self.current_position: Optional[Tuple[int, int]] = None
        self.time_step = 0
        self.path_history: List[Tuple[int, int]] = []
        
    def _get_or_create_memory(self, position: Tuple[int, int]) -> PositionMemory:
        """位置の記憶を取得または作成"""
        if position not in self.position_memories:
            self.position_memories[position] = PositionMemory(position=position)
        return self.position_memories[position]
        
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """マンハッタン距離"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def add_episode(self, position: Tuple[int, int], action: int, 
                   result_position: Tuple[int, int], success: bool):
        """エピソードを記憶に追加"""
        memory = self._get_or_create_memory(position)
        
        # ゴールまでの距離計算
        if self.goal_position:
            dist_before = self._manhattan_distance(position, self.goal_position)
            dist_after = self._manhattan_distance(result_position, self.goal_position)
        else:
            # ゴールが未知の場合は探索範囲の拡大を評価
            dist_before = 0
            dist_after = -1 if success else 0  # 新しい場所への移動を促進
            
        episode = NavigationEpisode(
            position=position,
            action=action,
            result_position=result_position,
            success=success,
            goal_distance_before=dist_before,
            goal_distance_after=dist_after,
            timestamp=self.time_step
        )
        
        memory.episodes.append(episode)
        memory.visits += 1
        memory.last_visit = self.time_step
        self.time_step += 1
        
    def query_best_action(self, position: Tuple[int, int], 
                         possible_actions: List[int]) -> int:
        """位置での最適行動をクエリ"""
        memory = self._get_or_create_memory(position)
        
        # 各行動のgeDIGスコアを計算
        action_scores = {}
        
        for action in range(4):
            if action not in possible_actions:
                action_scores[action] = float('-inf')
                continue
                
            stats = memory.get_action_statistics(action)
            
            # グラフ編集距離（GED）的な評価
            if stats['count'] == 0:
                # 未探索は高い情報利得
                ged_score = 0.0
                ig_score = 3.0  # 高い情報利得
            else:
                # 成功率とゴール接近度から評価
                ged_score = stats['success_rate'] * (1 + stats['avg_progress'])
                # 探索回数が多いほど情報利得は低い
                ig_score = 1.0 / (stats['count'] + 1)
                
            # 時間的減衰を考慮（最近の記憶を重視）
            recency_factor = 1.0
            if memory.last_visit > 0:
                recency_factor = np.exp(-(self.time_step - memory.last_visit) * 0.01)
                
            # geDIG目的関数
            gediq_score = self.config.w_ged * ged_score - self.config.k_ig * ig_score
            gediq_score *= recency_factor
            
            action_scores[action] = gediq_score
            
        # 最高スコアの行動を選択（ε-greedy）
        if np.random.random() < self.config.exploration_epsilon:
            return np.random.choice(possible_actions)
        else:
            valid_actions = [(a, s) for a, s in action_scores.items() 
                           if a in possible_actions]
            if valid_actions:
                return max(valid_actions, key=lambda x: x[1])[0]
            else:
                return np.random.choice(possible_actions)
                
    def decide_action(self, obs, maze) -> int:
        """観測に基づいて行動を決定"""
        self.current_position = obs.position
        self.path_history.append(self.current_position)
        
        # ゴール発見
        if obs.is_goal and not self.goal_position:
            self.goal_position = self.current_position
            print(f"🎯 ゴール発見！位置: {self.goal_position}")
            
        # 最適行動をクエリ
        return self.query_best_action(self.current_position, obs.possible_moves)
        
    def update_after_move(self, old_pos: Tuple[int, int], 
                         new_pos: Tuple[int, int], action: int):
        """移動後の更新"""
        success = old_pos != new_pos
        self.add_episode(old_pos, action, new_pos, success)
        
    def get_exploration_statistics(self) -> Dict:
        """探索統計を取得"""
        total_episodes = sum(len(m.episodes) for m in self.position_memories.values())
        successful_episodes = sum(
            sum(1 for e in m.episodes if e.success) 
            for m in self.position_memories.values()
        )
        
        return {
            'positions_visited': len(self.position_memories),
            'total_episodes': total_episodes,
            'successful_episodes': successful_episodes,
            'failure_episodes': total_episodes - successful_episodes,
            'average_visits_per_position': total_episodes / len(self.position_memories) if self.position_memories else 0
        }


def visualize_episodic_navigation():
    """エピソード記憶ナビゲーションの可視化"""
    print("エピソード記憶による自律的geDIGナビゲーション")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    
    # 複数の迷路でテスト
    n_trials = 5
    results = []
    
    for trial in range(n_trials):
        print(f"\n試行 {trial + 1}/{n_trials}")
        print("-" * 40)
        
        np.random.seed(trial)
        maze = SimpleMaze(size=(15, 15), maze_type='dfs')
        navigator = EpisodicGeDIGNavigator(config)
        
        obs = maze.reset()
        steps = 0
        
        for _ in range(500):
            old_pos = obs.position
            action = navigator.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            new_pos = obs.position
            
            navigator.update_after_move(old_pos, new_pos, action)
            steps += 1
            
            if done and maze.agent_pos == maze.goal_pos:
                print(f"✅ ゴール到達！ステップ数: {steps}")
                break
                
        stats = navigator.get_exploration_statistics()
        stats['steps'] = steps
        stats['success'] = maze.agent_pos == maze.goal_pos
        results.append(stats)
        
        print(f"探索統計:")
        print(f"  訪問位置数: {stats['positions_visited']}")
        print(f"  総エピソード数: {stats['total_episodes']}")
        print(f"  成功/失敗: {stats['successful_episodes']}/{stats['failure_episodes']}")
    
    # 結果のまとめ
    print("\n" + "=" * 60)
    print("全試行の結果:")
    
    success_count = sum(1 for r in results if r['success'])
    avg_steps = np.mean([r['steps'] for r in results if r['success']])
    avg_positions = np.mean([r['positions_visited'] for r in results])
    
    print(f"成功率: {success_count}/{n_trials} ({success_count/n_trials*100:.0f}%)")
    print(f"平均ステップ数（成功時）: {avg_steps:.1f}")
    print(f"平均探索位置数: {avg_positions:.1f}")
    
    print("\n重要な特徴:")
    print("✨ エピソード記憶から自律的に行動決定")
    print("✨ ゴール位置は探索中に発見")
    print("✨ 失敗経験も貴重な記憶として活用")
    print("✨ geDIG目的関数により効率的な探索を実現")
    print("=" * 60)


if __name__ == "__main__":
    visualize_episodic_navigation()