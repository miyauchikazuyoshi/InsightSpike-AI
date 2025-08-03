#!/usr/bin/env python3
"""
No-Cheat Maze Agent
===================

ゴール位置を事前に知らない、より現実的な迷路探索エージェント
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from donut_search_maze import DonutSearchMaze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExplorationMemory:
    """探索の記憶"""
    position: Tuple[int, int]
    action: int
    result: str
    vector: np.ndarray
    step: int
    value: float = 0.0  # この経路の価値


class NoCheatMazeAgent:
    """チートなし迷路エージェント"""
    
    def __init__(self):
        self.memories: List[ExplorationMemory] = []
        self.position = (0, 0)
        self.step_count = 0
        self.found_goal = False
        self.goal_memory: Optional[ExplorationMemory] = None
        
        # 探索戦略
        self.exploration_rate = 0.8  # 初期は探索重視
        self.visited_positions = {}  # 位置 -> 訪問回数
        
    def add_memory(self, position: Tuple[int, int], action: int, 
                   result: str, vector: np.ndarray):
        """探索の記憶を追加"""
        memory = ExplorationMemory(
            position=position,
            action=action,
            result=result,
            vector=vector,
            step=self.step_count
        )
        
        # ゴールを発見したら記録
        if result == 'goal' and not self.found_goal:
            self.found_goal = True
            self.goal_memory = memory
            memory.value = 100.0  # ゴールは最高価値
            logger.info(f"🎯 Goal discovered at {position}!")
        elif result == 'empty':
            # 新しい場所の発見は価値がある
            if position not in self.visited_positions:
                memory.value = 10.0
            else:
                memory.value = 1.0 / (1 + self.visited_positions[position])
        else:  # wall
            memory.value = -5.0
            
        self.memories.append(memory)
        
        # 訪問回数を更新
        if position not in self.visited_positions:
            self.visited_positions[position] = 0
        self.visited_positions[position] += 1
        
    def decide_action(self, possible_actions: List[int]) -> int:
        """次の行動を決定（ゴール位置を知らない）"""
        
        if not possible_actions:
            return 0
            
        # 探索率に基づいて戦略を選択
        if np.random.random() < self.exploration_rate:
            # 探索モード：未訪問の方向を優先
            return self._exploration_strategy(possible_actions)
        else:
            # 活用モード：過去の良い経験を活用
            return self._exploitation_strategy(possible_actions)
    
    def _exploration_strategy(self, possible_actions: List[int]) -> int:
        """探索戦略：未知の領域を優先"""
        action_scores = {}
        
        for action in possible_actions:
            # この行動の予測位置
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            next_pos = (self.position[0] + dx, self.position[1] + dy)
            
            # 未訪問なら高スコア
            if next_pos not in self.visited_positions:
                action_scores[action] = 100.0
            else:
                # 訪問回数が少ないほど高スコア
                visit_count = self.visited_positions[next_pos]
                action_scores[action] = 10.0 / (1 + visit_count)
                
        # 最高スコアの行動を選択
        if action_scores:
            return max(action_scores.keys(), key=lambda a: action_scores[a])
        else:
            return np.random.choice(possible_actions)
    
    def _exploitation_strategy(self, possible_actions: List[int]) -> int:
        """活用戦略：過去の良い経験を活用"""
        if not self.memories:
            return np.random.choice(possible_actions)
            
        action_scores = {action: 0.0 for action in possible_actions}
        
        # 類似した状況での成功体験を探す
        current_visits = self.visited_positions.get(self.position, 0)
        
        for memory in self.memories:
            # 同じ位置での記憶
            if memory.position == self.position:
                if memory.action in possible_actions:
                    # 成功体験（特にゴール発見）は高く評価
                    action_scores[memory.action] += memory.value
                    
                    # 最近の記憶ほど重視
                    recency_bonus = 1.0 / (1 + self.step_count - memory.step)
                    action_scores[memory.action] += recency_bonus
        
        # ゴールを見つけた後は、ゴールへの経路を逆算
        if self.found_goal and self.goal_memory:
            # 簡単な経路逆算（本来はもっと賢い方法が必要）
            goal_x, goal_y = self.goal_memory.position
            curr_x, curr_y = self.position
            
            # マンハッタン距離で方向を推定
            dx = goal_x - curr_x
            dy = goal_y - curr_y
            
            if dx > 0 and 1 in possible_actions:  # 右
                action_scores[1] += 50.0
            elif dx < 0 and 3 in possible_actions:  # 左
                action_scores[3] += 50.0
                
            if dy > 0 and 2 in possible_actions:  # 下
                action_scores[2] += 50.0
            elif dy < 0 and 0 in possible_actions:  # 上
                action_scores[0] += 50.0
        
        # 最高スコアの行動を選択
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        
        # 全てのスコアが0なら探索に戻る
        if all(score == 0 for score in action_scores.values()):
            return self._exploration_strategy(possible_actions)
            
        return best_action
    
    def update_exploration_rate(self):
        """探索率を更新"""
        # ゴールを見つけたら探索率を下げる
        if self.found_goal:
            self.exploration_rate = 0.2
        else:
            # 時間とともに探索率を下げる（でも最低0.3は保つ）
            self.exploration_rate = max(0.3, 0.8 - self.step_count * 0.01)


def demonstrate_no_cheat():
    """チートなし探索のデモ"""
    print("=== チートなし迷路探索デモ ===\n")
    
    agent = NoCheatMazeAgent()
    
    # 仮想的な迷路での動作をシミュレート
    print("ゴール位置を知らない状態で探索開始...\n")
    
    # ステップ1: 初期位置から探索
    agent.position = (0, 0)
    possible_actions = [1, 2]  # 右と下が可能
    
    action = agent.decide_action(possible_actions)
    action_names = ['↑', '→', '↓', '←']
    print(f"Step 0: 位置{agent.position} → 行動: {action_names[action]}")
    print(f"  探索率: {agent.exploration_rate:.2f}")
    print(f"  戦略: {'探索' if np.random.random() < agent.exploration_rate else '活用'}")
    
    # 右に移動したとする
    agent.add_memory((0, 0), action, 'empty', np.array([0.0, 0.0, 0.25, 0.0, 0.1]))
    agent.position = (1, 0)
    agent.step_count += 1
    agent.update_exploration_rate()
    
    # ステップ2
    possible_actions = [1, 2, 3]
    action = agent.decide_action(possible_actions)
    print(f"\nStep 1: 位置{agent.position} → 行動: {action_names[action]}")
    print(f"  訪問済み位置: {list(agent.visited_positions.keys())}")
    
    # さらに探索...
    print("\n... 探索を続ける ...")
    
    # ゴール発見シミュレート
    agent.position = (2, 2)
    agent.add_memory((2, 2), 1, 'goal', np.array([1.0, 1.0, 0.25, 1.0, 0.1]))
    print(f"\n🎉 ゴール発見！位置: {agent.position}")
    print(f"  探索率が {0.8:.2f} から {agent.exploration_rate:.2f} に低下")
    print(f"  今後はゴールへの経路を活用して移動")


if __name__ == "__main__":
    demonstrate_no_cheat()