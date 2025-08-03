#!/usr/bin/env python3
"""最終版geDIGナビゲーター：実用的な統合実装"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class ActionMemory:
    """行動の記憶：位置での行動とその結果"""
    position: Tuple[int, int]
    action: int
    result_position: Tuple[int, int]
    success: bool
    is_dead_end: bool = False
    is_goal: bool = False
    goal_progress: float = 0.0
    visit_time: int = 0


@dataclass
class PositionNode:
    """位置ノード：行動記憶と経路情報を持つ"""
    position: Tuple[int, int]
    action_memories: List[ActionMemory] = field(default_factory=list)
    visit_count: int = 0
    is_junction: bool = False
    is_dead_end: bool = False
    is_goal: bool = False
    dead_end_actions: Set[int] = field(default_factory=set)  # 行き止まりにつながる行動
    goal_path_actions: Set[int] = field(default_factory=set)  # ゴールにつながる行動
    
    def get_action_value(self, action: int) -> float:
        """行動の価値を計算"""
        # ゴールへの経路なら高評価
        if action in self.goal_path_actions:
            return 10.0
            
        # 行き止まりなら低評価
        if action in self.dead_end_actions:
            return -10.0
            
        # 行動記憶から評価
        memories = [m for m in self.action_memories if m.action == action]
        if memories:
            # 成功率と進捗度で評価
            success_rate = sum(1 for m in memories if m.success) / len(memories)
            avg_progress = np.mean([m.goal_progress for m in memories])
            return success_rate * 2.0 + avg_progress
        
        return 0.0
        
    def get_action_count(self, action: int) -> int:
        """行動の実行回数"""
        return sum(1 for m in self.action_memories if m.action == action)


class FinalGeDIGNavigator:
    """最終版geDIGナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.nodes: Dict[Tuple[int, int], PositionNode] = {}
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.time_step = 0
        
        # 経路追跡
        self.path_history: List[Tuple[int, int]] = []
        self.current_path_start: Optional[Tuple[int, int]] = None
        self.current_path_action: Optional[int] = None
        
        # デッドエンド回避
        self.recent_positions = deque(maxlen=10)  # 最近の位置履歴
        
    def _get_or_create_node(self, pos: Tuple[int, int]) -> PositionNode:
        """ノードを取得または作成"""
        if pos not in self.nodes:
            self.nodes[pos] = PositionNode(position=pos)
        return self.nodes[pos]
        
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """マンハッタン距離"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def _detect_loop(self) -> bool:
        """ループを検出"""
        if len(self.recent_positions) < 4:
            return False
        # 同じ位置に繰り返し戻っているか
        position_counts = defaultdict(int)
        for pos in self.recent_positions:
            position_counts[pos] += 1
        return any(count >= 3 for count in position_counts.values())
        
    def _propagate_dead_end(self, path: List[Tuple[int, int]], start_action: int):
        """行き止まり情報を経路上に伝播"""
        if len(path) < 2:
            return
            
        # 開始点に行き止まり情報を記録
        start_node = self._get_or_create_node(path[0])
        start_node.dead_end_actions.add(start_action)
        
        # 経路上の各ノードに情報を伝播
        for i in range(len(path) - 1):
            node = self._get_or_create_node(path[i])
            next_pos = path[i + 1]
            
            # どの行動で次の位置に移動したか判定
            for action in range(4):
                dx, dy = SimpleMaze.ACTIONS[action]
                if (path[i][0] + dx, path[i][1] + dy) == next_pos:
                    node.dead_end_actions.add(action)
                    break
                    
    def _propagate_goal_path(self, path: List[Tuple[int, int]], start_action: int):
        """ゴール経路情報を伝播"""
        if len(path) < 2:
            return
            
        # 開始点にゴール経路情報を記録
        start_node = self._get_or_create_node(path[0])
        start_node.goal_path_actions.add(start_action)
        
        # 経路上の各ノードに情報を伝播
        for i in range(len(path) - 1):
            node = self._get_or_create_node(path[i])
            next_pos = path[i + 1]
            
            # どの行動で次の位置に移動したか判定
            for action in range(4):
                dx, dy = SimpleMaze.ACTIONS[action]
                if (path[i][0] + dx, path[i][1] + dy) == next_pos:
                    node.goal_path_actions.add(action)
                    break
                    
    def decide_action(self, obs, maze) -> int:
        """観測から行動を決定"""
        current_pos = obs.position
        current_node = self._get_or_create_node(current_pos)
        current_node.visit_count += 1
        
        # ノード属性更新
        current_node.is_junction = obs.is_junction
        current_node.is_dead_end = obs.is_dead_end
        current_node.is_goal = obs.is_goal
        
        # 位置履歴更新
        self.recent_positions.append(current_pos)
        
        # ゴール発見
        if obs.is_goal and not self.goal_pos:
            self.goal_pos = current_pos
            print(f"🎯 ゴール発見！位置: {self.goal_pos}")
            # ゴールまでの経路を記録
            if self.current_path_start and self.current_path_action is not None:
                full_path = [self.current_path_start] + self.path_history[self.path_history.index(self.current_path_start)+1:]
                self._propagate_goal_path(full_path, self.current_path_action)
                
        # 行き止まり到達
        if obs.is_dead_end:
            print(f"💀 行き止まり到達: {current_pos}")
            if self.current_path_start and self.current_path_action is not None:
                # 現在の経路を行き止まりとして記録
                start_idx = self.path_history.index(self.current_path_start) if self.current_path_start in self.path_history else 0
                dead_path = self.path_history[start_idx:]
                self._propagate_dead_end(dead_path, self.current_path_action)
                
        # 新しい経路の開始判定
        if obs.is_junction or current_node.visit_count == 1 or obs.is_dead_end:
            self.current_path_start = current_pos
            
        # 各行動の評価
        action_scores = {}
        
        for action in obs.possible_moves:
            # 基本価値
            base_value = current_node.get_action_value(action)
            
            # 試行回数による情報利得
            action_count = current_node.get_action_count(action)
            ig = 2.0 / (action_count + 1)
            
            # ループ検出時のペナルティ
            loop_penalty = 0.0
            if self._detect_loop():
                # 最も試行されていない行動を優遇
                loop_penalty = -action_count * 2.0
                
            # 最終スコア
            score = base_value + self.config.k_ig * ig + loop_penalty
            action_scores[action] = score
            
        # 最適行動を選択
        if action_scores:
            # ε-greedy戦略
            if np.random.random() < 0.1:  # 10%の確率でランダム
                best_action = np.random.choice(obs.possible_moves)
            else:
                best_action = max(action_scores.items(), key=lambda x: x[1])[0]
                
            # デバッグ情報
            if current_node.visit_count <= 2 or obs.is_junction:
                print(f"\n位置{current_pos}での意思決定 (訪問{current_node.visit_count}回目):")
                for a in obs.possible_moves:
                    print(f"  {['上','右','下','左'][a]}: {action_scores[a]:.2f}")
                print(f"  → 選択: {['上','右','下','左'][best_action]}")
                
            # 新経路の開始行動を記録
            if current_pos == self.current_path_start:
                self.current_path_action = best_action
                
            return best_action
        else:
            return np.random.choice([0, 1, 2, 3])  # 緊急時
            
    def update_after_action(self, old_pos: Tuple[int, int], action: int, 
                           new_pos: Tuple[int, int], obs):
        """行動後の更新"""
        # 経路履歴更新
        if new_pos not in self.path_history or self.path_history[-1] != new_pos:
            self.path_history.append(new_pos)
            
        # 行動記憶を作成
        success = old_pos != new_pos
        goal_progress = 0.0
        
        if self.goal_pos and success:
            dist_before = self._manhattan_distance(old_pos, self.goal_pos)
            dist_after = self._manhattan_distance(new_pos, self.goal_pos)
            goal_progress = dist_before - dist_after
            
        memory = ActionMemory(
            position=old_pos,
            action=action,
            result_position=new_pos,
            success=success,
            is_dead_end=obs.is_dead_end,
            is_goal=obs.is_goal,
            goal_progress=goal_progress,
            visit_time=self.time_step
        )
        
        # ノードに記憶を追加
        node = self._get_or_create_node(old_pos)
        node.action_memories.append(memory)
        
        self.time_step += 1
        
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        total_memories = sum(len(n.action_memories) for n in self.nodes.values())
        dead_ends = sum(1 for n in self.nodes.values() if n.is_dead_end)
        junctions = sum(1 for n in self.nodes.values() if n.is_junction)
        
        return {
            'nodes': len(self.nodes),
            'memories': total_memories,
            'dead_ends': dead_ends,
            'junctions': junctions,
            'goal_found': self.goal_pos is not None
        }


def demonstrate_final_gedig():
    """最終版geDIGナビゲーターのデモ"""
    print("最終版geDIGナビゲーター：実用的な統合実装")
    print("=" * 60)
    print("特徴：")
    print("- 行動記憶による経験の蓄積")
    print("- 行き止まり情報の即座の伝播")
    print("- ループ検出と回避")
    print("- シンプルで効率的な実装")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    config.w_ged = 1.0
    config.k_ig = 2.0
    
    # 複数試行で性能評価
    n_trials = 5
    results = []
    
    for trial in range(n_trials):
        print(f"\n試行 {trial + 1}/{n_trials}")
        print("-" * 40)
        
        np.random.seed(trial + 200)
        maze = SimpleMaze(size=(15, 15), maze_type='dfs')
        navigator = FinalGeDIGNavigator(config)
        
        print(f"迷路: {maze.size}")
        print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
        
        obs = maze.reset()
        steps = 0
        
        for _ in range(1000):  # より長い制限時間
            old_pos = obs.position
            action = navigator.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            navigator.update_after_action(old_pos, action, obs.position, obs)
            steps += 1
            
            # 進捗表示
            if steps % 100 == 0:
                stats = navigator.get_statistics()
                print(f"  ステップ{steps}: ノード{stats['nodes']}, 行き止まり{stats['dead_ends']}")
                
            if done and maze.agent_pos == maze.goal_pos:
                print(f"\n✅ ゴール到達！ステップ数: {steps}")
                stats = navigator.get_statistics()
                results.append({
                    'success': True,
                    'steps': steps,
                    **stats
                })
                break
        else:
            print(f"\n❌ タイムアウト（{steps}ステップ）")
            stats = navigator.get_statistics()
            results.append({
                'success': False,
                'steps': steps,
                **stats
            })
            
        # 統計表示
        print(f"\n探索統計:")
        print(f"  訪問ノード数: {stats['nodes']}")
        print(f"  行動記憶数: {stats['memories']}")
        print(f"  発見した行き止まり: {stats['dead_ends']}")
        print(f"  分岐点: {stats['junctions']}")
        
    # 全体統計
    print("\n" + "=" * 60)
    print("全試行の結果:")
    success_count = sum(1 for r in results if r['success'])
    success_results = [r for r in results if r['success']]
    
    print(f"成功率: {success_count}/{n_trials} ({success_count/n_trials*100:.0f}%)")
    
    if success_results:
        print(f"平均ステップ数（成功時）: {np.mean([r['steps'] for r in success_results]):.1f}")
        print(f"平均ノード数: {np.mean([r['nodes'] for r in success_results]):.1f}")
        print(f"平均行き止まり発見数: {np.mean([r['dead_ends'] for r in success_results]):.1f}")
        
    print("\n" + "=" * 60)
    print("✨ 最終版geDIGの実装：")
    print("✨ シンプルながら効果的な行動記憶")
    print("✨ 行き止まり情報の即座の活用")
    print("✨ ループ検出による無限ループ回避")
    print("✨ 実用的で拡張可能な設計")


if __name__ == "__main__":
    demonstrate_final_gedig()