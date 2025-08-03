#!/usr/bin/env python3
"""メッセージパッシングによる意思決定と行き止まり記憶を活用するナビゲーター"""

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
class PathMemory:
    """経路の記憶：どの分岐からどこへ到達したか"""
    junction_pos: Tuple[int, int]  # 分岐点
    action: int  # 選択した行動
    path: List[Tuple[int, int]]  # 辿った経路
    destination: Tuple[int, int]  # 到達地点
    is_dead_end: bool  # 行き止まりか
    is_goal: bool  # ゴールか
    distance_traveled: int  # 移動距離
    value: float  # この経路の価値


@dataclass
class PositionNode:
    """位置ノード：メッセージパッシングの単位"""
    position: Tuple[int, int]
    is_junction: bool = False
    is_dead_end: bool = False
    is_goal: bool = False
    neighbors: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    messages_in: Dict[Tuple[int, int], float] = field(default_factory=dict)
    messages_out: Dict[Tuple[int, int], float] = field(default_factory=dict)
    visit_count: int = 0
    path_memories: List[PathMemory] = field(default_factory=list)


class MessagePassingNavigator:
    """メッセージパッシングで意思決定するナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.nodes: Dict[Tuple[int, int], PositionNode] = {}
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.current_path: List[Tuple[int, int]] = []
        self.last_junction: Optional[Tuple[int, int]] = None
        self.last_junction_action: Optional[int] = None
        
    def _get_or_create_node(self, pos: Tuple[int, int]) -> PositionNode:
        """ノードを取得または作成"""
        if pos not in self.nodes:
            self.nodes[pos] = PositionNode(position=pos)
        return self.nodes[pos]
        
    def _propagate_dead_end_message(self, dead_end_pos: Tuple[int, int], 
                                   junction_pos: Tuple[int, int], 
                                   action: int):
        """行き止まり情報を分岐点に伝播"""
        # 行き止まりから分岐点へのメッセージ
        dead_end_node = self._get_or_create_node(dead_end_pos)
        junction_node = self._get_or_create_node(junction_pos)
        
        # 負の価値を伝播（行き止まりは避けるべき）
        message_value = -10.0
        dead_end_node.messages_out[junction_pos] = message_value
        junction_node.messages_in[dead_end_pos] = message_value
        
        # 分岐点の特定の行動に対する評価を下げる
        path_memory = PathMemory(
            junction_pos=junction_pos,
            action=action,
            path=self.current_path.copy(),
            destination=dead_end_pos,
            is_dead_end=True,
            is_goal=False,
            distance_traveled=len(self.current_path),
            value=-1.0  # 行き止まりは負の価値
        )
        junction_node.path_memories.append(path_memory)
        
        print(f"💀 行き止まり{dead_end_pos}の情報を分岐点{junction_pos}の行動{['上','右','下','左'][action]}に伝播")
        
    def _propagate_goal_message(self, goal_pos: Tuple[int, int], 
                               junction_pos: Tuple[int, int], 
                               action: int):
        """ゴール情報を分岐点に伝播"""
        goal_node = self._get_or_create_node(goal_pos)
        junction_node = self._get_or_create_node(junction_pos)
        
        # 正の価値を伝播
        message_value = 10.0
        goal_node.messages_out[junction_pos] = message_value
        junction_node.messages_in[goal_pos] = message_value
        
        # 成功経路の記憶
        path_memory = PathMemory(
            junction_pos=junction_pos,
            action=action,
            path=self.current_path.copy(),
            destination=goal_pos,
            is_dead_end=False,
            is_goal=True,
            distance_traveled=len(self.current_path),
            value=10.0  # ゴールは高い価値
        )
        junction_node.path_memories.append(path_memory)
        
        print(f"🎯 ゴール{goal_pos}の情報を分岐点{junction_pos}の行動{['上','右','下','左'][action]}に伝播")
        
    def _run_message_passing(self):
        """メッセージパッシングを実行"""
        # 各ノードがメッセージを更新
        for _ in range(3):  # 数回の反復
            for node in self.nodes.values():
                if node.is_goal:
                    # ゴールから正のメッセージ
                    for neighbor_pos in node.neighbors.values():
                        if neighbor_pos in self.nodes:
                            node.messages_out[neighbor_pos] = 10.0
                            
                elif node.is_dead_end:
                    # 行き止まりから負のメッセージ
                    for neighbor_pos in node.neighbors.values():
                        if neighbor_pos in self.nodes:
                            node.messages_out[neighbor_pos] = -10.0
                            
                else:
                    # 中間ノードは受信メッセージを集約して転送
                    if node.messages_in:
                        avg_message = np.mean(list(node.messages_in.values()))
                        for neighbor_pos in node.neighbors.values():
                            if neighbor_pos in self.nodes:
                                node.messages_out[neighbor_pos] = avg_message * 0.9  # 減衰
                                
    def decide_action(self, obs, maze) -> int:
        """メッセージパッシングに基づいて行動を決定"""
        current_pos = obs.position
        current_node = self._get_or_create_node(current_pos)
        current_node.visit_count += 1
        
        # ノードの属性を更新
        current_node.is_junction = obs.is_junction
        current_node.is_dead_end = obs.is_dead_end
        current_node.is_goal = obs.is_goal
        
        # 隣接ノードを記録
        for action in obs.possible_moves:
            dx, dy = maze.ACTIONS[action]
            neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
            current_node.neighbors[action] = neighbor_pos
            
        # ゴール発見
        if obs.is_goal and not self.goal_pos:
            self.goal_pos = current_pos
            print(f"🎯 ゴール発見！位置: {current_pos}")
            if self.last_junction and self.last_junction_action is not None:
                self._propagate_goal_message(current_pos, self.last_junction, 
                                           self.last_junction_action)
                
        # 行き止まり到達
        if obs.is_dead_end and self.last_junction and self.last_junction_action is not None:
            self._propagate_dead_end_message(current_pos, self.last_junction, 
                                           self.last_junction_action)
            # 分岐点に戻る必要があることを示す
            print(f"💀 行き止まり{current_pos}に到達、分岐点{self.last_junction}に戻る必要")
            
        # 分岐点の記録
        if obs.is_junction:
            if current_pos != self.last_junction:  # 新しい分岐点
                print(f"🔀 分岐点{current_pos}を発見（可能な行動: {obs.possible_moves}）")
                self.last_junction = current_pos
                self.current_path = [current_pos]
        else:
            if current_pos not in self.current_path:
                self.current_path.append(current_pos)
            
        # メッセージパッシングを実行
        self._run_message_passing()
        
        # 各行動の評価
        action_scores = {}
        
        for action in obs.possible_moves:
            neighbor_pos = current_node.neighbors[action]
            
            # その方向の経路記憶を確認
            relevant_memories = [m for m in current_node.path_memories 
                               if m.action == action]
            
            if relevant_memories:
                # 既知の経路
                memory_values = [m.value for m in relevant_memories]
                avg_value = np.mean(memory_values)
                
                # 行き止まりの記憶があれば大幅に減点
                dead_end_count = sum(1 for m in relevant_memories if m.is_dead_end)
                if dead_end_count > 0:
                    avg_value -= 5.0 * dead_end_count
                    
                # ゴールの記憶があれば大幅に加点
                goal_count = sum(1 for m in relevant_memories if m.is_goal)
                if goal_count > 0:
                    avg_value += 5.0 * goal_count
                    
                ig = 1.0 / (len(relevant_memories) + 1)
            else:
                # 未知の経路
                avg_value = 0.0
                ig = 3.0  # 高い情報利得
                
            # 隣接ノードからのメッセージも考慮
            if neighbor_pos in self.nodes and current_pos in self.nodes[neighbor_pos].messages_out:
                message_value = self.nodes[neighbor_pos].messages_out[current_pos]
                avg_value += message_value * 0.5
                
            # geDIG評価
            action_scores[action] = self.config.w_ged * avg_value - self.config.k_ig * ig
            
        # 最高スコアの行動を選択
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            best_score = action_scores[best_action]
            
            # デバッグ情報
            if current_node.visit_count <= 2 or obs.is_junction:
                print(f"\n位置{current_pos}での行動評価:")
                for a, score in action_scores.items():
                    if a in obs.possible_moves:
                        dir_name = ['上','右','下','左'][a]
                        print(f"  {dir_name}: スコア={score:.2f}")
                print(f"  → 選択: {['上','右','下','左'][best_action]}")
            
            if obs.is_junction:
                self.last_junction_action = best_action
            return best_action
        else:
            return np.random.choice(obs.possible_moves)
            
    def visualize_knowledge_graph(self):
        """知識グラフの可視化"""
        if not self.nodes:
            return
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # ノードの描画
        for node in self.nodes.values():
            x, y = node.position[1], -node.position[0]  # 座標変換
            
            # ノードの色
            if node.is_goal:
                color = 'gold'
                marker = '*'
                size = 200
            elif node.is_dead_end:
                color = 'red'
                marker = 'x'
                size = 150
            elif node.is_junction:
                color = 'blue'
                marker = 'o'
                size = 100
            else:
                color = 'gray'
                marker = 'o'
                size = 50
                
            ax.scatter(x, y, c=color, marker=marker, s=size)
            
            # 訪問回数を表示
            if node.visit_count > 0:
                ax.text(x, y-0.3, str(node.visit_count), 
                       fontsize=8, ha='center')
                
        # メッセージの描画
        for node in self.nodes.values():
            for target_pos, message_value in node.messages_out.items():
                if target_pos in self.nodes:
                    x1, y1 = node.position[1], -node.position[0]
                    x2, y2 = target_pos[1], -target_pos[0]
                    
                    # メッセージの強さで色を変える
                    if message_value > 0:
                        color = 'green'
                        alpha = min(message_value / 10.0, 1.0)
                    else:
                        color = 'red'
                        alpha = min(-message_value / 10.0, 1.0)
                        
                    ax.arrow(x1, y1, x2-x1, y2-y1, 
                           color=color, alpha=alpha, 
                           head_width=0.1, head_length=0.1)
                           
        ax.set_title('Knowledge Graph with Message Passing')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('message_passing_graph.png', dpi=150)
        plt.close()


def demonstrate_message_passing():
    """メッセージパッシングナビゲーションのデモ"""
    print("メッセージパッシングによる意思決定")
    print("=" * 60)
    print("重要な概念：")
    print("- 行き止まりの記憶を分岐点に伝播")
    print("- ゴールの情報も分岐点に伝播")
    print("- メッセージパッシングで知識を共有")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    config.exploration_epsilon = 0.1  # 探索率を設定
    
    # テスト実行
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')  # より小さい迷路でテスト
    navigator = MessagePassingNavigator(config)
    
    print(f"\n迷路: {maze.size}")
    print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
    print("-" * 40)
    
    obs = maze.reset()
    steps = 0
    visited_positions = set()
    
    for _ in range(300):
        old_pos = obs.position
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        steps += 1
        
        visited_positions.add(obs.position)
        
        # 定期的な進捗表示
        if steps % 50 == 0:
            print(f"\nステップ {steps}: 訪問位置数 {len(visited_positions)}")
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            break
            
    # 知識グラフの可視化
    navigator.visualize_knowledge_graph()
    
    # 統計情報
    print("\n" + "=" * 60)
    print("探索統計：")
    print(f"訪問ノード数: {len(navigator.nodes)}")
    
    junctions = [n for n in navigator.nodes.values() if n.is_junction]
    dead_ends = [n for n in navigator.nodes.values() if n.is_dead_end]
    
    print(f"分岐点: {len(junctions)}")
    print(f"行き止まり: {len(dead_ends)}")
    
    # 行き止まり情報の伝播状況
    print("\n行き止まり記憶の活用：")
    for junction in junctions:
        dead_end_memories = [m for m in junction.path_memories if m.is_dead_end]
        if dead_end_memories:
            print(f"分岐点{junction.position}: {len(dead_end_memories)}個の行き止まり記憶")
            for mem in dead_end_memories:
                print(f"  - 行動{['上','右','下','左'][mem.action]} → 行き止まり{mem.destination}")
                
    print("\n" + "=" * 60)
    print("✨ メッセージパッシングにより分岐点で適切な判断")
    print("✨ 行き止まりの記憶が次の探索を効率化")
    print("✨ 分散的な知識共有による集合知の形成")


if __name__ == "__main__":
    demonstrate_message_passing()