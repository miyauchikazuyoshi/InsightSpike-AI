#!/usr/bin/env python3
"""
Maze Agent Wrapper
==================

MainAgentを活用した迷路解法エージェントのラッパー実装。
Wake Modeと5次元ベクトルを使用して効率的に迷路を探索。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.config import load_config
from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.algorithms.gedig_wake_mode import ProcessingMode
from insightspike.environments.maze import SimpleMaze

logger = logging.getLogger(__name__)


@dataclass
class MazeState:
    """迷路の現在状態"""
    position: Tuple[int, int]
    visited_positions: List[Tuple[int, int]]
    last_action: Optional[int] = None
    last_result: Optional[str] = None
    step_count: int = 0


class VectorAdapter:
    """384次元と5次元ベクトルの相互変換"""
    
    def __init__(self, maze_size: Tuple[int, int]):
        self.maze_width, self.maze_height = maze_size
        
    def to_compact(self, position: Tuple[int, int], 
                   action: Optional[int] = None,
                   result: Optional[str] = None,
                   visits: int = 0) -> np.ndarray:
        """5次元コンパクトベクトルに変換"""
        # 位置を正規化
        norm_x = position[0] / (self.maze_width - 1) if self.maze_width > 1 else 0
        norm_y = position[1] / (self.maze_height - 1) if self.maze_height > 1 else 0
        
        # 行動を正規化（0-3 → 0-1）
        norm_action = (action / 3.0) if action is not None else 0.5
        
        # 結果をエンコード
        result_map = {'wall': -1.0, 'empty': 0.0, 'goal': 1.0, None: 0.0}
        norm_result = result_map.get(result, 0.0)
        
        # 訪問回数を正規化（上限10回）
        norm_visits = min(visits / 10.0, 1.0)
        
        return np.array([norm_x, norm_y, norm_action, norm_result, norm_visits])
    
    def to_query(self, state: MazeState) -> str:
        """状態を自然言語クエリに変換"""
        x, y = state.position
        
        # 基本的な状態説明
        query = f"現在位置は({x}, {y})です。"
        
        # 最後の行動結果
        if state.last_action is not None:
            actions = ['上', '右', '下', '左']
            query += f" 直前に{actions[state.last_action]}に移動しました。"
            
        if state.last_result:
            if state.last_result == 'wall':
                query += " 壁にぶつかりました。"
            elif state.last_result == 'goal':
                query += " ゴールに到達しました！"
                
        # 訪問情報
        visit_count = state.visited_positions.count(state.position)
        if visit_count > 1:
            query += f" この位置は{visit_count}回目の訪問です。"
            
        query += " 次はどの方向に進むべきですか？"
        
        return query


class MazeVisualizer:
    """迷路と知識グラフのビジュアライゼーション"""
    
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze = maze
        self.start = start
        self.goal = goal
        
        # Figure設定
        self.fig, (self.ax_maze, self.ax_graph) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('Maze Agent Integration: Wake Mode Visualization', fontsize=16)
        
        # 迷路パネルの初期化
        self.ax_maze.set_title('Maze Exploration')
        self.ax_maze.set_aspect('equal')
        
        # グラフパネルの初期化
        self.ax_graph.set_title('Knowledge Graph Growth')
        
        # 履歴
        self.path_history = []
        self.graph_history = []
        
    def update(self, state: MazeState, knowledge_graph: nx.Graph, gedig_values: Dict):
        """表示を更新"""
        # 迷路の描画
        self._draw_maze(state)
        
        # 知識グラフの描画
        self._draw_graph(knowledge_graph, gedig_values)
        
        # 履歴に追加
        self.path_history.append(state.position)
        self.graph_history.append(knowledge_graph.copy())
        
        plt.pause(0.1)
        
    def _draw_maze(self, state: MazeState):
        """迷路パネルを描画"""
        self.ax_maze.clear()
        
        # 迷路を描画
        maze_display = np.ones_like(self.maze, dtype=float)
        maze_display[self.maze == 1] = 0  # 壁は黒
        
        # 訪問済み位置をマーク
        for pos in set(state.visited_positions):
            if pos != state.position:
                maze_display[pos[1], pos[0]] = 0.7  # 訪問済みは灰色
                
        self.ax_maze.imshow(maze_display, cmap='gray', vmin=0, vmax=1)
        
        # スタートとゴール
        self.ax_maze.plot(self.start[0], self.start[1], 'go', markersize=15, label='Start')
        self.ax_maze.plot(self.goal[0], self.goal[1], 'r*', markersize=20, label='Goal')
        
        # 現在位置
        self.ax_maze.plot(state.position[0], state.position[1], 'bo', 
                         markersize=12, label='Current')
        
        # 経路を描画
        if len(state.visited_positions) > 1:
            path = np.array(state.visited_positions)
            self.ax_maze.plot(path[:, 0], path[:, 1], 'g-', linewidth=2, alpha=0.5)
            
        self.ax_maze.set_title(f'Maze Exploration (Step {state.step_count})')
        self.ax_maze.legend()
        self.ax_maze.grid(True, alpha=0.3)
        
    def _draw_graph(self, graph: nx.Graph, gedig_values: Dict):
        """知識グラフパネルを描画"""
        self.ax_graph.clear()
        
        if len(graph.nodes()) == 0:
            self.ax_graph.text(0.5, 0.5, 'No episodes yet', 
                              ha='center', va='center', transform=self.ax_graph.transAxes)
            return
            
        # レイアウト計算
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # ノードの色をgeDIG値に基づいて設定
        node_colors = []
        for node in graph.nodes():
            gedig = gedig_values.get(node, 0.0)
            node_colors.append(gedig)
            
        # グラフを描画
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              cmap='coolwarm', node_size=300,
                              ax=self.ax_graph)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=self.ax_graph)
        
        # ノードラベル（簡略化）
        labels = {}
        for node in graph.nodes():
            if hasattr(graph.nodes[node], 'position'):
                pos_data = graph.nodes[node]['position']
                labels[node] = f"{pos_data}"
            else:
                labels[node] = str(node)[:8]
                
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=self.ax_graph)
        
        self.ax_graph.set_title(f'Knowledge Graph ({len(graph.nodes())} nodes)')
        self.ax_graph.axis('off')
        
    def save_animation(self, filename: str = 'maze_solving.gif'):
        """アニメーションを保存"""
        # TODO: matplotlib.animationを使用してGIF生成
        logger.info(f"Animation would be saved to {filename}")


class MazeAgentWrapper:
    """MainAgentを使った迷路解法ラッパー"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 設定読み込み
        if config_path:
            self.config = load_config(config_path)
        else:
            # デフォルトでexperimentプリセットを使用
            self.config = ConfigPresets.experiment()
            # MockProviderを使用してテスト
            self.config.llm.provider = 'mock'
        
        # Wake Mode設定
        self.config.wake_sleep.mode = "wake"
        
        # MainAgent作成
        self.main_agent = MainAgent(config=self.config)
        if not self.main_agent.initialize():
            raise RuntimeError("Failed to initialize MainAgent")
        
        # コンポーネント初期化
        self.vector_adapter = None
        self.visualizer = None
        self.maze_env = None
        
        # 状態
        self.current_state = None
        self.knowledge_graph = nx.Graph()
        self.gedig_values = {}
        
    def solve_maze(self, maze_query: Dict[str, Any], visualize: bool = True):
        """迷路を解く"""
        # 迷路環境をセットアップ
        self._setup_maze(maze_query)
        
        # ビジュアライザー初期化
        if visualize:
            self.visualizer = MazeVisualizer(
                self.maze_env.grid,  # mazeではなくgrid
                self.maze_env.start_pos,
                self.maze_env.goal_pos
            )
            
        # 初期知識を追加
        self._add_initial_knowledge()
        
        # メインループ
        max_steps = 20  # 小さな迷路なので20ステップに制限
        
        while self.current_state.step_count < max_steps:
            print(f"\nStep {self.current_state.step_count}: Position {self.current_state.position}")
            
            # 現在状態からクエリ生成
            query = self.vector_adapter.to_query(self.current_state)
            print(f"Query: {query}")
            
            # MainAgentに問い合わせ
            result = self.main_agent.process_question(query, max_cycles=1)
            
            # 行動決定と実行
            action = self._extract_action(result)
            print(f"Action selected: {action}")
            self._execute_action(action)
            
            # 知識グラフ更新
            self._update_knowledge_graph(result)
            
            # ビジュアライゼーション更新
            if visualize:
                self.visualizer.update(
                    self.current_state,
                    self.knowledge_graph,
                    self.gedig_values
                )
                
            # ゴール判定
            if self.current_state.position == self.maze_env.goal_pos:
                print(f"Goal reached in {self.current_state.step_count} steps!")
                break
                
        # 結果を返す
        return {
            'success': self.current_state.position == self.maze_env.goal_pos,
            'steps': self.current_state.step_count,
            'path': self.current_state.visited_positions,
            'unique_positions': len(set(self.current_state.visited_positions))
        }
        
    def _setup_maze(self, maze_query: Dict):
        """迷路環境をセットアップ"""
        # カスタム迷路の場合は直接設定
        maze_array = np.array(maze_query['maze'])
        
        # SimpleMazeにカスタム迷路を渡す
        self.maze_env = SimpleMaze(
            size=maze_array.shape[::-1],  # (width, height)
            maze_type='custom',
            maze_layout=maze_array  # 直接迷路配列を渡す
        )
        self.maze_env.start_pos = maze_query['start']
        self.maze_env.goal_pos = maze_query['goal']
        self.maze_env.agent_pos = maze_query['start']
        
        # アダプター初期化
        self.vector_adapter = VectorAdapter(self.maze_env.size)
        
        # 初期状態
        self.current_state = MazeState(
            position=self.maze_env.start_pos,
            visited_positions=[self.maze_env.start_pos]
        )
        
    def _add_initial_knowledge(self):
        """初期知識を追加"""
        # 迷路の基本知識
        self.main_agent.add_knowledge("迷路では壁を避けて通路を進む必要があります。")
        self.main_agent.add_knowledge("同じ場所を何度も訪問するのは非効率です。")
        self.main_agent.add_knowledge("ゴールに向かって進むことが重要です。")
        
        # ゴール位置の知識
        goal_x, goal_y = self.maze_env.goal_pos
        self.main_agent.add_knowledge(f"ゴールは位置({goal_x}, {goal_y})にあります。")
        
    def _extract_action(self, result) -> int:
        """MainAgentの結果から行動を抽出"""
        response = result.response if hasattr(result, 'response') else str(result)
        
        # MockProviderの場合はゴールに向かう簡単な戦略を使用
        if 'mock' in response.lower():
            # 現在位置とゴールの差分を計算
            current_x, current_y = self.current_state.position
            goal_x, goal_y = self.maze_env.goal_pos
            
            # 未訪問の方向を優先しつつゴールに向かう
            possible_actions = self._get_possible_actions()
            if not possible_actions:
                return 0
                
            # 各行動の評価スコア
            action_scores = []
            for action in possible_actions:
                dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
                new_x, new_y = current_x + dx, current_y + dy
                
                # ゴールまでのマンハッタン距離
                dist_to_goal = abs(new_x - goal_x) + abs(new_y - goal_y)
                
                # 訪問回数
                visit_count = self.current_state.visited_positions.count((new_x, new_y))
                
                # スコア（距離が小さいほど、訪問回数が少ないほど良い）
                score = -dist_to_goal - visit_count * 2
                action_scores.append(score)
                
            # 最高スコアのアクションを選択
            best_idx = np.argmax(action_scores)
            return possible_actions[best_idx]
        
        # 簡単なキーワードマッチング
        if '上' in response or 'up' in response.lower():
            return 0
        elif '右' in response or 'right' in response.lower():
            return 1
        elif '下' in response or 'down' in response.lower():
            return 2
        elif '左' in response or 'left' in response.lower():
            return 3
        else:
            # ランダムに選択
            possible_actions = self._get_possible_actions()
            return np.random.choice(possible_actions) if possible_actions else 0
            
    def _get_possible_actions(self) -> List[int]:
        """可能な行動のリスト"""
        actions = []
        x, y = self.current_state.position
        
        # 上下左右をチェック
        for action, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.maze_env.size[0] and 
                0 <= ny < self.maze_env.size[1] and
                self.maze_env.grid[ny, nx] == 0):
                actions.append(action)
                
        return actions
        
    def _execute_action(self, action: int):
        """行動を実行"""
        old_pos = self.current_state.position
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
        
        # 境界チェック
        if (0 <= new_x < self.maze_env.size[0] and 
            0 <= new_y < self.maze_env.size[1]):
            
            if self.maze_env.grid[new_y, new_x] == 0:  # 通路
                # 移動成功
                self.current_state.position = (new_x, new_y)
                self.current_state.last_result = 'goal' if (new_x, new_y) == self.maze_env.goal_pos else 'empty'
            else:
                # 壁
                self.current_state.last_result = 'wall'
        else:
            # 境界外
            self.current_state.last_result = 'wall'
            
        self.current_state.last_action = action
        self.current_state.visited_positions.append(self.current_state.position)
        self.current_state.step_count += 1
        
    def _update_knowledge_graph(self, result):
        """知識グラフを更新"""
        # エピソードノードを追加
        node_id = f"ep_{self.current_state.step_count}"
        self.knowledge_graph.add_node(
            node_id,
            position=self.current_state.position,
            action=self.current_state.last_action,
            result=self.current_state.last_result
        )
        
        # 前のノードとエッジを追加
        if self.current_state.step_count > 1:
            prev_node = f"ep_{self.current_state.step_count - 1}"
            self.knowledge_graph.add_edge(prev_node, node_id)
            
        # geDIG値を記録（仮の値）
        if hasattr(result, 'gedig_value'):
            self.gedig_values[node_id] = result.gedig_value
        else:
            # Wake Modeなので低い値を想定
            self.gedig_values[node_id] = 0.1 + np.random.random() * 0.3


def main():
    """メイン実行関数"""
    # テスト用の迷路（3x3の簡単なもの）
    test_maze = {
        "type": "maze",
        "maze": [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        "start": (0, 0),
        "goal": (2, 2)
    }
    
    # MazeAgentWrapperを作成
    agent = MazeAgentWrapper()
    
    # 迷路を解く
    print("Starting maze solving...")
    result = agent.solve_maze(test_maze, visualize=False)
    
    # 結果を表示
    print("\n=== Maze Solving Result ===")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Unique positions visited: {result['unique_positions']}")
    print(f"Efficiency: {result['unique_positions'] / result['steps']:.2%}")
    
    plt.show()


if __name__ == "__main__":
    main()