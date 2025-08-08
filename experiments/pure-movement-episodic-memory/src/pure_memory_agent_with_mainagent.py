#!/usr/bin/env python3
"""
純粋記憶エージェント - MainAgent統合版
add_knowledgeを適切に使用してDataStoreと連携
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.episode import Episode


class PureMemoryAgentWithMainAgent(MainAgent):
    """MainAgentを継承した純粋記憶エージェント"""
    
    def __init__(self, maze: np.ndarray, datastore_path: str = "data/maze_memory", 
                 config: Optional[Dict] = None):
        """
        Args:
            maze: 迷路配列
            datastore_path: DataStore保存先
            config: 設定
        """
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.height - 2, self.width - 2)
        
        # 行動定義
        self.actions = ['up', 'right', 'down', 'left']
        self.action_deltas = {
            'up': (-1, 0),
            'right': (0, 1),
            'down': (1, 0),
            'left': (0, -1)
        }
        
        # 統計
        self.steps = 0
        self.wall_hits = 0
        self.path = [self.position]
        self.visit_counts = defaultdict(int)
        
        # DataStore作成
        datastore = DataStoreFactory.create("filesystem", base_path=datastore_path)
        
        # MainAgent初期化（レガシー設定形式で）
        legacy_config = {
            'datastore': {'type': 'filesystem', 'base_path': datastore_path},
            'llm': {'provider': 'mock', 'model': 'mock'},  # LLM不要
            'layers': {
                'layer2': {
                    'vector_store_type': 'numpy',
                    'embedding_model': 'mock'
                },
                'layer3': {
                    'gedig_threshold': config.get('gedig_threshold', 0.5) if config else 0.5,
                    'max_edges_per_node': config.get('max_edges_per_node', 7) if config else 7
                },
                'layer4': {
                    'max_depth': config.get('max_depth', 20) if config else 20,
                    'search_k': config.get('search_k', 50) if config else 50
                }
            }
        }
        
        super().__init__(datastore=datastore, config=legacy_config)
        
        # 初期化メッセージ
        print(f"✅ MainAgent統合版エージェント初期化")
        print(f"  DataStore: {datastore_path}")
        print(f"  エッジ数: {config.get('max_edges_per_node', 7) if config else 7}")
        print(f"  最大深度: {config.get('max_depth', 20) if config else 20}")
    
    def _create_episode_text(self, observation_type: str, **kwargs) -> str:
        """観測をテキスト形式のエピソードに変換"""
        x, y = self.position
        
        if observation_type == 'visual':
            direction = kwargs['direction']
            is_wall = kwargs['is_wall']
            return (f"At position ({x},{y}), looking {direction}: "
                   f"{'wall' if is_wall else 'passage'}")
        
        elif observation_type == 'movement':
            action = kwargs['action']
            success = kwargs['success']
            new_pos = kwargs.get('new_pos', self.position)
            return (f"From ({x},{y}) moved {action} to {new_pos}: "
                   f"{'success' if success else 'failed (wall)'}")
        
        elif observation_type == 'goal_check':
            distance = abs(x - self.goal[0]) + abs(y - self.goal[1])
            return f"At ({x},{y}), distance to goal: {distance} steps"
        
        else:
            return f"At position ({x},{y})"
    
    def _add_visual_observations(self):
        """現在位置から4方向の視覚観測を追加"""
        x, y = self.position
        
        for direction in self.actions:
            dx, dy = self.action_deltas[direction]
            nx, ny = x + dx, y + dy
            
            is_wall = True
            if 0 <= nx < self.height and 0 <= ny < self.width:
                is_wall = (self.maze[nx, ny] == 1)
            
            # テキスト形式のエピソードを作成
            episode_text = self._create_episode_text(
                'visual',
                direction=direction,
                is_wall=is_wall
            )
            
            # MainAgentのadd_knowledgeを使用！
            self.add_knowledge(episode_text)
            
            # デバッグ出力（最初の数回のみ）
            if self.steps < 3:
                print(f"  👁️ 視覚: {episode_text}")
    
    def _add_movement_episode(self, action: str, success: bool, new_pos: Tuple[int, int]):
        """移動エピソードを追加"""
        episode_text = self._create_episode_text(
            'movement',
            action=action,
            success=success,
            new_pos=new_pos
        )
        
        # MainAgentのadd_knowledgeを使用！
        self.add_knowledge(episode_text)
        
        # デバッグ出力
        if self.steps < 3:
            print(f"  🚶 移動: {episode_text}")
    
    def _add_goal_check(self):
        """ゴールまでの距離チェックを追加"""
        episode_text = self._create_episode_text('goal_check')
        self.add_knowledge(episode_text)
    
    def get_action(self) -> str:
        """行動を決定（MainAgentの推論機能を活用）"""
        # 視覚観測を追加
        self._add_visual_observations()
        
        # ゴールチェック
        if self.steps % 10 == 0:
            self._add_goal_check()
        
        # 現在の状況を質問として構築
        x, y = self.position
        distance = abs(x - self.goal[0]) + abs(y - self.goal[1])
        
        question = (f"I am at position ({x},{y}). "
                   f"Goal is at ({self.goal[0]},{self.goal[1]}). "
                   f"Distance: {distance} steps. "
                   f"Which direction should I move? (up/right/down/left)")
        
        # MainAgentのprocess_questionを使用
        try:
            result = self.process_question(question)
            
            # CycleResultオブジェクトから応答を取得
            if hasattr(result, 'response'):
                response = result.response.lower()
            else:
                response = str(result).lower()
            
            # 応答から方向を抽出
            for action in self.actions:
                if action in response:
                    if self.steps < 3:
                        print(f"  💭 推論結果: {action}")
                    return action
            
            # 方向が見つからない場合はランダム
            return np.random.choice(self.actions)
            
        except Exception as e:
            # エラー時はランダム
            if self.steps < 3:
                print(f"  ⚠️ 推論エラー: {e}")
            return np.random.choice(self.actions)
    
    def execute_action(self, action: str) -> bool:
        """行動を実行"""
        dx, dy = self.action_deltas[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        success = False
        if (0 <= new_x < self.height and 
            0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            old_pos = self.position
            self.position = (new_x, new_y)
            success = True
            
            # 移動エピソードを追加
            self._add_movement_episode(action, True, self.position)
        else:
            self.wall_hits += 1
            # 壁衝突エピソードを追加
            self._add_movement_episode(action, False, self.position)
        
        self.steps += 1
        self.path.append(self.position)
        self.visit_counts[self.position] += 1
        
        return success
    
    def is_goal_reached(self) -> bool:
        """ゴール到達判定"""
        return self.position == self.goal
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        distance = abs(self.position[0] - self.goal[0]) + \
                  abs(self.position[1] - self.goal[1])
        
        # DataStoreから実際のエピソード数を取得
        episode_count = len(self.datastore.list_episodes())
        
        return {
            'steps': self.steps,
            'wall_hits': self.wall_hits,
            'wall_hit_rate': self.wall_hits / max(1, self.steps),
            'distance_to_goal': distance,
            'unique_visits': len(set(self.path)),
            'episode_count': episode_count,
            'datastore_path': self.datastore.storage_path
        }


def test_mainagent_integration():
    """MainAgent統合版のテスト"""
    print("="*70)
    print("🧪 MainAgent統合版エージェントのテスト")
    print("="*70)
    
    # 11×11の迷路生成
    from test_true_perfect_maze import generate_perfect_maze_dfs
    
    maze = generate_perfect_maze_dfs((11, 11), seed=42)
    
    print("\n迷路構造:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == 9 and j == 9:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # エージェント作成
    print("\n" + "="*70)
    print("エージェント初期化")
    print("="*70)
    
    agent = PureMemoryAgentWithMainAgent(
        maze=maze,
        datastore_path="data/maze_memory_mainagent",
        config={
            'max_depth': 20,
            'search_k': 50,
            'gedig_threshold': 0.5,
            'max_edges_per_node': 7  # マジカルナンバー
        }
    )
    
    # 実行
    print("\n" + "="*70)
    print("実行開始")
    print("="*70)
    
    for step in range(500):
        if agent.is_goal_reached():
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            break
        
        # 行動決定と実行
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # 進捗表示
        if step % 50 == 49:
            stats = agent.get_statistics()
            print(f"\nStep {step+1}:")
            print(f"  位置: {agent.position}")
            print(f"  距離: {stats['distance_to_goal']}")
            print(f"  エピソード数: {stats['episode_count']}")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
    else:
        print(f"\n⏰ {step+1}ステップで終了")
    
    # 最終統計
    stats = agent.get_statistics()
    
    print("\n" + "="*70)
    print("📊 最終結果")
    print("="*70)
    
    print(f"\nゴール到達: {'✅' if agent.is_goal_reached() else '❌'}")
    print(f"総ステップ: {stats['steps']}")
    print(f"壁衝突率: {stats['wall_hit_rate']:.1%}")
    print(f"ユニーク訪問: {stats['unique_visits']}")
    print(f"エピソード数: {stats['episode_count']}")
    print(f"DataStore: {stats['datastore_path']}")
    
    print("\n✨ MainAgentのadd_knowledgeを使用してDataStoreに永続化！")


if __name__ == "__main__":
    test_mainagent_integration()