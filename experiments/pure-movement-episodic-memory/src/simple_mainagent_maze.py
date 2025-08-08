#!/usr/bin/env python3
"""
シンプルなMainAgent利用版
DataStoreに直接エピソードを保存
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.episode import Episode


class SimpleMainAgentMaze:
    """DataStoreを使うシンプルな迷路エージェント"""
    
    def __init__(self, maze: np.ndarray, datastore_path: str = "data/maze_episodes"):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.height - 2, self.width - 2)
        
        # 行動定義
        self.actions = ['up', 'right', 'down', 'left']
        self.action_deltas = {
            'up': (-1, 0), 'right': (0, 1),
            'down': (1, 0), 'left': (0, -1)
        }
        
        # DataStore作成（FileSystemDataStore）
        self.datastore = DataStoreFactory.create("filesystem", base_path=datastore_path)
        
        # エピソードリスト（メモリ内キャッシュ）
        self.episodes = []
        
        # 統計
        self.steps = 0
        self.wall_hits = 0
        
        print(f"✅ DataStore初期化: {datastore_path}")
        print(f"  既存エピソード数: {len(self.datastore.list_episodes())}")
    
    def add_episode(self, text: str, metadata: Dict = None):
        """エピソードをDataStoreに保存"""
        # エピソード作成
        episode = Episode(
            text=text,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # DataStoreに保存
        episode_id = self.datastore.store_episode(episode)
        
        # メモリ内キャッシュにも追加
        self.episodes.append({
            'id': episode_id,
            'text': text,
            'metadata': metadata
        })
        
        return episode_id
    
    def add_visual_observations(self):
        """視覚観測をエピソードとして保存"""
        x, y = self.position
        
        for direction in self.actions:
            dx, dy = self.action_deltas[direction]
            nx, ny = x + dx, y + dy
            
            is_wall = True
            if 0 <= nx < self.height and 0 <= ny < self.width:
                is_wall = (self.maze[nx, ny] == 1)
            
            # エピソードテキスト
            text = f"At ({x},{y}) looking {direction}: {'wall' if is_wall else 'passage'}"
            
            # メタデータ
            metadata = {
                'type': 'visual',
                'position': [x, y],
                'direction': direction,
                'is_wall': is_wall,
                'step': self.steps
            }
            
            # DataStoreに保存
            self.add_episode(text, metadata)
    
    def add_movement_episode(self, action: str, success: bool):
        """移動エピソードを保存"""
        x, y = self.position
        
        text = f"From ({x},{y}) moved {action}: {'success' if success else 'hit wall'}"
        
        metadata = {
            'type': 'movement',
            'position': [x, y],
            'action': action,
            'success': success,
            'step': self.steps
        }
        
        self.add_episode(text, metadata)
    
    def get_similar_episodes(self, query: str, k: int = 10) -> List[Dict]:
        """類似エピソードを検索（簡易版）"""
        # 現在位置を含むエピソードを優先
        x, y = self.position
        current_pos_str = f"({x},{y})"
        
        relevant_episodes = []
        
        # DataStoreから全エピソードを取得
        for episode_id in self.datastore.list_episodes()[-100:]:  # 最新100件
            episode = self.datastore.get_episode(episode_id)
            if episode:
                # 位置情報でフィルタ
                if current_pos_str in episode.text:
                    relevant_episodes.append({
                        'id': episode_id,
                        'text': episode.text,
                        'metadata': episode.metadata
                    })
        
        return relevant_episodes[:k]
    
    def get_action(self) -> str:
        """行動決定（エピソード記憶に基づく）"""
        # 視覚観測を追加
        self.add_visual_observations()
        
        # 類似エピソードを検索
        x, y = self.position
        query = f"At ({x},{y})"
        similar = self.get_similar_episodes(query)
        
        # 成功した移動エピソードを探す
        successful_actions = []
        for ep in similar:
            if ep['metadata'].get('type') == 'movement' and ep['metadata'].get('success'):
                action = ep['metadata'].get('action')
                if action:
                    successful_actions.append(action)
        
        # 壁がない方向を探す
        safe_directions = []
        for ep in similar:
            if ep['metadata'].get('type') == 'visual' and not ep['metadata'].get('is_wall'):
                direction = ep['metadata'].get('direction')
                if direction:
                    safe_directions.append(direction)
        
        # 優先順位：成功した行動 > 壁がない方向 > ランダム
        if successful_actions:
            return successful_actions[0]
        elif safe_directions:
            # ゴール方向を優先
            goal_dx = self.goal[0] - x
            goal_dy = self.goal[1] - y
            
            best_action = None
            best_score = -999
            
            for direction in safe_directions:
                dx, dy = self.action_deltas[direction]
                score = dx * np.sign(goal_dx) + dy * np.sign(goal_dy)
                if score > best_score:
                    best_score = score
                    best_action = direction
            
            return best_action if best_action else np.random.choice(safe_directions)
        else:
            return np.random.choice(self.actions)
    
    def execute_action(self, action: str) -> bool:
        """行動実行"""
        dx, dy = self.action_deltas[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        success = False
        if (0 <= new_x < self.height and 
            0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            self.position = (new_x, new_y)
            success = True
        else:
            self.wall_hits += 1
        
        # 移動エピソードを保存
        self.add_movement_episode(action, success)
        
        self.steps += 1
        return success
    
    def is_goal_reached(self) -> bool:
        return self.position == self.goal


def test_simple_mainagent():
    """シンプル版のテスト"""
    print("="*70)
    print("🧪 DataStore統合版エージェントのテスト")
    print("="*70)
    
    # 簡単な迷路
    maze = np.array([
        [1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,1],
        [1,0,1,0,1,0,1,0,1],
        [1,0,1,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1]
    ])
    
    print("\n迷路構造:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == 7 and j == 7:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # エージェント作成
    agent = SimpleMainAgentMaze(maze, "data/simple_maze_episodes")
    
    print("\n実行開始...")
    print("-" * 70)
    
    for step in range(200):
        if agent.is_goal_reached():
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            break
        
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # 進捗表示
        if step < 5 or step % 20 == 19:
            x, y = agent.position
            distance = abs(x - agent.goal[0]) + abs(y - agent.goal[1])
            print(f"Step {step+1}: 位置({x},{y}), 距離{distance}, "
                  f"行動={action}, {'成功' if success else '壁'}")
    else:
        print(f"\n⏰ タイムアウト")
    
    # 最終統計
    print("\n" + "="*70)
    print("📊 最終結果")
    print("="*70)
    
    total_episodes = len(agent.datastore.list_episodes())
    
    print(f"ゴール到達: {'✅' if agent.is_goal_reached() else '❌'}")
    print(f"総ステップ: {agent.steps}")
    print(f"壁衝突: {agent.wall_hits}")
    print(f"壁衝突率: {agent.wall_hits/max(1, agent.steps)*100:.1f}%")
    print(f"総エピソード数: {total_episodes}")
    print(f"DataStore保存先: {agent.datastore.storage_path}")
    
    # エピソード例を表示
    print("\n保存されたエピソード例（最新5件）:")
    for episode_id in agent.datastore.list_episodes()[-5:]:
        episode = agent.datastore.get_episode(episode_id)
        if episode:
            print(f"  - {episode.text}")
    
    print("\n✨ DataStoreに全エピソードが永続化されました！")


if __name__ == "__main__":
    test_simple_mainagent()