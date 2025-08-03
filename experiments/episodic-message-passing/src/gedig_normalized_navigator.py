#!/usr/bin/env python3
"""正規化版geDIGナビゲーター：メインコードとの統合用"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig
from insightspike.algorithms.gedig_core import geDIG
from insightspike.core.base.datastore import DataStore
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore


@dataclass
class NormalizedGedigEpisode:
    """正規化されたベクトルを持つエピソード"""
    episode_type: str
    content: Dict
    raw_vector: np.ndarray      # 生のベクトル
    normalized_vector: np.ndarray  # 正規化されたベクトル
    node_id: int
    
    # geDIG関連
    ged_delta: float = 0.0
    ig_delta: float = 0.0
    gedig_value: float = 0.0
    
    # 訪問情報
    visit_count: int = 0
    timestamp: int = 0


class NormalizedGedigNavigator:
    """メインコードと統合可能な正規化版ナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig, maze_size: int = 20):
        self.config = config
        self.maze_size = maze_size
        self.episodes: List[NormalizedGedigEpisode] = []
        self.episode_counter = 0
        self.current_step = 0
        
        # 正規化パラメータ
        self.max_visits_expected = 200  # 期待される最大訪問回数
        
        # geDIGコアの初期化（メインコードから）
        self.gedig_core = geDIG(
            gamma=0.99,
            alpha=0.1,
            epsilon=1e-10,
            similarity_threshold=0.8
        )
        
        # データストア（メインコードとの互換性）
        self.datastore = FileSystemDataStore(base_path="experiments/episodic-message-passing/data/normalized")
        
        # 統計
        self.position_visit_counts = {}
        self.transition_counts = {}
        
    def normalize_vector(self, raw_vector: np.ndarray, episode_type: str) -> np.ndarray:
        """ベクトルを[-1, 1]の範囲に正規化"""
        if episode_type == "goal_info":
            # ゴール情報の正規化
            normalized = np.zeros(10)  # 10次元に統一
            normalized[0] = (raw_vector[0] / self.maze_size) * 2 - 1  # X座標
            normalized[1] = (raw_vector[1] / self.maze_size) * 2 - 1  # Y座標
            normalized[2] = 1.0  # ゴールフラグ
            return normalized
            
        else:
            # 移動エピソードの正規化（8次元→10次元）
            normalized = np.zeros(10)
            
            # 座標の正規化 [-1, 1]
            normalized[0] = (raw_vector[0] / self.maze_size) * 2 - 1  # from_x
            normalized[1] = (raw_vector[1] / self.maze_size) * 2 - 1  # from_y
            normalized[2] = (raw_vector[2] / self.maze_size) * 2 - 1  # to_x
            normalized[3] = (raw_vector[3] / self.maze_size) * 2 - 1  # to_y
            
            # 結果（既に正規化済み）
            normalized[4] = raw_vector[4]  # result_val: -1.0 or 1.0
            
            # 行動方向の正規化（one-hot的な表現）
            action = int(raw_vector[5])
            normalized[5] = -1.0 + (action / 1.5)  # [0,3] → [-1, 1]
            
            # 訪問回数の正規化（シグモイド的）
            from_visits_log = raw_vector[6]
            to_visits_log = raw_vector[7]
            normalized[6] = np.tanh(from_visits_log / 3.0)  # tanh で [-1, 1] に
            normalized[7] = np.tanh(to_visits_log / 3.0)
            
            # 追加的な特徴量
            # 移動距離（マンハッタン距離）
            distance = abs(raw_vector[2] - raw_vector[0]) + abs(raw_vector[3] - raw_vector[1])
            normalized[8] = np.tanh(distance / 5.0)
            
            # 時間的特徴（エピソード番号の正規化）
            normalized[9] = np.tanh(self.episode_counter / 1000.0)
            
            return normalized
            
    def calculate_gedig_using_core(self, episode: NormalizedGedigEpisode, 
                                  recent_episodes: List[NormalizedGedigEpisode]) -> Tuple[float, float, float]:
        """メインコードのgeDIG計算を使用"""
        if not recent_episodes:
            return 1.0, 1.0, 1.0
            
        # 類似エピソードとの比較
        max_similarity = 0.0
        for past_ep in recent_episodes:
            # コサイン類似度
            similarity = np.dot(episode.normalized_vector, past_ep.normalized_vector) / (
                np.linalg.norm(episode.normalized_vector) * np.linalg.norm(past_ep.normalized_vector) + 1e-10
            )
            max_similarity = max(max_similarity, similarity)
            
        # 新規性 = 1 - 類似度
        ged_delta = 1.0 - max_similarity
        
        # 情報利得（簡易版）
        if episode.episode_type == "movement":
            from_pos = tuple(episode.content['from'])
            visit_count = self.position_visit_counts.get(from_pos, 0)
            ig_delta = 1.0 / (1.0 + math.log(1 + visit_count))
        else:
            ig_delta = 1.0
            
        # geDIG値
        gedig_value = ged_delta * ig_delta
        
        return ged_delta, ig_delta, gedig_value
        
    def add_episode_normalized(self, episode_type: str, content: Dict) -> NormalizedGedigEpisode:
        """正規化されたエピソードを追加"""
        # 生のベクトル作成
        if episode_type == "goal_info":
            pos = content['position']
            raw_vector = np.array([pos[0], pos[1], 100.0])
        else:
            from_pos = content['from']
            to_pos = content['to']
            result_val = 1.0 if content['result'] == "成功" else -1.0
            
            from_visits = self.position_visit_counts.get(tuple(from_pos), 0)
            to_visits = self.position_visit_counts.get(tuple(to_pos), 0)
            
            raw_vector = np.array([
                from_pos[0], from_pos[1],
                to_pos[0], to_pos[1],
                result_val,
                content['action'],
                math.log(1 + from_visits),
                math.log(1 + to_visits)
            ])
            
        # 正規化
        normalized_vector = self.normalize_vector(raw_vector, episode_type)
        
        # エピソード作成
        episode = NormalizedGedigEpisode(
            episode_type=episode_type,
            content=content,
            raw_vector=raw_vector,
            normalized_vector=normalized_vector,
            node_id=self.episode_counter,
            timestamp=self.current_step
        )
        
        # geDIG計算（最近のエピソードと比較）
        recent_episodes = self.episodes[-50:] if len(self.episodes) > 50 else self.episodes
        ged_delta, ig_delta, gedig_value = self.calculate_gedig_using_core(episode, recent_episodes)
        
        episode.ged_delta = ged_delta
        episode.ig_delta = ig_delta
        episode.gedig_value = gedig_value
        
        # エピソード追加
        self.episodes.append(episode)
        
        # DataStoreに保存（メインコードとの互換性）
        self.datastore.add_episode({
            'id': episode.node_id,
            'type': episode_type,
            'content': content,
            'vector': normalized_vector.tolist(),
            'gedig_value': gedig_value,
            'timestamp': self.current_step
        })
        
        # 統計更新
        if episode_type == "movement":
            from_pos = tuple(content['from'])
            to_pos = tuple(content['to'])
            
            self.position_visit_counts[from_pos] = self.position_visit_counts.get(from_pos, 0) + 1
            if content['result'] == "成功":
                self.position_visit_counts[to_pos] = self.position_visit_counts.get(to_pos, 0) + 1
                
        self.episode_counter += 1
        self.current_step += 1
        
        return episode
        
    def decide_action_normalized(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """正規化ベクトルを使った行動決定"""
        # 現在位置の正規化座標
        norm_x = (current_pos[0] / self.maze_size) * 2 - 1
        norm_y = (current_pos[1] / self.maze_size) * 2 - 1
        
        # 各行動の期待値を計算
        action_values = {}
        
        for action in possible_actions:
            # この行動を取った場合の仮想ベクトル
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # 仮想エピソードのベクトル
            visit_count = self.position_visit_counts.get(current_pos, 0)
            test_vector = self.normalize_vector(np.array([
                current_pos[0], current_pos[1],
                next_pos[0], next_pos[1],
                1.0,  # 成功を仮定
                action,
                math.log(1 + visit_count),
                math.log(1 + self.position_visit_counts.get(next_pos, 0))
            ]), "movement")
            
            # 既存エピソードとの類似度から価値を推定
            if self.episodes:
                similarities = []
                values = []
                for ep in self.episodes[-20:]:  # 最近20エピソード
                    if ep.episode_type == "movement":
                        sim = np.dot(test_vector, ep.normalized_vector) / (
                            np.linalg.norm(test_vector) * np.linalg.norm(ep.normalized_vector) + 1e-10
                        )
                        similarities.append(sim)
                        values.append(ep.gedig_value)
                        
                if similarities:
                    # 類似度加重平均
                    weights = np.array(similarities)
                    weights = np.exp(weights * 5)  # 温度パラメータ
                    weights /= weights.sum()
                    expected_value = np.dot(weights, values)
                else:
                    expected_value = 1.0
            else:
                expected_value = 1.0
                
            # 未探索ボーナス
            if visit_count == 0:
                expected_value *= 2.0
                
            action_values[action] = expected_value
            
        # ソフトマックス選択
        if action_values:
            values = np.array(list(action_values.values()))
            exp_values = np.exp(values / 0.2)  # 温度0.2
            probs = exp_values / exp_values.sum()
            
            return list(action_values.keys())[np.random.choice(len(probs), p=probs)]
        else:
            return np.random.choice(possible_actions)


def test_normalized_integration():
    """正規化版の統合テスト"""
    print("正規化版geDIGナビゲーター - メインコード統合テスト")
    print("=" * 60)
    
    config = MazeNavigatorConfig()
    navigator = NormalizedGedigNavigator(config, maze_size=10)
    
    # 小さい迷路でテスト
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    
    print(f"迷路サイズ: {maze.size}")
    print(f"ベクトル次元: 10（正規化済み）")
    print(f"値域: [-1, 1]")
    
    # ゴール情報追加
    episode = navigator.add_episode_normalized("goal_info", {"position": maze.goal_pos})
    print(f"\nゴールベクトル（正規化済み）:")
    print(f"  生: {episode.raw_vector}")
    print(f"  正規化: {episode.normalized_vector}")
    
    # いくつかステップ実行
    obs = maze.reset()
    for step in range(20):
        current_pos = obs.position
        action = navigator.decide_action_normalized(current_pos, obs.possible_moves)
        
        old_pos = current_pos
        obs, _, done, _ = maze.step(action)
        new_pos = obs.position
        
        result = "壁" if old_pos == new_pos else ("行き止まり" if obs.is_dead_end else "成功")
        
        episode = navigator.add_episode_normalized("movement", {
            "from": old_pos,
            "to": new_pos,
            "action": action,
            "result": result
        })
        
        if step < 3:
            print(f"\nステップ {step+1}:")
            print(f"  移動: {old_pos} → {new_pos} ({result})")
            print(f"  正規化ベクトル: {episode.normalized_vector}")
            print(f"  geDIG値: {episode.gedig_value:.3f}")
            
        if done:
            print(f"\nゴール到達！ステップ数: {step+1}")
            break
            
    # DataStore確認
    print(f"\nDataStore保存エピソード数: {len(navigator.datastore.episodes)}")
    print("✅ メインコードとの統合準備完了")


if __name__ == "__main__":
    test_normalized_integration()