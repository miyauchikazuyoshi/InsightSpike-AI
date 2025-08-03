#!/usr/bin/env python3
"""
True Insight Vector Agent
=========================

エピソード記憶を混合して真の洞察ベクトルを生成する実装
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class TrueInsightVectorAgent:
    """洞察ベクトルを生成する迷路エージェント"""
    
    def __init__(self, vector_dim: int = 5):
        self.vector_dim = vector_dim
        self.episode_memory = []  # エピソード記憶
        self.initial_episodes = []  # 初期エピソード
        
    def generate_insight_vector(self, 
                              current_vector: np.ndarray,
                              episode_memory: List[Dict],
                              uncertainty: float = 0.5) -> np.ndarray:
        """
        Layer3的な洞察ベクトル生成
        
        エピソード記憶を混合して、現在の状況に対する
        「洞察」を表すベクトルを生成
        """
        
        # 1. 類似エピソードを検索（Layer2的処理）
        similar_episodes = self._find_similar_episodes(
            current_vector, 
            episode_memory, 
            k=5  # 上位5個
        )
        
        # 2. 重み付き混合で洞察ベクトルを生成（Layer3的処理）
        insight_vector = np.zeros(self.vector_dim)
        total_weight = 0.0
        
        for episode in similar_episodes:
            # 類似度に基づく重み
            similarity = episode['similarity']
            confidence = episode['confidence']
            weight = similarity * confidence * (1.0 - uncertainty)
            
            # エピソードベクトルを重み付き加算
            insight_vector += weight * episode['vector']
            total_weight += weight
        
        # 3. 正規化
        if total_weight > 0:
            insight_vector /= total_weight
        else:
            # 類似エピソードがない場合は現在ベクトルを使用
            insight_vector = current_vector.copy()
        
        # 4. 探索性の追加（uncertainty に基づく）
        exploration_noise = np.random.randn(self.vector_dim) * uncertainty * 0.1
        insight_vector += exploration_noise
        
        # 5. 方向成分の強調
        insight_vector = self._emphasize_direction_component(insight_vector)
        
        return insight_vector
    
    def _find_similar_episodes(self, 
                             query_vector: np.ndarray,
                             episode_memory: List[Dict],
                             k: int = 5) -> List[Dict]:
        """類似エピソードを検索"""
        similarities = []
        
        for episode in episode_memory:
            # コサイン類似度
            similarity = np.dot(query_vector, episode['vector']) / (
                np.linalg.norm(query_vector) * np.linalg.norm(episode['vector'])
            )
            
            similarities.append({
                'vector': episode['vector'],
                'similarity': similarity,
                'confidence': episode.get('confidence', 0.5),
                'result': episode.get('result', 'unknown')
            })
        
        # 類似度でソート
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:k]
    
    def _emphasize_direction_component(self, vector: np.ndarray) -> np.ndarray:
        """方向成分（3番目の要素）を強調"""
        emphasized = vector.copy()
        
        # 行動成分（index 2）の値を強調
        if self.vector_dim >= 3:
            action_value = emphasized[2]
            # 0.25刻みに量子化（4方向）
            quantized = round(action_value * 4) / 4
            emphasized[2] = quantized
        
        return emphasized
    
    def extract_action_from_insight(self, insight_vector: np.ndarray) -> int:
        """
        洞察ベクトルから行動を抽出
        
        本来のgeDIG理論では、洞察ベクトルの「方向成分」を
        読み取って行動を決定する
        """
        # 行動成分（3番目）を4方向にマッピング
        action_component = insight_vector[2]
        
        # 0-1の値を0-3の行動にマッピング
        action = int(action_component * 4) % 4
        
        # ゴール方向への補正
        if self.vector_dim >= 4 and insight_vector[3] > 0.8:  # result成分が高い
            # ゴールに近い可能性があるので、より慎重に
            # X, Y成分から方向を推定
            dx = insight_vector[0] - 0.5  # 中心からの差分
            dy = insight_vector[1] - 0.5
            
            if abs(dx) > abs(dy):
                action = 1 if dx > 0 else 3  # 右 or 左
            else:
                action = 2 if dy > 0 else 0  # 下 or 上
        
        return action


def demonstrate_insight_vector():
    """洞察ベクトル生成のデモンストレーション"""
    agent = TrueInsightVectorAgent()
    
    # サンプルエピソード記憶
    episode_memory = [
        {
            'vector': np.array([0.0, 0.0, 0.25, 0.0, 0.1]),  # 右に移動→空
            'confidence': 0.8,
            'result': 'empty'
        },
        {
            'vector': np.array([0.0, 0.0, 0.5, 0.0, 0.1]),   # 下に移動→空
            'confidence': 0.9,
            'result': 'empty'
        },
        {
            'vector': np.array([1.0, 1.0, 0.5, 1.0, 0.1]),   # ゴール
            'confidence': 1.0,
            'result': 'goal'
        }
    ]
    
    # 現在の状態ベクトル
    current_vector = np.array([0.0, 0.5, 0.5, 0.0, 0.2])
    
    print("=== 洞察ベクトル生成デモ ===\n")
    print(f"現在ベクトル: {current_vector}")
    print(f"エピソード数: {len(episode_memory)}\n")
    
    # 洞察ベクトル生成
    insight_vector = agent.generate_insight_vector(
        current_vector,
        episode_memory,
        uncertainty=0.3
    )
    
    print(f"生成された洞察ベクトル: {insight_vector}")
    print(f"  位置成分: X={insight_vector[0]:.3f}, Y={insight_vector[1]:.3f}")
    print(f"  行動成分: {insight_vector[2]:.3f}")
    print(f"  結果予測: {insight_vector[3]:.3f}")
    print(f"  訪問情報: {insight_vector[4]:.3f}")
    
    # 行動抽出
    action = agent.extract_action_from_insight(insight_vector)
    action_names = ['↑', '→', '↓', '←']
    print(f"\n抽出された行動: {action} ({action_names[action]})")


if __name__ == "__main__":
    demonstrate_insight_vector()