#!/usr/bin/env python3
"""
方向成分を除外した類似度検索の実験
"""

import numpy as np
from typing import Tuple, List


class MaskedSimilaritySearch:
    """特定次元をマスクした類似度検索"""
    
    def __init__(self, dimension: int = 7, mask_dims: List[int] = [2]):
        """
        Args:
            dimension: ベクトル次元数
            mask_dims: 除外する次元のインデックスリスト
        """
        self.dimension = dimension
        self.mask_dims = mask_dims
        
        # マスクベクトル作成（除外次元は0、それ以外は1）
        self.mask = np.ones(dimension, dtype=np.float32)
        for dim in mask_dims:
            self.mask[dim] = 0.0
        
        self.vectors = None
        
    def add(self, vectors: np.ndarray):
        """ベクトル追加"""
        # マスクを適用してから保存
        masked_vectors = vectors * self.mask
        
        if self.vectors is None:
            self.vectors = masked_vectors
        else:
            self.vectors = np.vstack([self.vectors, masked_vectors])
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        マスクした類似度検索
        
        Returns:
            distances: 類似度スコア
            indices: インデックス
        """
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([]), np.array([])
        
        # クエリにもマスクを適用
        masked_query = query * self.mask
        
        # コサイン類似度計算（マスクされた次元で）
        query_norm = np.linalg.norm(masked_query)
        vector_norms = np.linalg.norm(self.vectors, axis=1)
        
        # ゼロ除算を避ける
        norms = query_norm * vector_norms + 1e-8
        
        similarities = np.dot(self.vectors, masked_query) / norms
        
        # Top-k取得
        actual_k = min(k, len(similarities))
        top_k_idx = np.argpartition(similarities, -actual_k)[-actual_k:]
        top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
        
        return similarities[top_k_idx], top_k_idx


def test_masked_vs_normal():
    """マスクあり/なしの比較実験"""
    
    print("="*60)
    print("方向成分マスク検索 vs 通常検索の比較")
    print("="*60)
    
    # テストデータ作成（7次元エピソード）
    episodes = []
    
    # 位置(1,1)での様々な方向のエピソード
    for direction in [0, 0.33, 0.66, 1.0]:
        vec = np.array([
            0.1, 0.1,      # 位置 (1,1)
            direction,      # 方向（変化）
            1.0,           # 成功
            1.0,           # 通路
            0.1,           # 訪問回数少
            0.0            # ゴールではない
        ], dtype=np.float32)
        episodes.append(vec)
    
    # 位置(5,5)での様々な方向のエピソード
    for direction in [0, 0.33, 0.66, 1.0]:
        vec = np.array([
            0.5, 0.5,      # 位置 (5,5)
            direction,      # 方向（変化）
            0.0,           # 失敗
            -1.0,          # 壁
            0.5,           # 訪問回数中
            0.0            # ゴールではない
        ], dtype=np.float32)
        episodes.append(vec)
    
    episodes = np.array(episodes)
    
    # クエリ：位置(1,1)から成功する行動を探す
    query = np.array([
        0.1, 0.1,      # 現在位置 (1,1)
        0.5,           # 方向NULL
        1.0,           # 成功希望
        0.0,           # 壁/通路NULL
        0.1,           # 訪問回数
        0.0            # ゴールではない
    ], dtype=np.float32)
    
    print("\n🔍 クエリ: 位置(1,1)から成功する行動を探す")
    print(f"   クエリベクトル: {query}")
    
    # 通常検索
    print("\n📊 通常検索（全次元使用）:")
    normal_search = MaskedSimilaritySearch(dimension=7, mask_dims=[])
    normal_search.add(episodes)
    normal_scores, normal_indices = normal_search.search(query, k=4)
    
    for i, (score, idx) in enumerate(zip(normal_scores, normal_indices)):
        ep = episodes[idx]
        print(f"   {i+1}. スコア={score:.3f}, 位置=({ep[0]:.1f},{ep[1]:.1f}), "
              f"方向={ep[2]:.2f}, 成功={ep[3]}")
    
    # マスク検索（方向成分を除外）
    print("\n📊 マスク検索（方向成分を除外）:")
    masked_search = MaskedSimilaritySearch(dimension=7, mask_dims=[2])
    masked_search.add(episodes)
    masked_scores, masked_indices = masked_search.search(query, k=4)
    
    for i, (score, idx) in enumerate(zip(masked_scores, masked_indices)):
        ep = episodes[idx]
        print(f"   {i+1}. スコア={score:.3f}, 位置=({ep[0]:.1f},{ep[1]:.1f}), "
              f"方向={ep[2]:.2f}, 成功={ep[3]}")
    
    print("\n💡 分析:")
    print("- 通常検索: 方向成分(0.5)の影響で、全ての方向が同程度のスコア")
    print("- マスク検索: 方向を無視し、位置と成功/失敗で明確に区別")
    print("- マスク検索では同じ位置の成功エピソードが上位に")
    

def test_multi_mask():
    """複数次元のマスク実験"""
    
    print("\n" + "="*60)
    print("複数次元マスクの実験")
    print("="*60)
    
    episodes = np.random.rand(100, 7).astype(np.float32)
    query = np.array([0.1, 0.1, 0.5, 1.0, 0.0, 0.1, 0.0], dtype=np.float32)
    
    # 様々なマスクパターンを試す
    mask_patterns = [
        ([], "マスクなし"),
        ([2], "方向のみマスク"),
        ([4], "壁/通路のみマスク"),
        ([2, 4], "方向と壁/通路をマスク"),
        ([2, 4, 5], "方向、壁/通路、訪問回数をマスク")
    ]
    
    for mask_dims, description in mask_patterns:
        search = MaskedSimilaritySearch(dimension=7, mask_dims=mask_dims)
        search.add(episodes)
        scores, indices = search.search(query, k=3)
        
        print(f"\n📊 {description}:")
        print(f"   有効次元: {[i for i in range(7) if i not in mask_dims]}")
        print(f"   上位3件のスコア: {scores[:3]}")


if __name__ == "__main__":
    test_masked_vs_normal()
    test_multi_mask()