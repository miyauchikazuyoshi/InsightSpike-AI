#!/usr/bin/env python3
"""
Enhanced GeDIG Index with Inference Path Reinforcement
推論経路を強化する改良版GeDIGインデックス
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class EnhancedGeDIGIndex:
    """推論経路を記憶して強化するGeDIGインデックス"""
    
    def __init__(self, dimension: int, config: Optional[Dict] = None):
        self.dimension = dimension
        self.config = config or {}
        
        # 基本設定
        self.similarity_threshold = self.config.get('similarity_threshold', 0.5)
        self.gedig_threshold = self.config.get('gedig_threshold', 0.6)
        self.gedig_weight = self.config.get('gedig_weight', 0.3)
        self.max_edges_per_node = self.config.get('max_edges_per_node', 15)
        
        # 推論経路強化の設定
        self.inference_boost = self.config.get('inference_boost', 1.5)  # 推論経路のブースト係数
        self.path_memory_size = self.config.get('path_memory_size', 100)  # 記憶する経路数
        
        # ストレージ
        self.normalized_vectors = []
        self.vector_norms = []
        self.metadata = []
        self.graph = nx.Graph()
        
        # 推論経路の記憶
        self.inference_paths = []  # 最近の推論経路を記録
        self.path_success_rate = defaultdict(float)  # 経路の成功率
        
        # 統計
        self.stats = {
            'edge_additions': 0,
            'edge_rejections': 0,
            'inference_reinforcements': 0,
            'successful_paths': 0,
            'failed_paths': 0
        }
    
    def record_inference_path(self, indices: List[int], depth: int, success: bool = None):
        """
        推論経路を記録
        
        Args:
            indices: 検索で得られたインデックス
            depth: 使用された深度
            success: この推論が成功したか（行動の結果）
        """
        path_info = {
            'indices': indices[:20],  # 上位20個
            'depth': depth,
            'timestamp': len(self.inference_paths),
            'success': success
        }
        
        self.inference_paths.append(path_info)
        
        # 古い経路を削除
        if len(self.inference_paths) > self.path_memory_size:
            self.inference_paths.pop(0)
        
        # 成功率を更新
        if success is not None:
            path_key = tuple(sorted(indices[:5]))  # 上位5個をキーとして使用
            if success:
                self.path_success_rate[path_key] = (
                    self.path_success_rate[path_key] * 0.9 + 1.0 * 0.1
                )
                self.stats['successful_paths'] += 1
            else:
                self.path_success_rate[path_key] = (
                    self.path_success_rate[path_key] * 0.9 + 0.0 * 0.1
                )
                self.stats['failed_paths'] += 1
    
    def add_episode_with_inference_boost(self, episode: Dict, 
                                        last_inference_indices: Optional[List[int]] = None) -> int:
        """
        エピソードを追加（推論経路との関連を強化）
        
        Args:
            episode: 追加するエピソード
            last_inference_indices: 直前の推論で使用されたインデックス
        """
        # 通常のエピソード追加
        vec = np.array(episode['vec'], dtype=np.float32)
        norm = np.linalg.norm(vec)
        
        if norm == 0:
            normalized_vec = vec
        else:
            normalized_vec = vec / norm
        
        # 追加
        idx = len(self.normalized_vectors)
        self.normalized_vectors.append(normalized_vec)
        self.vector_norms.append(norm)
        self.metadata.append(episode)
        
        # グラフに追加
        self.graph.add_node(idx, **episode)
        
        # エッジ生成
        if len(self.normalized_vectors) > 1:
            # 通常のエッジ生成
            self._add_gedig_aware_edges(idx, normalized_vec)
            
            # 推論経路との関連を強化
            if last_inference_indices:
                self._reinforce_inference_connections(idx, last_inference_indices)
        
        return idx
    
    def _reinforce_inference_connections(self, new_idx: int, inference_indices: List[int]):
        """
        新しいエピソードと推論経路の間のエッジを強化
        
        Args:
            new_idx: 新しいエピソードのインデックス
            inference_indices: 推論で使用されたインデックス
        """
        new_vec = self.normalized_vectors[new_idx]
        
        # 推論経路の上位エピソードとの関連を強化
        for rank, inf_idx in enumerate(inference_indices[:10]):
            if inf_idx >= len(self.normalized_vectors) or inf_idx == new_idx:
                continue
            
            # 類似度計算
            inf_vec = self.normalized_vectors[inf_idx]
            similarity = np.dot(new_vec, inf_vec)
            
            # ランクに応じたブースト（上位ほど強い）
            rank_boost = 1.0 / (rank + 1)
            boosted_similarity = similarity * (1.0 + self.inference_boost * rank_boost)
            
            # geDIG値を計算（ブーストされた類似度で）
            gedig = self._calculate_gedig_simple(new_idx, inf_idx, boosted_similarity)
            
            # エッジが存在しない場合は追加、存在する場合は強化
            if not self.graph.has_edge(new_idx, inf_idx):
                if boosted_similarity >= self.similarity_threshold:
                    self.graph.add_edge(
                        new_idx, inf_idx,
                        weight=boosted_similarity,
                        similarity=similarity,
                        gedig=gedig,
                        inference_reinforced=True,
                        reinforcement_rank=rank
                    )
                    self.stats['inference_reinforcements'] += 1
            else:
                # 既存エッジの重みを強化
                edge_data = self.graph[new_idx][inf_idx]
                edge_data['weight'] = max(edge_data['weight'], boosted_similarity)
                edge_data['inference_reinforced'] = True
                self.stats['inference_reinforcements'] += 1
    
    def _calculate_gedig_simple(self, idx1: int, idx2: int, similarity: float) -> float:
        """簡易版geDIG計算"""
        # 位置距離
        meta1 = self.metadata[idx1]
        meta2 = self.metadata[idx2]
        
        if 'pos' in meta1 and 'pos' in meta2:
            pos1 = meta1['pos']
            pos2 = meta2['pos']
            spatial_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        else:
            spatial_dist = 0
        
        # 時間距離
        temporal_dist = abs(idx1 - idx2) / 100.0
        
        # GED (Generalized Edit Distance) - simplified
        ged = spatial_dist * 0.3 + temporal_dist * 0.2 + (1 - similarity) * 0.5
        
        # IG (Information Gain) - simplified
        ig = similarity * 0.5
        
        # geDIG
        return ged - ig
    
    def _add_gedig_aware_edges(self, idx: int, normalized_vec: np.ndarray):
        """通常のgeDIG基準でエッジを追加"""
        # 既存ベクトルとの類似度計算
        if len(self.normalized_vectors) <= 1:
            return
        
        existing_vectors = np.array(self.normalized_vectors[:-1])
        similarities = np.dot(existing_vectors, normalized_vec)
        
        # 候補を選定
        candidates = []
        for i, sim in enumerate(similarities):
            if sim >= self.similarity_threshold * 0.8:
                gedig = self._calculate_gedig_simple(idx, i, sim)
                if gedig < self.gedig_threshold:
                    candidates.append((i, sim, gedig))
        
        # スコアでソート
        candidates.sort(key=lambda x: x[1] - x[2] * self.gedig_weight, reverse=True)
        
        # 上位k個にエッジを追加
        for sim_idx, sim, gedig in candidates[:self.max_edges_per_node]:
            self.graph.add_edge(
                idx, sim_idx,
                weight=sim,
                similarity=sim,
                gedig=gedig,
                inference_reinforced=False
            )
            self.stats['edge_additions'] += 1
    
    def search_with_path_memory(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        経路記憶を考慮した検索
        
        Returns:
            (indices, scores)
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # 正規化
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            normalized_query = query_vector
        else:
            normalized_query = query_vector / query_norm
        
        if len(self.normalized_vectors) == 0:
            return np.array([]), np.array([])
        
        # 類似度計算
        all_vectors = np.array(self.normalized_vectors)
        similarities = np.dot(all_vectors, normalized_query)
        
        # 成功した経路のブースト
        boosted_scores = similarities.copy()
        for path_info in self.inference_paths[-10:]:  # 最近10個の経路
            if path_info.get('success'):
                for idx in path_info['indices'][:5]:
                    if idx < len(boosted_scores):
                        boosted_scores[idx] *= 1.2  # 成功経路をブースト
        
        # 上位k個を選択
        if k >= len(boosted_scores):
            indices = np.arange(len(boosted_scores))
            scores = boosted_scores
        else:
            indices = np.argpartition(boosted_scores, -k)[-k:]
            scores = boosted_scores[indices]
        
        # ソート
        sorted_idx = np.argsort(scores)[::-1]
        indices = indices[sorted_idx]
        scores = scores[sorted_idx]
        
        return indices, scores
    
    def get_inference_statistics(self) -> Dict:
        """推論統計を取得"""
        return {
            'total_paths_recorded': len(self.inference_paths),
            'successful_paths': self.stats['successful_paths'],
            'failed_paths': self.stats['failed_paths'],
            'inference_reinforcements': self.stats['inference_reinforcements'],
            'success_rate': (
                self.stats['successful_paths'] / 
                max(1, self.stats['successful_paths'] + self.stats['failed_paths'])
            ),
            'top_successful_paths': sorted(
                self.path_success_rate.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# 既存のGeDIGAwareIntegratedIndexとの互換性のためのエイリアス
GeDIGAwareIntegratedIndex = EnhancedGeDIGIndex