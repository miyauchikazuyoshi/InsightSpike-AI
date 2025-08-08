#!/usr/bin/env python3
"""
Pure GeDIG Index - 純粋な情報理論的エッジ生成
推論は一時的、結果のみを記憶
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class PureGeDIGIndex:
    """
    純粋なgeDIG評価によるインデックス
    - 推論結果は破棄（一時的な計算のみ）
    - 実際の行動結果のみを記憶
    - 報酬や強化なし、純粋な情報価値で評価
    """
    
    def __init__(self, dimension: int, config: Optional[Dict] = None):
        self.dimension = dimension
        self.config = config or {}
        
        # 基本設定
        self.similarity_threshold = self.config.get('similarity_threshold', 0.5)
        self.gedig_threshold = self.config.get('gedig_threshold', 0.6)
        self.gedig_weight = self.config.get('gedig_weight', 0.3)
        self.max_edges_per_node = self.config.get('max_edges_per_node', 15)
        
        # ストレージ（実際の経験のみ）
        self.experience_vectors = []  # 実際の行動結果のみ
        self.experience_metadata = []
        self.experience_graph = nx.Graph()
        
        # 空間インデックス（効率的な検索用）
        self.spatial_index = defaultdict(list)
        
        # 統計
        self.stats = {
            'total_experiences': 0,
            'total_edges': 0,
            'avg_gedig': 0,
            'inference_count': 0,  # 推論回数（破棄されるが数は記録）
            'gedig_distribution': []
        }
    
    def add_experience(self, experience: Dict) -> int:
        """
        実際の経験（行動結果）のみを追加
        視覚観測や移動結果など、実際に起きたことのみ
        
        Args:
            experience: 経験エピソード
                - vec: 7次元ベクトル
                - type: 'visual' or 'movement'
                - pos: 位置
                - success: 成功/失敗（移動の場合）
        """
        # ベクトル正規化
        vec = np.array(experience['vec'], dtype=np.float32)
        norm = np.linalg.norm(vec)
        
        if norm > 0:
            normalized_vec = vec / norm
        else:
            normalized_vec = vec
        
        # 経験を追加
        idx = len(self.experience_vectors)
        self.experience_vectors.append(normalized_vec)
        self.experience_metadata.append(experience)
        
        # グラフノード追加
        self.experience_graph.add_node(idx, **experience)
        
        # 空間インデックス更新
        if 'pos' in experience:
            pos_key = f"{experience['pos'][0]},{experience['pos'][1]}"
            self.spatial_index[pos_key].append(idx)
        
        # 純粋なgeDIG評価でエッジ生成
        if idx > 0:
            self._create_pure_gedig_edges(idx, normalized_vec)
        
        self.stats['total_experiences'] += 1
        
        return idx
    
    def _create_pure_gedig_edges(self, new_idx: int, new_vec: np.ndarray):
        """
        純粋な情報理論的評価でエッジ生成
        報酬や強化なし、geDIG値のみで判断
        """
        if len(self.experience_vectors) <= 1:
            return
        
        # 全経験との類似度計算
        existing_vectors = np.array(self.experience_vectors[:-1])
        similarities = np.dot(existing_vectors, new_vec)
        
        # geDIG評価
        edge_candidates = []
        
        for i, similarity in enumerate(similarities):
            if similarity < self.similarity_threshold * 0.7:
                continue  # 類似度が低すぎる
            
            # 純粋なgeDIG計算
            gedig_value = self._calculate_pure_gedig(new_idx, i, similarity)
            
            if gedig_value < self.gedig_threshold:
                edge_candidates.append({
                    'target_idx': i,
                    'similarity': similarity,
                    'gedig': gedig_value,
                    'info_value': similarity - gedig_value  # 情報価値
                })
        
        # 情報価値でソート（高い方が良い）
        edge_candidates.sort(key=lambda x: x['info_value'], reverse=True)
        
        # 上位k個のエッジを生成
        for edge in edge_candidates[:self.max_edges_per_node]:
            self.experience_graph.add_edge(
                new_idx,
                edge['target_idx'],
                weight=edge['similarity'],
                gedig=edge['gedig'],
                info_value=edge['info_value']
            )
            self.stats['total_edges'] += 1
            self.stats['gedig_distribution'].append(edge['gedig'])
    
    def _calculate_pure_gedig(self, idx1: int, idx2: int, similarity: float) -> float:
        """
        純粋なgeDIG値計算
        Generalized Edit Distance - Information Gain
        """
        meta1 = self.experience_metadata[idx1]
        meta2 = self.experience_metadata[idx2]
        
        # 1. 空間距離成分
        spatial_distance = 0
        if 'pos' in meta1 and 'pos' in meta2:
            pos1 = np.array(meta1['pos'])
            pos2 = np.array(meta2['pos'])
            spatial_distance = np.linalg.norm(pos1 - pos2) / 10.0  # 正規化
        
        # 2. 時間距離成分（エピソードの順序）
        temporal_distance = abs(idx1 - idx2) / max(100, len(self.experience_vectors))
        
        # 3. タイプの違い（視覚vs移動）
        type_difference = 0
        if meta1.get('type') != meta2.get('type'):
            type_difference = 0.3
        
        # 4. 成功/失敗の差（移動エピソードの場合）
        outcome_difference = 0
        if 'success' in meta1 and 'success' in meta2:
            if meta1['success'] != meta2['success']:
                outcome_difference = 0.2
        
        # GED: 編集距離（違いの総和）
        ged = (spatial_distance * 0.3 + 
               temporal_distance * 0.3 + 
               type_difference * 0.2 + 
               outcome_difference * 0.2)
        
        # IG: 情報利得（類似性から得られる情報）
        ig = similarity * 0.5  # 類似度が高いほど情報価値が高い
        
        # geDIG = GED - IG（低いほど良い）
        return ged - ig
    
    def search_experiences(self, query_vec: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        経験を検索（推論用、結果は破棄される）
        
        Returns:
            (indices, scores) - 検索結果
        """
        self.stats['inference_count'] += 1
        
        # クエリ正規化
        query_vec = np.array(query_vec, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        if len(self.experience_vectors) == 0:
            return [], []
        
        # 類似度計算
        all_vectors = np.array(self.experience_vectors)
        similarities = np.dot(all_vectors, query_vec)
        
        # 上位k個を選択
        if k >= len(similarities):
            indices = list(range(len(similarities)))
            scores = similarities.tolist()
        else:
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            indices = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]].tolist()
            scores = similarities[indices].tolist()
        
        return indices, scores
    
    def get_experience(self, idx: int) -> Optional[Dict]:
        """特定の経験を取得"""
        if 0 <= idx < len(self.experience_metadata):
            return self.experience_metadata[idx]
        return None
    
    def get_neighbors(self, idx: int) -> List[Tuple[int, Dict]]:
        """
        指定経験の隣接ノードを取得
        geDIG値が低い（情報価値が高い）順にソート
        """
        if idx not in self.experience_graph:
            return []
        
        neighbors = []
        for neighbor_idx in self.experience_graph.neighbors(idx):
            edge_data = self.experience_graph[idx][neighbor_idx]
            neighbors.append((neighbor_idx, edge_data))
        
        # geDIG値でソート（低い方が良い）
        neighbors.sort(key=lambda x: x[1].get('gedig', float('inf')))
        
        return neighbors
    
    def pure_message_passing(self, start_indices: List[int], depth: int) -> np.ndarray:
        """
        純粋なメッセージパッシング（報酬なし）
        情報の伝播のみ、強化なし
        
        Returns:
            集約された洞察ベクトル（これも破棄される）
        """
        if not start_indices or depth <= 0:
            return np.zeros(self.dimension)
        
        # 初期メッセージ（ランクベースの重み）
        messages = {}
        for rank, idx in enumerate(start_indices[:20]):
            if idx < len(self.experience_vectors):
                messages[idx] = 1.0 / (rank + 1)
        
        # メッセージ伝播
        for d in range(depth):
            new_messages = {}
            decay = 0.8 ** d  # 距離による減衰
            
            for node_idx, msg_value in messages.items():
                # 自己ループ
                if d < depth - 1:
                    new_messages[node_idx] = msg_value * 0.5 * decay
                
                # 隣接ノードへ伝播
                for neighbor_idx, edge_data in self.get_neighbors(node_idx):
                    # geDIG値に基づく伝播（低いgeDIG = 高い情報価値）
                    gedig = edge_data.get('gedig', 1.0)
                    propagation_weight = (1.0 - min(gedig, 1.0)) * decay
                    propagated_value = msg_value * propagation_weight
                    
                    if neighbor_idx in new_messages:
                        new_messages[neighbor_idx] = max(
                            new_messages[neighbor_idx],
                            propagated_value
                        )
                    else:
                        new_messages[neighbor_idx] = propagated_value
            
            messages = new_messages
            if not messages:
                break
        
        # メッセージ集約（重み付き平均）
        aggregated = np.zeros(self.dimension)
        total_weight = 0
        
        for idx, weight in messages.items():
            if idx < len(self.experience_vectors):
                vec = self.experience_vectors[idx]
                aggregated += vec * weight
                total_weight += weight
        
        if total_weight > 0:
            aggregated = aggregated / total_weight
        
        return aggregated
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        stats = self.stats.copy()
        
        # geDIG分布の統計
        if self.stats['gedig_distribution']:
            gedig_values = self.stats['gedig_distribution']
            stats['avg_gedig'] = np.mean(gedig_values)
            stats['min_gedig'] = np.min(gedig_values)
            stats['max_gedig'] = np.max(gedig_values)
            stats['median_gedig'] = np.median(gedig_values)
        
        # グラフ統計
        if self.experience_graph.number_of_nodes() > 0:
            stats['graph_nodes'] = self.experience_graph.number_of_nodes()
            stats['graph_edges'] = self.experience_graph.number_of_edges()
            stats['graph_density'] = nx.density(self.experience_graph)
            
            # 次数分布
            degrees = [self.experience_graph.degree(n) for n in self.experience_graph.nodes()]
            stats['avg_degree'] = np.mean(degrees)
            stats['max_degree'] = np.max(degrees)
        
        return stats