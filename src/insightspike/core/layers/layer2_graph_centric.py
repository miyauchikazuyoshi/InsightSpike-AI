"""
Graph-Centric L2 Memory Manager (C値なし)
=========================================

C値を削除し、グラフ構造から動的に重要度を計算する新しいメモリマネージャー。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

import faiss
import numpy as np
import torch

from .layer2_memory_manager import L2MemoryManager
from ...config import get_config
from ..config import get_config as get_core_config
from ...utils.embedder import get_model

logger = logging.getLogger(__name__)


@dataclass
class GraphEpisode:
    """C値なしのシンプルなエピソード"""
    vec: np.ndarray
    text: str
    metadata: Dict[str, Any]
    
    def __init__(self, vec: np.ndarray, text: str, metadata: Optional[Dict] = None):
        self.vec = vec.astype(np.float32)
        self.text = text
        self.metadata = metadata or {}
        # アクセス追跡
        self.metadata.setdefault('access_count', 0)
        self.metadata.setdefault('last_access', time.time())
        self.metadata.setdefault('created_at', time.time())
    
    def __repr__(self):
        return f"GraphEpisode(text='{self.text[:50]}...')"


class GraphCentricMemoryManager(L2MemoryManager):
    """
    グラフ中心のメモリマネージャー
    
    特徴:
    - C値なし
    - グラフ構造から動的に重要度を計算
    - エッジ重みを活用した統合
    - アクセスパターンの追跡
    """
    
    def __init__(self, dim: int = 384, *args, **kwargs):
        # 親クラスの初期化をスキップして独自実装
        self.dim = dim
        self.episodes: List[GraphEpisode] = []
        self.is_trained = False
        self._is_initialized = False
        
        # グラフ関連
        self.l3_graph = None
        
        # 統合設定
        self.integration_config = IntegrationConfig()
        self.splitting_config = SplittingConfig()
        
        # 統計
        self.stats = {
            'total_episodes_added': 0,
            'total_integrations': 0,
            'total_splits': 0,
            'graph_assisted_integrations': 0
        }
        
        # FAISSインデックス（必要に応じて初期化）
        self.index = None
        
        logger.info("Graph-Centric Memory Manager initialized (C-value free)")
    
    def add_episode(self, vector: np.ndarray, text: str, c_value: float = None) -> int:
        """
        エピソード追加（C値は無視）
        
        Args:
            vector: エピソードベクトル
            text: テキスト
            c_value: 互換性のため受け取るが無視
            
        Returns:
            エピソードインデックス
        """
        self.stats['total_episodes_added'] += 1
        
        # グラフベースの統合チェック
        integration_result = self._check_integration(vector, text)
        
        if integration_result['should_integrate']:
            # 統合実行
            target_idx = integration_result['target_index']
            self._integrate_episodes(target_idx, vector, text, integration_result.get('weight', 0.5))
            self.stats['total_integrations'] += 1
            
            if integration_result.get('graph_assisted', False):
                self.stats['graph_assisted_integrations'] += 1
                
            return target_idx
        else:
            # 新規エピソードとして追加
            episode = GraphEpisode(vector, text)
            self.episodes.append(episode)
            
            # インデックス更新
            self._update_index()
            
            # 分裂チェック
            if self.splitting_config.enable_auto_split:
                self._check_and_split_all()
            
            return len(self.episodes) - 1
    
    def _check_integration(self, vector: np.ndarray, text: str) -> Dict[str, Any]:
        """
        グラフ構造を使った統合判定
        """
        if not self.episodes:
            return {'should_integrate': False}
        
        best_match = {
            'index': -1,
            'similarity': 0.0,
            'graph_weight': 0.0,
            'combined_score': 0.0
        }
        
        # 各エピソードとの類似度計算
        for i, episode in enumerate(self.episodes):
            # ベクトル類似度
            vec_sim = np.dot(vector, episode.vec) / (
                np.linalg.norm(vector) * np.linalg.norm(episode.vec)
            )
            
            # グラフ重み（もしあれば）
            graph_weight = self._get_graph_weight(len(self.episodes), i)
            
            # コンテンツ重複
            content_overlap = self._calculate_content_overlap(text, episode.text)
            
            # 複合スコア
            if graph_weight > 0:
                # グラフ接続がある場合は重視
                combined = 0.5 * vec_sim + 0.3 * graph_weight + 0.2 * content_overlap
            else:
                # グラフ接続がない場合は従来の判定
                combined = 0.7 * vec_sim + 0.3 * content_overlap
            
            if combined > best_match['combined_score']:
                best_match = {
                    'index': i,
                    'similarity': vec_sim,
                    'graph_weight': graph_weight,
                    'combined_score': combined,
                    'content_overlap': content_overlap
                }
        
        # 閾値判定（グラフ接続があれば緩和）
        threshold = self.integration_config.similarity_threshold
        if best_match['graph_weight'] > 0.5:
            threshold -= self.integration_config.graph_connection_bonus
        
        should_integrate = (
            best_match['combined_score'] >= threshold and
            best_match['content_overlap'] >= self.integration_config.content_overlap_threshold
        )
        
        return {
            'should_integrate': should_integrate,
            'target_index': best_match['index'],
            'weight': best_match['graph_weight'] if best_match['graph_weight'] > 0 else best_match['similarity'],
            'graph_assisted': best_match['graph_weight'] > 0,
            'scores': best_match
        }
    
    def _integrate_episodes(self, target_idx: int, new_vec: np.ndarray, new_text: str, weight: float):
        """
        エピソード統合（C値なし、重みベース）
        """
        if not (0 <= target_idx < len(self.episodes)):
            return
        
        target = self.episodes[target_idx]
        
        # 重み付き平均（weightが高いほど新しいベクトルを重視）
        integrated_vec = (1 - weight) * target.vec + weight * new_vec
        integrated_vec = integrated_vec / np.linalg.norm(integrated_vec)
        
        # テキスト結合
        target.text = f"{target.text} | {new_text}"
        target.vec = integrated_vec.astype(np.float32)
        
        # メタデータ更新
        target.metadata['integration_count'] = target.metadata.get('integration_count', 0) + 1
        target.metadata['last_integration'] = time.time()
        
        # アクセス情報更新
        self._update_access(target_idx)
    
    def _check_and_split_all(self):
        """
        全エピソードの分裂チェック
        """
        episodes_to_split = []
        
        for i in range(len(self.episodes)):
            # 長さチェック
            if len(self.episodes[i].text) > self.splitting_config.max_episode_length:
                episodes_to_split.append((i, 'length'))
                continue
            
            # コンフリクトチェック
            conflict_score = self._calculate_conflict(i)
            if conflict_score > self.splitting_config.conflict_threshold:
                episodes_to_split.append((i, f'conflict:{conflict_score:.2f}'))
        
        # 逆順で分裂（インデックスのずれを防ぐ）
        for idx, reason in reversed(episodes_to_split):
            if idx < len(self.episodes):
                self._split_episode(idx, reason)
                self.stats['total_splits'] += 1
    
    def _split_episode(self, idx: int, reason: str):
        """
        エピソード分裂
        """
        if not (0 <= idx < len(self.episodes)):
            return
        
        episode = self.episodes[idx]
        sentences = [s.strip() for s in episode.text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return
        
        # 分裂数決定
        n_splits = min(len(sentences), self.splitting_config.max_splits_per_episode)
        
        # 新エピソード作成
        new_episodes = []
        for i in range(n_splits):
            start = i * len(sentences) // n_splits
            end = (i + 1) * len(sentences) // n_splits
            split_text = '. '.join(sentences[start:end]) + '.'
            
            # ベクトル変動
            noise = np.random.normal(0, 0.05, episode.vec.shape)
            split_vec = episode.vec + noise
            split_vec = split_vec / np.linalg.norm(split_vec)
            
            metadata = episode.metadata.copy()
            metadata.update({
                'split_from': idx,
                'split_reason': reason,
                'split_part': i + 1
            })
            
            new_episodes.append(GraphEpisode(split_vec, split_text, metadata))
        
        # 元エピソード削除
        del self.episodes[idx]
        
        # 新エピソード追加
        self.episodes.extend(new_episodes)
        
        # インデックス更新
        self._update_index()
    
    def get_importance(self, episode_idx: int) -> float:
        """
        エピソードの重要度を動的に計算
        
        考慮要素:
        - グラフ次数（接続数）
        - アクセス頻度
        - 時間的減衰
        """
        if not (0 <= episode_idx < len(self.episodes)):
            return 0.0
        
        episode = self.episodes[episode_idx]
        
        # アクセス頻度
        access_count = episode.metadata.get('access_count', 0)
        access_score = np.log1p(access_count) / 10.0  # 対数スケール
        
        # 時間的減衰
        last_access = episode.metadata.get('last_access', time.time())
        time_decay = np.exp(-(time.time() - last_access) / 86400)  # 1日で減衰
        
        # グラフ次数
        graph_degree = 0.0
        if self.l3_graph and hasattr(self.l3_graph, 'previous_graph'):
            graph = self.l3_graph.previous_graph
            if graph and hasattr(graph, 'edge_index'):
                edge_index = graph.edge_index
                degree = (edge_index[0] == episode_idx).sum() + (edge_index[1] == episode_idx).sum()
                graph_degree = degree.item() / max(len(self.episodes), 1)
        
        # 複合スコア
        importance = (
            0.4 * graph_degree +      # グラフ構造
            0.3 * access_score +      # アクセス頻度
            0.3 * time_decay         # 時間的関連性
        )
        
        return float(min(1.0, importance))
    
    def search_episodes(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        エピソード検索（重要度を考慮）
        """
        from ...utils.embedder import get_model
        model = get_model()
        query_vec = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        
        results = []
        for i, episode in enumerate(self.episodes):
            similarity = np.dot(query_vec, episode.vec)
            importance = self.get_importance(i)
            
            # 重要度で調整されたスコア
            adjusted_score = similarity * (0.7 + 0.3 * importance)
            
            results.append({
                'index': i,
                'text': episode.text,
                'similarity': float(similarity),
                'importance': importance,
                'score': adjusted_score
            })
            
            # アクセス更新
            self._update_access(i)
        
        # スコアでソート
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:k]
    
    def _update_access(self, episode_idx: int):
        """エピソードアクセス情報を更新"""
        if 0 <= episode_idx < len(self.episodes):
            episode = self.episodes[episode_idx]
            episode.metadata['access_count'] = episode.metadata.get('access_count', 0) + 1
            episode.metadata['last_access'] = time.time()
    
    def _calculate_content_overlap(self, text1: str, text2: str) -> float:
        """コンテンツ重複度計算"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_graph_weight(self, idx1: int, idx2: int) -> float:
        """グラフエッジ重みを取得"""
        if not self.l3_graph or not hasattr(self.l3_graph, 'previous_graph'):
            return 0.0
        
        graph = self.l3_graph.previous_graph
        if not graph or not hasattr(graph, 'edge_index'):
            return 0.0
        
        edge_index = graph.edge_index
        
        # エッジ存在チェック
        mask = ((edge_index[0] == idx1) & (edge_index[1] == idx2)) | \
               ((edge_index[0] == idx2) & (edge_index[1] == idx1))
        
        if mask.any():
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                edge_idx = torch.where(mask)[0][0]
                return float(graph.edge_attr[edge_idx])
            else:
                return 0.8  # デフォルト重み
        
        return 0.0
    
    def _calculate_conflict(self, idx: int) -> float:
        """エピソードのコンフリクトスコア計算"""
        if not self.l3_graph or not hasattr(self.l3_graph, 'previous_graph'):
            return 0.0
        
        graph = self.l3_graph.previous_graph
        if not graph or not hasattr(graph, 'edge_index'):
            return 0.0
        
        # 隣接ノード取得
        edge_index = graph.edge_index
        neighbors = edge_index[1][edge_index[0] == idx].tolist()
        
        if len(neighbors) < self.splitting_config.min_connections_for_split:
            return 0.0
        
        # ペアワイズコンフリクト計算
        conflicts = []
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[i] < len(self.episodes) and neighbors[j] < len(self.episodes):
                    vec1 = self.episodes[neighbors[i]].vec
                    vec2 = self.episodes[neighbors[j]].vec
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    conflicts.append(1.0 - similarity)
        
        return np.mean(conflicts) if conflicts else 0.0
    
    def _update_index(self):
        """FAISSインデックス更新"""
        if len(self.episodes) < 2:
            self.is_trained = False
            return
        
        # ベクトル抽出
        vectors = np.array([ep.vec for ep in self.episodes], dtype=np.float32)
        
        # インデックス作成
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.is_trained = True
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        # 重要度分布
        importances = [self.get_importance(i) for i in range(len(self.episodes))]
        
        return {
            'total_episodes': len(self.episodes),
            'total_added': self.stats['total_episodes_added'],
            'total_integrations': self.stats['total_integrations'],
            'total_splits': self.stats['total_splits'],
            'integration_rate': self.stats['total_integrations'] / max(1, self.stats['total_episodes_added']),
            'graph_assist_rate': self.stats['graph_assisted_integrations'] / max(1, self.stats['total_integrations']),
            'avg_importance': np.mean(importances) if importances else 0.0,
            'max_importance': max(importances) if importances else 0.0,
            'min_importance': min(importances) if importances else 0.0
        }
    
    def set_layer3_graph(self, l3_graph):
        """Layer3グラフ参照を設定"""
        self.l3_graph = l3_graph
        logger.info("Layer3 graph reference set for graph-centric manager")


# 設定クラス（C値関連を削除）
@dataclass
class IntegrationConfig:
    """統合設定"""
    similarity_threshold: float = 0.85
    content_overlap_threshold: float = 0.70
    graph_connection_bonus: float = 0.1


@dataclass  
class SplittingConfig:
    """分裂設定"""
    conflict_threshold: float = 0.7
    min_connections_for_split: int = 3
    max_episode_length: int = 500
    enable_auto_split: bool = True
    max_splits_per_episode: int = 3