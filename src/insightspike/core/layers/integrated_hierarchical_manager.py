"""
Integrated Hierarchical Memory Manager
=====================================

階層的グラフとグラフ中心メモリ管理を統合
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import time

from .layer2_graph_centric import GraphCentricMemoryManager
from .hierarchical_graph_builder import HierarchicalGraphBuilder

logger = logging.getLogger(__name__)


class IntegratedHierarchicalManager:
    """
    統合階層的メモリマネージャー
    
    特徴:
    - GraphCentricMemoryManagerのエピソード管理
    - HierarchicalGraphBuilderの大規模検索
    - 自動的な階層構造の最適化
    """
    
    def __init__(self,
                 dimension: int = 384,
                 cluster_size: int = 100,
                 super_cluster_size: int = 100,
                 rebuild_threshold: int = 1000):
        """
        Args:
            dimension: ベクトル次元数
            cluster_size: 基本クラスタサイズ
            super_cluster_size: スーパークラスタサイズ
            rebuild_threshold: 階層再構築の閾値
        """
        # グラフ中心メモリマネージャー
        self.memory_manager = GraphCentricMemoryManager(dim=dimension)
        
        # 階層的グラフビルダー
        self.graph_builder = HierarchicalGraphBuilder(
            dimension=dimension,
            cluster_size=cluster_size,
            super_cluster_size=super_cluster_size
        )
        
        # 設定
        self.rebuild_threshold = rebuild_threshold
        self.episodes_since_rebuild = 0
        self.last_rebuild_time = 0
        
        # 統計
        self.stats = {
            'total_episodes': 0,
            'total_rebuilds': 0,
            'total_searches': 0,
            'avg_search_time': 0
        }
        
        logger.info(f"Integrated Hierarchical Manager initialized: "
                   f"cluster_size={cluster_size}, rebuild_threshold={rebuild_threshold}")
    
    def add_episode(self, vector: np.ndarray, text: str, 
                   metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        エピソード追加（階層的管理付き）
        
        Args:
            vector: エピソードベクトル
            text: テキスト
            metadata: メタデータ
            
        Returns:
            追加結果
        """
        # まずメモリマネージャーに追加（metadataは直接渡せないので後で設定）
        episode_idx = self.memory_manager.add_episode(vector, text)
        
        if episode_idx < 0:
            return {'success': False, 'error': 'Failed to add episode'}
        
        # メタデータを設定
        if metadata and 0 <= episode_idx < len(self.memory_manager.episodes):
            self.memory_manager.episodes[episode_idx].metadata.update(metadata)
        
        # 階層グラフに追加
        doc = {
            'embedding': vector,
            'text': text,
            'metadata': metadata or {},
            'episode_idx': episode_idx
        }
        
        if len(self.memory_manager.episodes) > self.graph_builder.stats['total_nodes'][0]:
            # 新規エピソードの場合、階層構造に追加
            self.graph_builder.add_document(doc)
            self.episodes_since_rebuild += 1
        
        # 再構築チェック
        if self._should_rebuild():
            self._rebuild_hierarchy()
        
        self.stats['total_episodes'] = len(self.memory_manager.episodes)
        
        return {
            'success': True,
            'episode_idx': episode_idx,
            'total_episodes': self.stats['total_episodes'],
            'importance': self.memory_manager.get_importance(episode_idx),
            'hierarchy_stats': self.graph_builder.get_statistics()
        }
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        階層的検索
        
        Args:
            query: 検索クエリ
            k: 返す結果数
            
        Returns:
            検索結果
        """
        start_time = time.time()
        
        # クエリをベクトル化
        from ...utils.embedder import get_model
        model = get_model()
        query_vector = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        
        # 階層的検索
        if self.graph_builder.stats['total_nodes'][0] > 0:
            # 階層構造を使った効率的な検索
            hierarchical_results = self.graph_builder.search_hierarchical(query_vector, k=k*2)
            
            # エピソードインデックスを取得
            episode_indices = [res['index'] for res in hierarchical_results]
            
            # メモリマネージャーから詳細情報取得
            results = []
            for idx in episode_indices[:k]:
                if 0 <= idx < len(self.memory_manager.episodes):
                    episode = self.memory_manager.episodes[idx]
                    importance = self.memory_manager.get_importance(idx)
                    
                    # 類似度再計算
                    similarity = float(np.dot(query_vector, episode.vec))
                    
                    results.append({
                        'index': idx,
                        'text': episode.text,
                        'score': similarity,
                        'importance': importance,
                        'metadata': episode.metadata
                    })
        else:
            # フォールバック：通常の検索
            results = self.memory_manager.search_episodes(query, k)
        
        # アクセス記録更新
        for res in results:
            self.memory_manager._update_access(res['index'])
        
        # 統計更新
        search_time = time.time() - start_time
        self.stats['total_searches'] += 1
        self.stats['avg_search_time'] = (
            (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time) /
            self.stats['total_searches']
        )
        
        logger.debug(f"Hierarchical search completed in {search_time:.3f}s")
        
        return results
    
    def _should_rebuild(self) -> bool:
        """再構築が必要かチェック"""
        # 新規エピソード数が閾値を超えた
        if self.episodes_since_rebuild >= self.rebuild_threshold:
            return True
        
        # 最後の再構築から24時間経過
        if time.time() - self.last_rebuild_time > 86400:
            return True
        
        # メモリ使用効率が悪化（圧縮率が低下）
        if self.graph_builder.stats['total_nodes'][0] > 0:
            compression_ratio = (
                self.graph_builder.stats['total_nodes'][0] / 
                max(1, self.graph_builder.stats['total_nodes'][2])
            )
            if compression_ratio < 10:  # 圧縮率が10倍未満
                return True
        
        return False
    
    def _rebuild_hierarchy(self):
        """階層構造を再構築"""
        logger.info("Rebuilding hierarchical structure...")
        start_time = time.time()
        
        # 全エピソードからドキュメント作成
        documents = []
        for i, episode in enumerate(self.memory_manager.episodes):
            documents.append({
                'embedding': episode.vec,
                'text': episode.text,
                'metadata': episode.metadata,
                'episode_idx': i
            })
        
        # 階層グラフ再構築
        result = self.graph_builder.build_hierarchical_graph(documents)
        
        # 統計更新
        self.episodes_since_rebuild = 0
        self.last_rebuild_time = time.time()
        self.stats['total_rebuilds'] += 1
        
        rebuild_time = time.time() - start_time
        logger.info(f"Hierarchy rebuilt in {rebuild_time:.2f}s: {result}")
    
    def optimize(self) -> Dict[str, Any]:
        """
        メモリとグラフ構造の最適化
        
        Returns:
            最適化結果
        """
        results = {}
        
        # 1. メモリ最適化（不要エピソードの削除）
        initial_count = len(self.memory_manager.episodes)
        removed_indices = []
        
        # 重要度が低く、アクセスされていないエピソードを特定
        for i in range(len(self.memory_manager.episodes)):
            importance = self.memory_manager.get_importance(i)
            episode = self.memory_manager.episodes[i]
            access_count = episode.metadata.get('access_count', 0)
            
            # 削除条件
            if importance < 0.1 and access_count == 0:
                removed_indices.append(i)
        
        # 削除実行（逆順）
        for idx in reversed(removed_indices):
            del self.memory_manager.episodes[idx]
        
        results['memory_optimization'] = {
            'initial_count': initial_count,
            'removed_count': len(removed_indices),
            'final_count': len(self.memory_manager.episodes)
        }
        
        # 2. インデックス再構築
        self.memory_manager._update_index()
        
        # 3. 階層構造の強制再構築
        if len(removed_indices) > 0:
            self._rebuild_hierarchy()
        
        # 4. 統計情報
        hierarchy_stats = self.graph_builder.get_statistics()
        results['hierarchy_stats'] = hierarchy_stats
        
        logger.info(f"Optimization complete: {results}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        統合統計情報
        
        Returns:
            統計情報
        """
        memory_stats = self.memory_manager.get_stats()
        hierarchy_stats = self.graph_builder.get_statistics()
        
        return {
            'memory': memory_stats,
            'hierarchy': hierarchy_stats,
            'integration': {
                'total_episodes': self.stats['total_episodes'],
                'total_rebuilds': self.stats['total_rebuilds'],
                'total_searches': self.stats['total_searches'],
                'avg_search_time_ms': self.stats['avg_search_time'] * 1000,
                'episodes_since_rebuild': self.episodes_since_rebuild,
                'compression_ratio': (
                    hierarchy_stats['nodes_per_level'][0] / 
                    max(1, hierarchy_stats['nodes_per_level'][2])
                    if hierarchy_stats['nodes_per_level'][0] > 0 else 0
                )
            }
        }
    
    def save_state(self, path: str):
        """状態を保存"""
        import pickle
        
        state = {
            'memory_episodes': self.memory_manager.episodes,
            'memory_config': {
                'integration': self.memory_manager.integration_config,
                'splitting': self.memory_manager.splitting_config
            },
            'hierarchy_levels': self.graph_builder.levels,
            'hierarchy_mappings': {
                'child_to_parent': dict(self.graph_builder.child_to_parent),
                'parent_to_children': dict(self.graph_builder.parent_to_children)
            },
            'stats': self.stats
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"State saved to {path}")
    
    def load_state(self, path: str):
        """状態を読み込み"""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # メモリ復元
        self.memory_manager.episodes = state['memory_episodes']
        self.memory_manager.integration_config = state['memory_config']['integration']
        self.memory_manager.splitting_config = state['memory_config']['splitting']
        
        # 階層構造復元
        self.graph_builder.levels = state['hierarchy_levels']
        self.graph_builder.child_to_parent = state['hierarchy_mappings']['child_to_parent']
        self.graph_builder.parent_to_children = state['hierarchy_mappings']['parent_to_children']
        
        # 統計復元
        self.stats = state['stats']
        
        # インデックス再構築
        self.memory_manager._update_index()
        
        logger.info(f"State loaded from {path}")
    
    def visualize_hierarchy(self) -> Dict[str, Any]:
        """階層構造の可視化情報を返す"""
        vis_data = {
            'levels': [],
            'connections': []
        }
        
        # 各レベルの情報
        for level in self.graph_builder.levels:
            vis_data['levels'].append({
                'level': level.level,
                'node_count': len(level.nodes),
                'sample_nodes': [
                    {
                        'id': node['id'],
                        'text': node['text'][:50] + '...',
                        'metadata': node.get('metadata', {})
                    }
                    for node in level.nodes[:5]  # 最初の5ノード
                ]
            })
        
        # 階層間接続のサンプル
        for (child_level, child_idx), (parent_level, parent_idx) in list(self.graph_builder.child_to_parent.items())[:20]:
            vis_data['connections'].append({
                'from': {'level': child_level, 'idx': child_idx},
                'to': {'level': parent_level, 'idx': parent_idx}
            })
        
        return vis_data