"""
統合インデックスを使用する拡張FileSystemDataStore
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

from insightspike.core.base.datastore import DataStore
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
from insightspike.index import IntegratedVectorGraphIndex, BackwardCompatibleWrapper

logger = logging.getLogger(__name__)


class EnhancedFileSystemDataStore(FileSystemDataStore):
    """
    統合インデックスを内部で使用するFileSystemDataStore
    
    既存のAPIを維持しながら、内部実装を統合インデックスに置き換え
    """
    
    def __init__(self, base_path: str, config: Optional[Dict] = None):
        """
        Args:
            base_path: データ保存先のベースパス
            config: 設定オプション
                - use_integrated_index: 統合インデックスを使用するか
                - dimension: ベクトル次元数（統合インデックス使用時）
                - migration_mode: 'shadow', 'partial', 'full'
        """
        super().__init__(base_path)
        
        self.config = config or {}
        self.use_integrated_index = self.config.get('use_integrated_index', False)
        
        if self.use_integrated_index:
            # 統合インデックスを初期化
            dimension = self.config.get('dimension', 768)
            # Minimal stub index does not accept additional config; dimension only
            self._integrated_index = IntegratedVectorGraphIndex(dimension=dimension)
            self._wrapper = BackwardCompatibleWrapper(self._integrated_index)
            
            # 既存データの移行（必要に応じて）
            self._handle_migration()
        else:
            self._integrated_index = None
            self._wrapper = None
    
    def _handle_migration(self):
        """既存データから統合インデックスへの移行処理"""
        migration_mode = self.config.get('migration_mode', 'shadow')
        
        if migration_mode == 'shadow':
            # シャドウモード：読み込み時に両方を更新
            logger.info("シャドウモードで統合インデックスを実行")
        elif migration_mode == 'partial':
            # 部分移行：新規データのみ統合インデックスに
            logger.info("部分移行モードで実行")
        elif migration_mode == 'full':
            # 完全移行：既存データを統合インデックスに移行
            logger.info("既存データを統合インデックスに移行中...")
            from insightspike.index import MigrationHelper
            stats = MigrationHelper.migrate_from_filesystem_store(
                self.base_path, 
                self._integrated_index
            )
            logger.info(f"移行完了: {stats}")
    
    def save_episodes(self, episodes: List[Dict[str, Any]], namespace: str = "episodes") -> bool:
        """エピソードの保存（統合インデックス対応）"""
        # 既存の実装を呼び出し
        success = super().save_episodes(episodes, namespace)
        
        # 統合インデックスにも保存
        if self.use_integrated_index and self._wrapper:
            try:
                wrapper_success = self._wrapper.save_episodes(episodes, namespace)
                success = success and wrapper_success
            except Exception as e:
                logger.error(f"統合インデックスへの保存でエラー: {e}")
        
        return success
    
    def load_episodes(self, namespace: str = "episodes") -> List[Dict[str, Any]]:
        """エピソードの読み込み（統合インデックス対応）"""
        if self.use_integrated_index and self._wrapper:
            # 統合インデックスから読み込み
            return self._wrapper.load_episodes(namespace)
        else:
            # 既存の実装
            return super().load_episodes(namespace)
    
    def find_similar(self, query_vector: np.ndarray, k: int = 10, 
                    namespace: str = "vectors") -> Tuple[List[int], List[float]]:
        """類似ベクトル検索（統合インデックス対応）"""
        if self.use_integrated_index and self._wrapper:
            # 統合インデックスで高速検索
            return self._wrapper.find_similar(query_vector, k, namespace)
        else:
            # 既存の実装がベクトルファイルを持たない場合があるため、
            # episodes.json から直接簡易コサイン検索を行う（ベースライン実装）。
            try:
                episodes = super().load_episodes("episodes")
                if episodes:
                    import numpy as _np
                    q = _np.array(query_vector, dtype=float).reshape(-1)
                    scored = []
                    for i, ep in enumerate(episodes):
                        v = ep.get("vec")
                        if v is None:
                            v = ep.get("embedding")
                        if v is None:
                            continue
                        v = _np.array(v, dtype=float).reshape(-1)
                        denom = (_np.linalg.norm(q) * _np.linalg.norm(v)) + 1e-12
                        sim = float((_np.dot(q, v)) / denom)
                        scored.append((i, sim))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    scored = scored[:k]
                    if scored:
                        idxs = [i for i, _ in scored]
                        dists = [1.0 - s for _, s in scored]
                        return idxs, dists
            except Exception:
                pass
            # 最後のフォールバック: FileSystemDataStore の search_vectors
            return super().search_vectors(query_vector, k, namespace)
    
    def search_vectors(self, query_vector: np.ndarray, k: int = 10,
                      namespace: str = "vectors") -> Tuple[List[int], List[float]]:
        """ベクトル検索（統合インデックス対応）"""
        if self.use_integrated_index and self._wrapper:
            return self._wrapper.search_vectors(query_vector, k, namespace)
        else:
            # find_similarを呼び出し（既存実装）
            return self.find_similar(query_vector, k, namespace)
    
    def get_performance_stats(self) -> Dict:
        """性能統計情報の取得"""
        stats = {
            'use_integrated_index': self.use_integrated_index
        }
        
        if self.use_integrated_index and self._integrated_index:
            stats.update({
                'total_vectors': len(self._integrated_index.normalized_vectors),
                'graph_nodes': self._integrated_index.graph.number_of_nodes(),
                'graph_edges': self._integrated_index.graph.number_of_edges(),
                'dimension': self._integrated_index.dimension
            })
        
        return stats
    
    def switch_to_integrated_index(self, migrate_existing: bool = True):
        """統合インデックスへの切り替え"""
        if not self.use_integrated_index:
            self.use_integrated_index = True
            
            # インデックス初期化
            dimension = self.config.get('dimension', 768)
            self._integrated_index = IntegratedVectorGraphIndex(dimension=dimension)
            self._wrapper = BackwardCompatibleWrapper(self._integrated_index)
            
            # 既存データの移行
            if migrate_existing:
                self.config['migration_mode'] = 'full'
                self._handle_migration()
                
            logger.info("統合インデックスに切り替えました")
