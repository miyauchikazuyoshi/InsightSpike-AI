"""
既存DataStoreから統合インデックスへの移行ヘルパー
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import logging

from .integrated_vector_graph_index import IntegratedVectorGraphIndex

logger = logging.getLogger(__name__)


class MigrationHelper:
    """既存システムから統合インデックスへの移行を支援"""
    
    @staticmethod
    def migrate_from_filesystem_store(
        old_store_path: str, 
        new_index: IntegratedVectorGraphIndex,
        dry_run: bool = False
    ) -> Dict:
        """
        FileSystemDataStoreから統合インデックスへ移行
        
        Args:
            old_store_path: 既存DataStoreのパス
            new_index: 移行先の統合インデックス
            dry_run: Trueの場合、実際の移行は行わずに検証のみ
            
        Returns:
            移行統計情報
        """
        stats = {
            'episodes_migrated': 0,
            'vectors_migrated': 0,
            'graphs_migrated': 0,
            'errors': [],
            'dry_run': dry_run
        }
        
        old_path = Path(old_store_path)
        
        try:
            # 1. エピソードの移行
            episodes_path = old_path / "core" / "episodes.json"
            if episodes_path.exists():
                logger.info(f"エピソードファイルを読み込み中: {episodes_path}")
                with open(episodes_path, 'r') as f:
                    episodes = json.load(f)
                    
                for ep in episodes:
                    try:
                        if not dry_run:
                            new_index.add_episode(ep)
                        stats['episodes_migrated'] += 1
                    except Exception as e:
                        stats['errors'].append(f"Episode error: {str(e)}")
                        
                logger.info(f"{stats['episodes_migrated']}個のエピソードを移行")
            
            # 2. 追加ベクトルの移行（もしあれば）
            vectors_path = old_path / "core" / "vectors.npy"
            metadata_path = old_path / "core" / "vectors_metadata.json"
            
            if vectors_path.exists() and metadata_path.exists():
                logger.info("追加ベクトルを読み込み中")
                vectors = np.load(vectors_path)
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # エピソードとして既に追加されているものはスキップ
                existing_ids = set()
                if 'episodes' in locals():
                    existing_ids = {ep.get('id') for ep in episodes if 'id' in ep}
                
                for vec, meta in zip(vectors, metadata):
                    if meta.get('id') not in existing_ids:
                        try:
                            if not dry_run:
                                new_index.add_vector(vec, meta)
                            stats['vectors_migrated'] += 1
                        except Exception as e:
                            stats['errors'].append(f"Vector error: {str(e)}")
                            
                logger.info(f"{stats['vectors_migrated']}個のベクトルを移行")
            
            # 3. グラフ情報（エッジは再構築されるため統計のみ）
            graph_files = list((old_path / "core").glob("*.pkl"))
            stats['graphs_migrated'] = len(graph_files)
            
        except Exception as e:
            stats['errors'].append(f"Migration error: {str(e)}")
            logger.error(f"移行中にエラーが発生: {e}")
            
        return stats
    
    @staticmethod
    def validate_migration(
        old_store_path: str,
        new_index: IntegratedVectorGraphIndex
    ) -> Dict:
        """
        移行結果の検証
        
        Returns:
            検証結果の辞書
        """
        validation = {
            'success': False,
            'episode_count_match': False,
            'vector_integrity': False,
            'search_functionality': False,
            'errors': []
        }
        
        try:
            # エピソード数の確認
            old_episodes_path = Path(old_store_path) / "core" / "episodes.json"
            if old_episodes_path.exists():
                with open(old_episodes_path, 'r') as f:
                    old_episodes = json.load(f)
                    
                validation['episode_count_match'] = (
                    len(old_episodes) == len(new_index.metadata)
                )
            
            # ベクトル整合性チェック（サンプリング）
            if len(new_index.metadata) > 0:
                sample_size = min(10, len(new_index.metadata))
                for i in range(sample_size):
                    try:
                        episode = new_index.get_episode(i)
                        vec = episode.get('vec')
                        if vec is not None and isinstance(vec, np.ndarray):
                            validation['vector_integrity'] = True
                            break
                    except Exception as e:
                        validation['errors'].append(f"Vector check error: {e}")
            
            # 検索機能の確認
            if len(new_index.normalized_vectors) > 0:
                try:
                    query = np.random.randn(new_index.dimension)
                    indices, scores = new_index.search(query, k=5)
                    validation['search_functionality'] = len(indices) > 0
                except Exception as e:
                    validation['errors'].append(f"Search test error: {e}")
            
            # 総合判定
            validation['success'] = (
                validation['episode_count_match'] and
                validation['vector_integrity'] and
                validation['search_functionality']
            )
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
            
        return validation
    
    @staticmethod
    def create_migration_report(
        migration_stats: Dict,
        validation_results: Dict
    ) -> str:
        """移行レポートの生成"""
        report = []
        report.append("=" * 60)
        report.append("統合インデックス移行レポート")
        report.append("=" * 60)
        report.append("")
        
        # 移行統計
        report.append("移行統計:")
        report.append(f"  - エピソード数: {migration_stats['episodes_migrated']}")
        report.append(f"  - ベクトル数: {migration_stats['vectors_migrated']}")
        report.append(f"  - グラフ数: {migration_stats['graphs_migrated']}")
        
        if migration_stats['errors']:
            report.append(f"  - エラー数: {len(migration_stats['errors'])}")
            report.append("    最初の3つのエラー:")
            for err in migration_stats['errors'][:3]:
                report.append(f"    * {err}")
        
        if migration_stats.get('dry_run'):
            report.append("\n※ ドライランモードで実行（実際の移行は行われていません）")
        
        report.append("")
        
        # 検証結果
        report.append("検証結果:")
        for key, value in validation_results.items():
            if key != 'errors':
                status = "✅" if value else "❌"
                report.append(f"  {status} {key}")
        
        if validation_results['errors']:
            report.append("\n検証エラー:")
            for err in validation_results['errors']:
                report.append(f"  * {err}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)