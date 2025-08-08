#!/usr/bin/env python3
"""
統合インデックスのメインコード統合テスト
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
import tempfile
from pathlib import Path

from insightspike.index import (
    IntegratedVectorGraphIndex,
    BackwardCompatibleWrapper,
    MigrationHelper
)
from insightspike.implementations.datastore.enhanced_filesystem_store import EnhancedFileSystemDataStore
from insightspike.config.index_config import IntegratedIndexConfig


class TestMainCodeIntegration:
    """メインコードとの統合テスト"""
    
    def test_enhanced_datastore_shadow_mode(self):
        """拡張DataStoreのシャドウモードテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # シャドウモードで初期化
            config = {
                'use_integrated_index': True,
                'dimension': 128,
                'migration_mode': 'shadow'
            }
            
            store = EnhancedFileSystemDataStore(tmpdir, config)
            
            # エピソード追加
            episodes = []
            for i in range(20):
                episode = {
                    'vec': list(np.random.randn(128)),
                    'text': f'Shadow mode episode {i}',
                    'c_value': 0.5 + i * 0.01
                }
                episodes.append(episode)
            
            # 保存
            success = store.save_episodes(episodes)
            assert success
            
            # 検索テスト
            query = np.random.randn(128)
            indices, scores = store.find_similar(query, k=5)
            assert len(indices) == 5
            
            # 性能統計確認
            stats = store.get_performance_stats()
            assert stats['use_integrated_index'] is True
            assert stats['total_vectors'] == 20
    
    def test_migration_from_existing_store(self):
        """既存DataStoreからの移行テスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 既存形式でデータ作成
            old_store = EnhancedFileSystemDataStore(tmpdir, {'use_integrated_index': False})
            
            episodes = []
            for i in range(30):
                episode = {
                    'vec': list(np.random.randn(64)),
                    'text': f'Legacy episode {i}',
                    'timestamp': i
                }
                episodes.append(episode)
            
            old_store.save_episodes(episodes)
            
            # 2. 統合インデックスに移行
            new_index = IntegratedVectorGraphIndex(dimension=64)
            stats = MigrationHelper.migrate_from_filesystem_store(
                tmpdir, new_index
            )
            
            assert stats['episodes_migrated'] == 30
            assert len(stats['errors']) == 0
            
            # 3. 検証
            validation = MigrationHelper.validate_migration(tmpdir, new_index)
            assert validation['success']
            assert validation['episode_count_match']
            assert validation['search_functionality']
    
    def test_config_based_switching(self):
        """設定ベースの切り替えテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 初期は統合インデックス無効
            store = EnhancedFileSystemDataStore(
                tmpdir, 
                {'use_integrated_index': False, 'dimension': 256}
            )
            
            # データ追加
            episodes = [
                {'vec': list(np.random.randn(256)), 'text': f'Ep {i}'}
                for i in range(10)
            ]
            store.save_episodes(episodes)
            
            # 統合インデックスに切り替え
            store.switch_to_integrated_index(migrate_existing=True)
            
            # 切り替え後の動作確認
            query = np.random.randn(256)
            indices, scores = store.find_similar(query, k=3)
            assert len(indices) == 3
            
            stats = store.get_performance_stats()
            assert stats['use_integrated_index'] is True
            assert stats['total_vectors'] == 10
    
    def test_performance_comparison(self):
        """性能比較テスト"""
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:
            
            # レガシーストア
            legacy_store = EnhancedFileSystemDataStore(
                tmpdir1, {'use_integrated_index': False}
            )
            
            # 統合インデックスストア
            integrated_store = EnhancedFileSystemDataStore(
                tmpdir2, {'use_integrated_index': True, 'dimension': 512}
            )
            
            # テストデータ
            episodes = []
            for i in range(100):
                episode = {
                    'vec': list(np.random.randn(512)),
                    'text': f'Performance test {i}'
                }
                episodes.append(episode)
            
            # データ追加
            legacy_store.save_episodes(episodes)
            integrated_store.save_episodes(episodes)
            
            # 検索性能測定
            query = np.random.randn(512)
            
            # レガシー
            start = time.time()
            for _ in range(10):
                legacy_store.find_similar(query, k=10)
            legacy_time = (time.time() - start) / 10
            
            # 統合インデックス
            start = time.time()
            for _ in range(10):
                integrated_store.find_similar(query, k=10)
            integrated_time = (time.time() - start) / 10
            
            print(f"\n性能比較:")
            print(f"  レガシー: {legacy_time*1000:.2f}ms")
            print(f"  統合インデックス: {integrated_time*1000:.2f}ms")
            print(f"  高速化: {legacy_time/integrated_time:.1f}x")
            
            # 統合インデックスの方が高速であることを確認
            assert integrated_time < legacy_time


def test_config_model():
    """設定モデルのテスト"""
    config = IntegratedIndexConfig(
        enabled=True,
        dimension=384,
        similarity_threshold=0.5,
        migration_mode="partial"
    )
    
    assert config.enabled is True
    assert config.dimension == 384
    assert config.similarity_threshold == 0.5
    assert config.migration_mode == "partial"
    
    # JSON変換テスト
    config_dict = config.dict()
    assert 'faiss_threshold' in config_dict
    assert config_dict['auto_save'] is True


if __name__ == "__main__":
    # 単体実行
    tester = TestMainCodeIntegration()
    
    print("=== 統合インデックス メインコード統合テスト ===\n")
    
    print("1. シャドウモードテスト...")
    tester.test_enhanced_datastore_shadow_mode()
    print("✅ 成功")
    
    print("\n2. 移行テスト...")
    tester.test_migration_from_existing_store()
    print("✅ 成功")
    
    print("\n3. 設定切り替えテスト...")
    tester.test_config_based_switching()
    print("✅ 成功")
    
    print("\n4. 性能比較テスト...")
    tester.test_performance_comparison()
    print("✅ 成功")
    
    print("\n5. 設定モデルテスト...")
    test_config_model()
    print("✅ 成功")
    
    print("\n🎉 全てのメインコード統合テストに合格！")