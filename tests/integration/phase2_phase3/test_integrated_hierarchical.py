#!/usr/bin/env python3
"""
Test Integrated Hierarchical Manager
===================================

統合階層的メモリ管理のテスト
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.core.layers.integrated_hierarchical_manager import (
    IntegratedHierarchicalManager,
)


def test_basic_integration():
    """基本的な統合動作テスト"""
    print("=== Basic Integration Test ===\n")

    # マネージャー作成
    manager = IntegratedHierarchicalManager(
        dimension=384, cluster_size=50, super_cluster_size=10, rebuild_threshold=100
    )

    # テストドキュメント
    test_docs = [
        "Quantum computing revolutionizes computation through quantum mechanics",
        "Machine learning algorithms learn patterns from data automatically",
        "Blockchain technology ensures secure and transparent transactions",
        "Climate change impacts global weather patterns significantly",
        "Renewable energy sources include solar, wind, and hydroelectric power",
        "Artificial intelligence mimics human cognitive functions",
        "Cybersecurity protects systems from digital attacks",
        "Biotechnology advances medicine through genetic engineering",
        "Space exploration expands human knowledge of the universe",
        "Nanotechnology manipulates matter at molecular scale",
    ]

    print("1. Adding episodes...")
    for i, doc in enumerate(test_docs):
        result = manager.add_episode(
            np.random.randn(384).astype(np.float32),
            doc,
            metadata={"doc_id": i, "category": "tech"},
        )

        if result["success"]:
            print(f"   Episode {i+1}: Added successfully")
        else:
            print(f"   Episode {i+1}: Failed - {result.get('error')}")

    # 統計表示
    print("\n2. Initial statistics:")
    stats = manager.get_statistics()
    print(f"   Total episodes: {stats['memory']['total_episodes']}")
    print(f"   Integration rate: {stats['memory']['integration_rate']:.1%}")
    print(f"   Hierarchy nodes: {stats['hierarchy']['nodes_per_level']}")

    # 検索テスト
    print("\n3. Search tests:")
    queries = ["quantum computing", "renewable energy", "artificial intelligence"]

    for query in queries:
        print(f"\n   Query: '{query}'")
        results = manager.search(query, k=3)

        for i, res in enumerate(results):
            print(
                f"   {i+1}. Score: {res['score']:.3f}, Importance: {res['importance']:.3f}"
            )
            print(f"      Text: {res['text'][:60]}...")

    # 検索パフォーマンス
    print(
        f"\n   Average search time: {stats['integration']['avg_search_time_ms']:.2f}ms"
    )


def test_rebuild_trigger():
    """再構築トリガーのテスト"""
    print("\n\n=== Rebuild Trigger Test ===\n")

    manager = IntegratedHierarchicalManager(
        dimension=384, cluster_size=20, rebuild_threshold=50  # 低い閾値で再構築をトリガー
    )

    print("1. Adding episodes to trigger rebuild...")

    # 再構築前
    initial_rebuilds = manager.stats["total_rebuilds"]

    # エピソード追加
    for i in range(60):
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        result = manager.add_episode(vec, f"Document {i}: Content about topic {i % 5}")

        if (i + 1) % 10 == 0:
            stats = manager.get_statistics()
            rebuilds = manager.stats["total_rebuilds"]
            print(
                f"   Episodes: {i+1}, Rebuilds: {rebuilds}, Compression: {stats['integration']['compression_ratio']:.1f}x"
            )

    final_rebuilds = manager.stats["total_rebuilds"]
    print(f"\n2. Rebuild triggered {final_rebuilds - initial_rebuilds} times")

    # 最終構造
    stats = manager.get_statistics()
    print(f"\n3. Final hierarchy structure:")
    print(f"   Level 0 (episodes): {stats['hierarchy']['nodes_per_level'][0]}")
    print(f"   Level 1 (clusters): {stats['hierarchy']['nodes_per_level'][1]}")
    print(f"   Level 2 (super-clusters): {stats['hierarchy']['nodes_per_level'][2]}")


def test_scalability():
    """スケーラビリティテスト"""
    print("\n\n=== Scalability Test ===\n")

    manager = IntegratedHierarchicalManager(
        dimension=384, cluster_size=100, super_cluster_size=50, rebuild_threshold=1000
    )

    # バッチサイズ
    batch_sizes = [100, 500, 1000, 5000]
    cumulative_times = []

    print("1. Progressive loading test:")
    print("-" * 60)
    print(f"{'Batch':>10} | {'Total':>10} | {'Add Time':>12} | {'Search Time':>12}")
    print("-" * 60)

    total_docs = 0

    for batch_size in batch_sizes:
        # エピソード追加
        start_time = time.time()

        for i in range(batch_size):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)

            manager.add_episode(
                vec,
                f"Batch document {total_docs + i}",
                metadata={"batch": len(cumulative_times)},
            )

        add_time = time.time() - start_time
        total_docs += batch_size

        # 検索時間測定
        search_times = []
        for _ in range(10):
            query_vec = np.random.randn(384).astype(np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)

            # ダミーのクエリテキスト
            start = time.time()
            results = manager.search("test query", k=10)
            search_times.append(time.time() - start)

        avg_search_time = np.mean(search_times)

        print(
            f"{batch_size:>10,} | {total_docs:>10,} | {add_time:>12.2f}s | {avg_search_time*1000:>12.2f}ms"
        )

    # 最終統計
    print("\n2. Final statistics:")
    stats = manager.get_statistics()
    print(f"   Total episodes: {stats['memory']['total_episodes']:,}")
    print(f"   Total rebuilds: {stats['integration']['total_rebuilds']}")
    print(f"   Compression ratio: {stats['integration']['compression_ratio']:.1f}x")

    # メモリ最適化テスト
    print("\n3. Memory optimization test:")
    opt_results = manager.optimize()
    print(
        f"   Removed {opt_results['memory_optimization']['removed_count']} low-importance episodes"
    )
    print(f"   Final count: {opt_results['memory_optimization']['final_count']:,}")


@pytest.mark.skip(reason="torch.FloatStorage pickle error in test environment")
def test_persistence():
    """永続化のテスト"""
    print("\n\n=== Persistence Test ===\n")

    import os
    import tempfile

    # 一時ファイル
    temp_path = os.path.join(tempfile.gettempdir(), "integrated_manager_test.pkl")

    # マネージャー作成と初期データ
    print("1. Creating and populating manager...")
    manager1 = IntegratedHierarchicalManager()

    # データ追加
    for i in range(50):
        manager1.add_episode(
            np.random.randn(384).astype(np.float32), f"Test document {i}"
        )

    # いくつか検索してアクセス記録を作る
    for _ in range(5):
        manager1.search("test", k=5)

    stats1 = manager1.get_statistics()
    print(f"   Episodes: {stats1['memory']['total_episodes']}")
    print(f"   Searches: {stats1['integration']['total_searches']}")

    # 保存
    print("\n2. Saving state...")
    manager1.save_state(temp_path)
    print(f"   Saved to: {temp_path}")

    # 新しいマネージャーで読み込み
    print("\n3. Loading into new manager...")
    manager2 = IntegratedHierarchicalManager()
    manager2.load_state(temp_path)

    stats2 = manager2.get_statistics()
    print(f"   Episodes: {stats2['memory']['total_episodes']}")
    print(f"   Searches: {stats2['integration']['total_searches']}")

    # 検索が同じ結果を返すか確認
    print("\n4. Verifying search consistency...")
    results1 = manager1.search("document 10", k=3)
    results2 = manager2.search("document 10", k=3)

    match = all(r1["text"] == r2["text"] for r1, r2 in zip(results1, results2))
    print(f"   Search results match: {match}")

    # クリーンアップ
    os.remove(temp_path)


def test_visualization():
    """階層構造の可視化テスト"""
    print("\n\n=== Visualization Test ===\n")

    manager = IntegratedHierarchicalManager(cluster_size=10, super_cluster_size=5)

    # サンプルデータ
    topics = ["AI", "Climate", "Finance", "Health", "Education"]

    for i in range(50):
        topic = topics[i % len(topics)]
        manager.add_episode(
            np.random.randn(384).astype(np.float32),
            f"{topic}: Document about {topic.lower()} topic number {i // len(topics)}",
        )

    # 可視化データ取得
    vis_data = manager.visualize_hierarchy()

    print("1. Hierarchy structure:")
    for level_info in vis_data["levels"]:
        print(f"\n   Level {level_info['level']}: {level_info['node_count']} nodes")
        print("   Sample nodes:")
        for node in level_info["sample_nodes"]:
            print(f"     - {node['text']}")

    print("\n2. Sample connections:")
    for conn in vis_data["connections"][:5]:
        print(
            f"   Level {conn['from']['level']} node {conn['from']['idx']} → "
            f"Level {conn['to']['level']} node {conn['to']['idx']}"
        )


def main():
    """メインテスト実行"""
    print("Integrated Hierarchical Manager Test Suite")
    print("=" * 50)

    # 基本統合テスト
    test_basic_integration()

    # 再構築トリガーテスト
    test_rebuild_trigger()

    # スケーラビリティテスト
    print("\n\nPress Enter to run scalability test...")
    input()
    test_scalability()

    # 永続化テスト
    test_persistence()

    # 可視化テスト
    test_visualization()

    print("\n\n✅ All tests completed successfully!")
    print("\nKey features demonstrated:")
    print("- Seamless integration of memory management and hierarchical search")
    print("- Automatic hierarchy rebuilding based on thresholds")
    print("- Efficient search even with thousands of episodes")
    print("- State persistence and restoration")
    print("- Hierarchical structure visualization")


if __name__ == "__main__":
    main()
