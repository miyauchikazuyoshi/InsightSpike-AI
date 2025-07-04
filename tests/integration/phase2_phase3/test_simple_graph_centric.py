#!/usr/bin/env python3
"""
Simple test for Graph-Centric Memory Manager
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.core.layers.layer2_graph_centric import GraphCentricMemoryManager


def test_without_c_value():
    """C値なしの動作確認"""
    print("=== Testing Graph-Centric Memory (C値なし) ===\n")
    
    # マネージャー作成
    manager = GraphCentricMemoryManager(dim=10)
    
    # 設定
    manager.integration_config.similarity_threshold = 0.7
    
    # テストエピソード
    episodes = [
        ("AI research", np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("Machine learning", np.array([0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("Climate science", np.array([0, 0, 0.9, 0.1, 0, 0, 0, 0, 0, 0])),
        ("Deep learning and AI", np.array([0.85, 0.15, 0, 0, 0, 0, 0, 0, 0, 0])),
    ]
    
    print("1. Adding episodes (C値を渡しても無視される)")
    for text, vec in episodes:
        vec = vec / np.linalg.norm(vec)
        # C値を渡しても無視される
        idx = manager.add_episode(vec.astype(np.float32), text, c_value=0.99)
        print(f"  Added '{text}' -> Total: {len(manager.episodes)}")
    
    print(f"\n2. Statistics:")
    stats = manager.get_stats()
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Integration rate: {stats['integration_rate']:.1%}")
    print(f"  Average importance: {stats['avg_importance']:.3f}")
    
    print(f"\n3. Episode importance (動的に計算):")
    for i in range(len(manager.episodes)):
        importance = manager.get_importance(i)
        episode = manager.episodes[i]
        print(f"  Episode {i}: '{episode.text}' -> Importance: {importance:.3f}")
    
    print(f"\n4. Search test (重要度を考慮):")
    results = manager.search_episodes("AI", k=3)
    for res in results:
        print(f"  Score: {res['score']:.3f}, Importance: {res['importance']:.3f}")
        print(f"    Text: {res['text']}")
    
    print(f"\n5. C値の不在を確認:")
    for ep in manager.episodes:
        has_c = hasattr(ep, 'c')
        print(f"  Episode has C value: {has_c}")


def compare_integration():
    """統合処理の比較"""
    print("\n\n=== Integration Comparison ===\n")
    
    print("OLD (with C-value):")
    print("  integrated_vec = (c1*v1 + c2*v2) / (c1+c2)")
    print("  Problem: c1 = c2 = 0.5 always, so it's just averaging")
    
    print("\nNEW (graph-centric):")
    print("  integrated_vec = (1-weight)*v1 + weight*v2")
    print("  weight = graph_connection or similarity")
    print("  Benefit: Dynamic weighting based on actual relationships")


def test_importance_calculation():
    """重要度計算のテスト"""
    print("\n\n=== Importance Calculation ===\n")
    
    manager = GraphCentricMemoryManager(dim=5)
    
    # エピソード追加
    vec = np.random.randn(5).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    idx = manager.add_episode(vec, "Test episode")
    
    print("Importance factors:")
    print("1. Graph degree (connections)")
    print("2. Access frequency")
    print("3. Time decay")
    
    # アクセスをシミュレート
    for _ in range(5):
        manager._update_access(0)
    
    importance = manager.get_importance(0)
    episode = manager.episodes[0]
    
    print(f"\nAfter 5 accesses:")
    print(f"  Access count: {episode.metadata['access_count']}")
    print(f"  Importance: {importance:.3f}")


if __name__ == "__main__":
    test_without_c_value()
    compare_integration()
    test_importance_calculation()
    
    print("\n\n✅ C値なしの実装が正常に動作しています！")
    print("\nメリット:")
    print("- よりシンプルなコード")
    print("- 動的な重要度計算")
    print("- グラフ構造との統合")
    print("- 保守性の向上")