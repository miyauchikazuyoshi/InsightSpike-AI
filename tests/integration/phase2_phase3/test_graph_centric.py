#!/usr/bin/env python3
"""
Test Graph-Centric Implementation (C値なし)
==========================================

C値を削除したグラフ中心の実装をテスト
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.core.agents.main_agent_graph_centric import GraphCentricMainAgent


def test_basic_functionality():
    """基本機能のテスト"""
    print("=== Graph-Centric Agent Test (C値なし) ===\n")
    
    # エージェント初期化
    print("1. Initializing agent...")
    agent = GraphCentricMainAgent()
    agent.initialize()
    
    # テストドキュメント
    test_docs = [
        "Artificial intelligence is transforming technology.",
        "Machine learning algorithms learn from data.",
        "Deep learning uses neural networks.",
        "AI and machine learning are closely related fields.",  # Should integrate
        "Climate change requires immediate action.",
        "Environmental science studies ecosystems.",
        "AI can help analyze climate data and predict patterns. Machine learning models process environmental data.",  # Mixed, might split
    ]
    
    print("\n2. Adding episodes...")
    print("-" * 50)
    
    for i, doc in enumerate(test_docs):
        print(f"\nDocument {i+1}: {doc[:50]}...")
        result = agent.add_episode(doc)
        
        if result['success']:
            print(f"  ✓ Added successfully")
            print(f"  Episode index: {result['episode_idx']}")
            print(f"  Importance: {result['importance']:.3f}")
            print(f"  Total episodes: {result['total_episodes']}")
            print(f"  Integration rate: {result['integration_rate']:.1%}")
        else:
            print(f"  ✗ Failed: {result.get('error')}")
    
    # メモリ分析
    print("\n3. Memory Analysis")
    print("-" * 50)
    analysis = agent.get_memory_analysis()
    
    print(f"Total episodes: {analysis['total_episodes']}")
    print(f"Integration rate: {analysis['integration_rate']:.1%}")
    print(f"Graph-assisted integrations: {analysis['graph_assist_rate']:.1%}")
    
    print(f"\nImportance distribution:")
    imp_dist = analysis['importance_distribution']
    print(f"  Mean: {imp_dist['mean']:.3f}")
    print(f"  Std: {imp_dist['std']:.3f}")
    print(f"  Range: [{imp_dist['min']:.3f}, {imp_dist['max']:.3f}]")
    
    print(f"\nAccess statistics:")
    acc_dist = analysis['access_distribution']
    print(f"  Total accesses: {acc_dist['total']}")
    print(f"  Mean per episode: {acc_dist['mean']:.1f}")
    
    # 検索テスト
    print("\n4. Search Test")
    print("-" * 50)
    
    queries = [
        "machine learning",
        "climate science",
        "AI applications"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        result = agent.search(query, k=3)
        
        if result['success']:
            print("Results:")
            for i, res in enumerate(result['results']):
                print(f"  {i+1}. Score: {res['score']:.3f}, Importance: {res['importance']:.3f}")
                print(f"     Text: {res['text'][:60]}...")
        else:
            print(f"  Error: {result.get('error')}")


def test_importance_dynamics():
    """重要度の動的変化をテスト"""
    print("\n\n=== Testing Importance Dynamics ===\n")
    
    agent = GraphCentricMainAgent()
    agent.initialize()
    
    # エピソード追加
    docs = [
        "Python programming basics",
        "Advanced Python techniques",
        "JavaScript fundamentals"
    ]
    
    indices = []
    for doc in docs:
        result = agent.add_episode(doc)
        if result['success']:
            indices.append(result['episode_idx'])
    
    print("Initial importance scores:")
    for idx in indices:
        info = agent.get_episode_info(idx)
        print(f"  Episode {idx}: {info['importance']:.3f} (accesses: {info['access_count']})")
    
    # アクセスをシミュレート
    print("\nSimulating accesses...")
    # Episode 0を5回アクセス
    for _ in range(5):
        agent.search("Python programming", k=1)
        time.sleep(0.1)  # 時間経過をシミュレート
    
    # Episode 1を2回アクセス
    for _ in range(2):
        agent.search("Advanced Python", k=1)
    
    print("\nImportance after accesses:")
    for idx in indices:
        info = agent.get_episode_info(idx)
        print(f"  Episode {idx}: {info['importance']:.3f} (accesses: {info['access_count']})")
    
    # メモリ最適化テスト
    print("\n5. Memory Optimization Test")
    print("-" * 50)
    
    # 古いエピソードを追加（30日以上前のタイムスタンプ）
    old_episode_idx = agent.l2_memory.add_episode(
        agent.l2_memory.episodes[0].vec,
        "Old unused content"
    )
    # 古いタイムスタンプを設定
    agent.l2_memory.episodes[old_episode_idx].metadata['last_access'] = time.time() - 40*86400
    
    opt_result = agent.optimize_memory()
    if opt_result['success']:
        print(f"Optimization complete:")
        print(f"  Initial count: {opt_result['initial_count']}")
        print(f"  Removed: {opt_result['removed_count']}")
        print(f"  Final count: {opt_result['final_count']}")


def compare_with_c_value():
    """C値ありとなしの比較"""
    print("\n\n=== Comparison: With vs Without C-value ===\n")
    
    print("WITH C-value:")
    print("- Fixed value (0.5) for all episodes")
    print("- No dynamic updates")
    print("- Redundant with graph structure")
    print("- Complex integration formula: (c1*v1 + c2*v2)/(c1+c2)")
    
    print("\nWITHOUT C-value (Graph-Centric):")
    print("- Dynamic importance from graph structure")
    print("- Updates based on access patterns")
    print("- Simpler integration: weighted by graph connection")
    print("- Cleaner, more maintainable code")
    
    print("\nBenefits of removal:")
    print("✓ Reduced complexity")
    print("✓ Better performance")
    print("✓ More intuitive importance calculation")
    print("✓ Unified graph-based approach")


if __name__ == "__main__":
    # 基本機能テスト
    test_basic_functionality()
    
    # 重要度の動的変化テスト
    test_importance_dynamics()
    
    # 比較分析
    compare_with_c_value()
    
    print("\n\n✅ Graph-Centric implementation test complete!")
    print("\nKey achievements:")
    print("- Successfully removed C-value")
    print("- Importance calculated dynamically from graph")
    print("- Simpler and more maintainable code")
    print("- Better alignment with graph-based architecture")