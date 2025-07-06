#!/usr/bin/env python3
"""
質問応答テスト - InsightSpike-AI
==============================

実際のユースケースをシミュレート：
1. 初期データの読み込み
2. 質問処理
3. 動的学習と統合
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_qa_workflow():
    """質問応答ワークフローテスト"""
    
    print("🤔 InsightSpike-AI 質問応答テスト")
    print("=" * 45)
    
    from insightspike.core.agents.main_agent import MainAgent
    from insightspike.core.config import get_config
    
    # エージェント初期化
    agent = MainAgent()
    
    # 初期知識を追加
    print("📚 初期知識ベースを構築中...")
    knowledge_base = [
        "人工知能は機械学習、深層学習、自然言語処理などの技術を含む広範な分野です",
        "機械学習は統計的手法を使ってデータからパターンを学習するAIの手法です",
        "深層学習は多層ニューラルネットワークを使った機械学習の一種です",
        "自然言語処理はコンピュータが人間の言語を理解・生成する技術です",
        "PyTorch Geometricはグラフニューラルネットワークのためのライブラリです"
    ]
    
    for i, knowledge in enumerate(knowledge_base):
        vector = np.random.random(384).astype(np.float32)
        agent.l2_memory.add_episode(vector, knowledge, c_value=0.5)
    
    print(f"✅ {len(knowledge_base)}個の初期知識を追加")
    
    # 質問リスト
    questions = [
        "人工知能とは何ですか？",
        "機械学習と深層学習の違いは？",
        "PyTorch Geometricは何に使われますか？",
        "自然言語処理の応用例は？"
    ]
    
    print(f"\n🔍 {len(questions)}個の質問でテスト中...")
    
    for i, question in enumerate(questions):
        print(f"\n質問 {i+1}: {question}")
        
        # 類似エピソードを検索（質問応答のシミュレーション）
        query_vector = np.random.random(384).astype(np.float32)
        results = agent.l2_memory.search_episodes(query_vector, k=2)
        
        if results:
            print("💡 関連する知識:")
            for j, result in enumerate(results[:2]):
                print(f"   {j+1}. [{result['weighted_score']:.3f}] {result['text'][:100]}...")
                
            # 学習フィードバック：良い質問には報酬
            episode_ids = [result['index'] for result in results[:1]]
            agent.l2_memory.update_c_values(episode_ids, [0.1])  # 小さな報酬
            print(f"   ✅ エピソード{episode_ids}にフィードバック報酬を付与")
        else:
            print("   ❌ 関連知識が見つかりませんでした")
    
    # 最終統計
    print(f"\n📊 最終メモリ状態:")
    stats = agent.l2_memory.get_memory_stats()
    print(f"   総エピソード: {stats['total_episodes']}")
    if agent.l2_memory.episodes:
        avg_c = sum(ep.c for ep in agent.l2_memory.episodes) / len(agent.l2_memory.episodes)
        print(f"   平均C-value: {avg_c:.3f}")
        
        # C-value分布
        c_values = [ep.c for ep in agent.l2_memory.episodes]
        print(f"   C-value範囲: {min(c_values):.3f} - {max(c_values):.3f}")
    else:
        print("   エピソードなし")
    
    print("\n🎉 質問応答テスト完了！")
    return True

if __name__ == "__main__":
    test_qa_workflow()
