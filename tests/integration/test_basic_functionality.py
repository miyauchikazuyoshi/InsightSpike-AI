#!/usr/bin/env python3
"""
基本機能テスト - InsightSpike-AI
==================================

エンドツーエンドの基本機能をテストします：
1. エージェント初期化
2. 質問処理
3. エピソード統合
4. グラフメトリクス計算
5. メモリ管理
"""

import sys
import traceback
from pathlib import Path

import numpy as np

# パス設定
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_basic_functionality():
    """基本機能の完全テスト"""

    print("🚀 InsightSpike-AI 基本機能テスト開始")
    print("=" * 50)

    try:
        # 1. 依存関係インポート
        print("📦 依存関係をインポート中...")
        from insightspike.core.agents.main_agent import MainAgent
        from insightspike.core.config import get_config
        from insightspike.utils.graph_metrics import delta_ged, delta_ig

        print("✅ 依存関係インポート成功")

        # 2. 設定確認
        print("\n⚙️  設定を確認中...")
        config = get_config()
        print(f"✅ 設定読み込み成功 - LLMプロバイダー: {config.llm.provider}")

        # 3. エージェント初期化
        print("\n🤖 エージェントを初期化中...")
        agent = MainAgent()
        print("✅ エージェント作成成功")

        # 4. 初期メモリ状態確認
        print("\n🧠 初期メモリ状態:")
        print(f"   エピソード数: {len(agent.l2_memory.episodes)}")
        print(f"   メモリサイズ: {len(agent.l2_memory.episodes)}")

        # 5. テストデータでエピソード追加
        print("\n📝 テストエピソードを追加中...")

        test_episodes = [
            "Machine learning is revolutionizing healthcare by enabling early disease detection",
            "AI in healthcare helps doctors diagnose diseases faster and more accurately",
            "Deep learning algorithms can analyze medical images to detect cancer",
            "Quantum computing may solve complex optimization problems in the future",
            "Blockchain technology ensures secure and transparent data transactions",
        ]

        for i, episode_content in enumerate(test_episodes):
            print(f"   エピソード {i+1}: {episode_content[:50]}...")

            # エピソードを直接メモリに追加（ベクトル化が必要）
            # シンプルなダミーベクトルを作成
            dummy_vector = np.random.random(384).astype(np.float32)
            agent.l2_memory.add_episode(dummy_vector, episode_content)

        print("✅ テストエピソード追加完了")

        # 6. メモリ統合後の状態確認
        print("\n🔍 エピソード統合後のメモリ状態:")
        print(f"   エピソード数: {len(agent.l2_memory.episodes)}")
        print(f"   メモリサイズ: {len(agent.l2_memory.episodes)}")

        # 各エピソードの詳細表示
        for i, episode in enumerate(agent.l2_memory.episodes):
            # Handle both Episode objects and dict representations
            if hasattr(episode, "c"):
                c_value = episode.c
                text = episode.text
            elif isinstance(episode, dict):
                c_value = episode.get("c", 0.5)
                text = episode.get("text", episode.get("content", ""))
            else:
                c_value = 0.5
                text = str(episode)
            print(f"   Episode {i+1}: C-value={c_value:.3f}, length={len(text)}")

        # 7. グラフ構築とメトリクス計算テスト
        print("\n📊 グラフメトリクス計算テスト...")

        # テスト用ドキュメント
        docs_old = [
            {"content": "AI is powerful", "id": 1},
            {"content": "Machine learning helps", "id": 2},
        ]
        docs_new = [
            {"content": "AI is powerful", "id": 1},
            {"content": "Machine learning helps", "id": 2},
            {"content": "Deep learning advances", "id": 3},
        ]

        # グラフ構築
        graph_old = agent.l3_graph.graph_builder.build_graph(docs_old)
        graph_new = agent.l3_graph.graph_builder.build_graph(docs_new)

        # メトリクス計算
        ged_value = delta_ged(graph_old, graph_new)
        ig_value = delta_ig(graph_old, graph_new)

        print(f"✅ ΔGED: {ged_value:.3f}")
        print(f"✅ ΔIG: {ig_value:.3f}")

        # 8. メモリ管理機能テスト
        print("\n🔧 メモリ管理機能テスト...")

        # C-value更新テスト
        if agent.l2_memory.episodes:
            old_c_value = agent.l2_memory.episodes[0].c

            # update_c_value メソッドを使用
            agent.l2_memory.update_c_value(0, 0.5)  # 報酬を追加
            new_c_value = agent.l2_memory.episodes[0].c

            print(f"✅ C-value更新: {old_c_value:.3f} → {new_c_value:.3f}")

        # 統計情報テスト
        initial_count = len(agent.l2_memory.episodes)
        stats = agent.l2_memory.get_memory_stats()  # 正しいメソッド名
        final_count = len(agent.l2_memory.episodes)

        print(
            f"✅ 統計取得: エピソード数={stats['total_episodes']}, 平均C-value={stats.get('avg_c_value', 0):.3f}"
        )

        # 9. 統計サマリー
        print("\n📈 最終統計:")
        print(f"   総エピソード数: {len(agent.l2_memory.episodes)}")
        print(f"   メモリサイズ: {len(agent.l2_memory.episodes)}")
        print(
            f"   平均C-value: {sum(ep.c for ep in agent.l2_memory.episodes) / max(len(agent.l2_memory.episodes), 1):.3f}"
        )

        print("\n🎉 基本機能テスト完了！")
        print("=" * 50)
        print("✅ すべての機能が正常に動作しています")

        return True

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        print("詳細なエラー情報:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
