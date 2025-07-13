"""
Graph-Centric Main Agent (C値なし)
==================================

グラフ構造を中心とした新しいエージェント実装。
"""

import logging
from typing import Any, Dict, List, Optional

from ..layers.layer2_graph_centric import GraphCentricMemoryManager
from .main_agent import MainAgent

logger = logging.getLogger(__name__)


class GraphCentricMainAgent(MainAgent):
    """
    グラフ中心のメインエージェント

    特徴:
    - C値を完全に削除
    - グラフ構造から動的に重要度を計算
    - よりシンプルで理解しやすい設計
    """

    def __init__(self):
        # 親クラスを初期化せず、独自に構築
        self.l1_perception = None
        self.l2_memory = None
        self.l3_graph = None
        self.l4_narrative = None
        self.config = None
        self._is_initialized = False

        logger.info("Initializing Graph-Centric Main Agent (C-value free)")

    def initialize(self):
        """各レイヤーを初期化"""
        from ...config import get_config

        # Layer1は今回使用しないのでスキップ
        from ..layers.layer3_graph_reasoner import L3GraphReasoner
        from ..layers.layer4_narrative_generator import L4NarrativeGenerator

        self.config = get_config()

        # Layer 1: Skip for now

        # Layer 2: Graph-Centric Memory
        self.l2_memory = GraphCentricMemoryManager(dim=384)

        # Layer 3: Graph Reasoner
        self.l3_graph = L3GraphReasoner()
        self.l3_graph.initialize()

        # Layer 4: Narrative Generator
        self.l4_narrative = L4NarrativeGenerator()
        self.l4_narrative.initialize()

        # Layer間の接続
        self.l2_memory.set_layer3_graph(self.l3_graph)

        self._is_initialized = True
        logger.info("Graph-Centric Agent initialization complete")

        return True

    def add_episode(self, text: str) -> Dict[str, Any]:
        """
        エピソード追加（シンプル版）

        Args:
            text: 追加するテキスト

        Returns:
            結果情報
        """
        if not self._is_initialized:
            return {"success": False, "error": "Agent not initialized"}

        try:
            # テキストをエンコード
            from ...utils.embedder import get_model

            model = get_model()
            vector = model.encode(
                text, normalize_embeddings=True, convert_to_numpy=True
            )

            # エピソード追加（C値なし）
            episode_idx = self.l2_memory.add_episode(vector, text)

            if episode_idx < 0:
                return {"success": False, "error": "Failed to add episode"}

            # グラフ更新
            graph_analysis = self._update_graph()

            # 統計情報取得
            stats = self.l2_memory.get_stats()

            # エピソードの重要度を計算
            importance = self.l2_memory.get_importance(episode_idx)

            return {
                "success": True,
                "episode_idx": episode_idx,
                "importance": importance,
                "total_episodes": stats["total_episodes"],
                "integration_rate": stats["integration_rate"],
                "graph_analysis": graph_analysis,
                "stats": stats,
            }

        except Exception as e:
            logger.error(f"Error adding episode: {e}")
            return {"success": False, "error": str(e)}

    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        検索（重要度を考慮）

        Args:
            query: 検索クエリ
            k: 返す結果数

        Returns:
            検索結果
        """
        if not self._is_initialized:
            return {"success": False, "error": "Agent not initialized"}

        try:
            # 検索実行
            results = self.l2_memory.search_episodes(query, k)

            # ナラティブ生成
            if results:
                narrative = self.l4_narrative.generate_narrative(query, results[:3])
            else:
                narrative = "No relevant episodes found."

            return {
                "success": True,
                "results": results,
                "narrative": narrative,
                "query": query,
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"success": False, "error": str(e)}

    def get_episode_info(self, episode_idx: int) -> Dict[str, Any]:
        """
        エピソード詳細情報取得

        Args:
            episode_idx: エピソードインデックス

        Returns:
            エピソード情報
        """
        if not (0 <= episode_idx < len(self.l2_memory.episodes)):
            return {"error": "Invalid episode index"}

        episode = self.l2_memory.episodes[episode_idx]
        importance = self.l2_memory.get_importance(episode_idx)

        # グラフ情報取得
        graph_info = self._get_graph_info(episode_idx)

        return {
            "index": episode_idx,
            "text": episode.text,
            "importance": importance,
            "metadata": episode.metadata,
            "graph_connections": graph_info.get("connections", 0),
            "conflict_score": graph_info.get("conflict_score", 0.0),
            "access_count": episode.metadata.get("access_count", 0),
            "last_access": episode.metadata.get("last_access", "Never"),
        }

    def optimize_memory(self) -> Dict[str, Any]:
        """
        メモリ最適化（不要なエピソードの削除など）
        """
        try:
            initial_count = len(self.l2_memory.episodes)
            removed_count = 0

            # 重要度が低く、長期間アクセスされていないエピソードを識別
            import time

            current_time = time.time()
            candidates = []

            for i, episode in enumerate(self.l2_memory.episodes):
                importance = self.l2_memory.get_importance(i)
                last_access = episode.metadata.get("last_access", current_time)
                days_inactive = (current_time - last_access) / 86400

                # 重要度が低く、30日以上アクセスなし
                if importance < 0.1 and days_inactive > 30:
                    candidates.append((i, importance))

            # 重要度の低い順にソート
            candidates.sort(key=lambda x: x[1])

            # 最大10%まで削除
            max_remove = max(1, len(self.l2_memory.episodes) // 10)
            to_remove = [idx for idx, _ in candidates[:max_remove]]

            # 逆順で削除（インデックスのずれを防ぐ）
            for idx in reversed(to_remove):
                del self.l2_memory.episodes[idx]
                removed_count += 1

            # インデックス再構築
            if removed_count > 0:
                self.l2_memory._update_index()

            return {
                "success": True,
                "initial_count": initial_count,
                "removed_count": removed_count,
                "final_count": len(self.l2_memory.episodes),
                "optimization_rate": removed_count / initial_count
                if initial_count > 0
                else 0,
            }

        except Exception as e:
            logger.error(f"Memory optimization error: {e}")
            return {"success": False, "error": str(e)}

    def _update_graph(self) -> Dict[str, Any]:
        """グラフ更新"""
        if not self.l3_graph:
            return {}

        # 全エピソードからドキュメント作成
        documents = []
        for i, episode in enumerate(self.l2_memory.episodes):
            documents.append(
                {
                    "text": episode.text,
                    "embedding": episode.vec,
                    "episode_idx": i,
                    "importance": self.l2_memory.get_importance(i),
                }
            )

        # グラフ解析
        return self.l3_graph.analyze_documents(documents)

    def _get_graph_info(self, episode_idx: int) -> Dict[str, Any]:
        """エピソードのグラフ情報取得"""
        info = {"connections": 0, "conflict_score": 0.0}

        if self.l3_graph and hasattr(self.l3_graph, "previous_graph"):
            graph = self.l3_graph.previous_graph
            if graph and hasattr(graph, "edge_index"):
                edge_index = graph.edge_index
                connections = (edge_index[0] == episode_idx).sum() + (
                    edge_index[1] == episode_idx
                ).sum()
                info["connections"] = connections.item()

                # コンフリクトスコア
                info["conflict_score"] = self.l2_memory._calculate_conflict(episode_idx)

        return info

    def get_memory_analysis(self) -> Dict[str, Any]:
        """
        メモリ全体の分析
        """
        stats = self.l2_memory.get_stats()

        # 重要度分布
        importances = [
            self.l2_memory.get_importance(i)
            for i in range(len(self.l2_memory.episodes))
        ]

        # アクセス分布
        access_counts = [
            ep.metadata.get("access_count", 0) for ep in self.l2_memory.episodes
        ]

        # 統合回数分布
        integration_counts = [
            ep.metadata.get("integration_count", 0) for ep in self.l2_memory.episodes
        ]

        import numpy as np

        return {
            "total_episodes": stats["total_episodes"],
            "integration_rate": stats["integration_rate"],
            "graph_assist_rate": stats["graph_assist_rate"],
            "importance_distribution": {
                "mean": np.mean(importances) if importances else 0,
                "std": np.std(importances) if importances else 0,
                "max": max(importances) if importances else 0,
                "min": min(importances) if importances else 0,
            },
            "access_distribution": {
                "total": sum(access_counts),
                "mean": np.mean(access_counts) if access_counts else 0,
                "max": max(access_counts) if access_counts else 0,
            },
            "integration_distribution": {
                "episodes_with_integration": sum(
                    1 for x in integration_counts if x > 0
                ),
                "max_integrations": max(integration_counts)
                if integration_counts
                else 0,
            },
        }

    def configure(
        self,
        integration_config: Optional[Dict] = None,
        splitting_config: Optional[Dict] = None,
    ):
        """
        設定を更新

        Args:
            integration_config: 統合設定
            splitting_config: 分裂設定
        """
        if integration_config:
            for key, value in integration_config.items():
                if hasattr(self.l2_memory.integration_config, key):
                    setattr(self.l2_memory.integration_config, key, value)
            logger.info(f"Updated integration config: {integration_config}")

        if splitting_config:
            for key, value in splitting_config.items():
                if hasattr(self.l2_memory.splitting_config, key):
                    setattr(self.l2_memory.splitting_config, key, value)
            logger.info(f"Updated splitting config: {splitting_config}")
