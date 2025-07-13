"""
Hierarchical Graph Builder - Phase 3 Implementation
==================================================

階層的グラフ構造で10万以上のエピソードを効率的に管理。
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import faiss
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class GraphLevel:
    """グラフ階層の1レベル"""

    level: int
    nodes: List[Dict[str, Any]]  # ノード情報
    graph: Optional[Data] = None  # PyG graph
    index: Optional[faiss.Index] = None  # FAISS index

    def __repr__(self):
        return f"GraphLevel(level={self.level}, nodes={len(self.nodes)})"


class HierarchicalGraphBuilder:
    """
    階層的グラフ構造ビルダー

    3層構造:
    - Level 0: 個別エピソード（葉ノード）
    - Level 1: クラスタ（中間ノード）
    - Level 2: スーパークラスタ（ルート付近）

    特徴:
    - 各レベルで独立したFAISSインデックス
    - 階層間の接続を保持
    - 効率的な検索とナビゲーション
    """

    def __init__(
        self,
        dimension: int = 384,
        cluster_size: int = 100,  # Level 0 → Level 1
        super_cluster_size: int = 100,  # Level 1 → Level 2
        similarity_threshold: float = 0.3,
        top_k: int = 50,
    ):
        """
        Args:
            dimension: ベクトル次元数
            cluster_size: 基本クラスタサイズ
            super_cluster_size: スーパークラスタサイズ
            similarity_threshold: エッジ作成閾値
            top_k: 近傍探索数
        """
        self.dimension = dimension
        self.cluster_size = cluster_size
        self.super_cluster_size = super_cluster_size
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        # 階層構造
        self.levels: List[GraphLevel] = [
            GraphLevel(level=0, nodes=[]),  # エピソード
            GraphLevel(level=1, nodes=[]),  # クラスタ
            GraphLevel(level=2, nodes=[]),  # スーパークラスタ
        ]

        # 階層間マッピング
        self.child_to_parent: Dict[
            Tuple[int, int], Tuple[int, int]
        ] = {}  # (level, idx) -> (parent_level, parent_idx)
        self.parent_to_children: Dict[
            Tuple[int, int], List[Tuple[int, int]]
        ] = defaultdict(list)

        # 統計
        self.stats = {
            "total_nodes": [0, 0, 0],
            "total_edges": [0, 0, 0],
            "build_times": [],
        }

        logger.info(
            f"Hierarchical Graph Builder initialized: "
            f"cluster_size={cluster_size}, super_cluster_size={super_cluster_size}"
        )

    def build_hierarchical_graph(
        self, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        階層的グラフを構築

        Args:
            documents: List of dicts with 'embedding', 'text', etc.

        Returns:
            構築結果の情報
        """
        import time

        start_time = time.time()

        # Level 0: 個別エピソード
        self._build_level_0(documents)

        # Level 1: クラスタ作成
        self._build_level_1()

        # Level 2: スーパークラスタ作成
        self._build_level_2()

        # 各レベルのグラフ構築
        for level in self.levels:
            if level.nodes:
                level.graph = self._build_graph_for_level(level)
                self.stats["total_nodes"][level.level] = len(level.nodes)
                self.stats["total_edges"][level.level] = (
                    level.graph.edge_index.size(1) if level.graph else 0
                )

        build_time = time.time() - start_time
        self.stats["build_times"].append(build_time)

        logger.info(f"Hierarchical graph built in {build_time:.2f}s")
        logger.info(f"Nodes per level: {self.stats['total_nodes']}")
        logger.info(f"Edges per level: {self.stats['total_edges']}")

        return {
            "levels": len(self.levels),
            "nodes_per_level": self.stats["total_nodes"],
            "edges_per_level": self.stats["total_edges"],
            "build_time": build_time,
            "total_nodes": sum(self.stats["total_nodes"]),
            "compression_ratio": len(documents) / max(1, self.stats["total_nodes"][2]),
        }

    def _build_level_0(self, documents: List[Dict[str, Any]]):
        """Level 0: エピソードレベル"""
        self.levels[0].nodes = []

        for i, doc in enumerate(documents):
            node = {
                "id": i,
                "embedding": doc["embedding"],
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            }
            self.levels[0].nodes.append(node)

        # FAISSインデックス構築
        if self.levels[0].nodes:
            embeddings = np.array(
                [n["embedding"] for n in self.levels[0].nodes], dtype=np.float32
            )
            self.levels[0].index = faiss.IndexFlatIP(self.dimension)
            self.levels[0].index.add(embeddings)

    def _build_level_1(self):
        """Level 1: クラスタレベル"""
        if not self.levels[0].nodes:
            return

        # K-meansクラスタリング
        embeddings = np.array(
            [n["embedding"] for n in self.levels[0].nodes], dtype=np.float32
        )
        n_clusters = max(1, len(self.levels[0].nodes) // self.cluster_size)

        # FAISSでクラスタリング
        kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=20, verbose=False)
        kmeans.train(embeddings)

        # クラスタ割り当て
        _, cluster_ids = kmeans.index.search(embeddings, 1)
        cluster_ids = cluster_ids.flatten()

        # クラスタセンターをLevel 1ノードとして作成
        self.levels[1].nodes = []
        for i in range(n_clusters):
            cluster_members = np.where(cluster_ids == i)[0]
            if len(cluster_members) > 0:
                # クラスタ中心
                center = kmeans.centroids[i]

                # メンバーのテキストを要約（簡易版）
                member_texts = [
                    self.levels[0].nodes[idx]["text"][:50]
                    for idx in cluster_members[:3]
                ]
                summary = f"Cluster {i}: {len(cluster_members)} episodes. Samples: {'; '.join(member_texts)}..."

                node = {
                    "id": i,
                    "embedding": center,
                    "text": summary,
                    "metadata": {
                        "level": 1,
                        "size": len(cluster_members),
                        "members": cluster_members.tolist(),
                    },
                }
                self.levels[1].nodes.append(node)

                # 階層マッピング更新
                for member_idx in cluster_members:
                    self.child_to_parent[(0, member_idx)] = (1, i)
                    self.parent_to_children[(1, i)].append((0, member_idx))

        # Level 1のFAISSインデックス
        if self.levels[1].nodes:
            l1_embeddings = np.array(
                [n["embedding"] for n in self.levels[1].nodes], dtype=np.float32
            )
            self.levels[1].index = faiss.IndexFlatIP(self.dimension)
            self.levels[1].index.add(l1_embeddings)

    def _build_level_2(self):
        """Level 2: スーパークラスタレベル"""
        if len(self.levels[1].nodes) <= self.super_cluster_size:
            # Level 1が小さい場合は全体を1つのスーパークラスタに
            if self.levels[1].nodes:
                all_embeddings = np.array(
                    [n["embedding"] for n in self.levels[1].nodes], dtype=np.float32
                )
                super_center = np.mean(all_embeddings, axis=0)

                node = {
                    "id": 0,
                    "embedding": super_center,
                    "text": f"Root: {len(self.levels[1].nodes)} clusters, {len(self.levels[0].nodes)} total episodes",
                    "metadata": {
                        "level": 2,
                        "total_clusters": len(self.levels[1].nodes),
                        "total_episodes": len(self.levels[0].nodes),
                    },
                }
                self.levels[2].nodes = [node]

                # マッピング
                for i, l1_node in enumerate(self.levels[1].nodes):
                    self.child_to_parent[(1, i)] = (2, 0)
                    self.parent_to_children[(2, 0)].append((1, i))
        else:
            # Level 1をさらにクラスタリング
            embeddings = np.array(
                [n["embedding"] for n in self.levels[1].nodes], dtype=np.float32
            )
            n_super_clusters = max(
                1, len(self.levels[1].nodes) // self.super_cluster_size
            )

            kmeans = faiss.Kmeans(
                self.dimension, n_super_clusters, niter=20, verbose=False
            )
            kmeans.train(embeddings)

            _, cluster_ids = kmeans.index.search(embeddings, 1)
            cluster_ids = cluster_ids.flatten()

            self.levels[2].nodes = []
            for i in range(n_super_clusters):
                cluster_members = np.where(cluster_ids == i)[0]
                if len(cluster_members) > 0:
                    center = kmeans.centroids[i]

                    # 統計情報
                    total_episodes = sum(
                        len(self.levels[1].nodes[idx]["metadata"]["members"])
                        for idx in cluster_members
                    )

                    node = {
                        "id": i,
                        "embedding": center,
                        "text": f"SuperCluster {i}: {len(cluster_members)} clusters, {total_episodes} episodes",
                        "metadata": {
                            "level": 2,
                            "clusters": len(cluster_members),
                            "total_episodes": total_episodes,
                        },
                    }
                    self.levels[2].nodes.append(node)

                    # マッピング
                    for member_idx in cluster_members:
                        self.child_to_parent[(1, member_idx)] = (2, i)
                        self.parent_to_children[(2, i)].append((1, member_idx))

        # Level 2のFAISSインデックス
        if self.levels[2].nodes:
            l2_embeddings = np.array(
                [n["embedding"] for n in self.levels[2].nodes], dtype=np.float32
            )
            self.levels[2].index = faiss.IndexFlatIP(self.dimension)
            self.levels[2].index.add(l2_embeddings)

    def _build_graph_for_level(self, level: GraphLevel) -> Data:
        """各レベルのグラフ構築"""
        if not level.nodes:
            return Data(
                x=torch.empty(0, self.dimension),
                edge_index=torch.empty(2, 0, dtype=torch.long),
            )

        embeddings = np.array([n["embedding"] for n in level.nodes], dtype=np.float32)

        # エッジ構築（効率的なtop-k近傍）
        edge_list = []

        if level.index is not None and len(level.nodes) > 1:
            k = min(self.top_k, len(level.nodes))

            # バッチ処理で高速化
            batch_size = 1000
            for i in range(0, len(level.nodes), batch_size):
                batch_end = min(i + batch_size, len(level.nodes))
                batch_embeddings = embeddings[i:batch_end]

                distances, neighbors = level.index.search(batch_embeddings, k)

                for j, (dist_row, neigh_row) in enumerate(zip(distances, neighbors)):
                    src_idx = i + j
                    for dist, dst_idx in zip(dist_row[1:], neigh_row[1:]):  # Skip self
                        if dist > self.similarity_threshold and dst_idx < len(
                            level.nodes
                        ):
                            edge_list.append([src_idx, dst_idx])

        # PyGグラフ作成
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)

        x = torch.tensor(embeddings, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, num_nodes=len(level.nodes))

    def search_hierarchical(
        self, query_vector: np.ndarray, k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        階層的検索

        1. Level 2で関連スーパークラスタを特定
        2. Level 1で関連クラスタを特定
        3. Level 0で最終的なエピソードを取得
        """
        results = []
        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Level 2検索
        if self.levels[2].index and self.levels[2].nodes:
            k2 = min(3, len(self.levels[2].nodes))  # 最大3つのスーパークラスタ
            distances2, indices2 = self.levels[2].index.search(query_vector, k2)

            # Level 1検索
            level1_candidates = []
            for idx in indices2[0]:
                if idx < len(self.levels[2].nodes):
                    # このスーパークラスタの子クラスタを取得
                    children = self.parent_to_children.get((2, idx), [])
                    level1_candidates.extend(
                        [child[1] for child in children if child[0] == 1]
                    )

            # Level 1候補から検索
            if level1_candidates and self.levels[1].index:
                candidate_embeddings = np.array(
                    [self.levels[1].nodes[i]["embedding"] for i in level1_candidates],
                    dtype=np.float32,
                )

                # 一時インデックスで検索
                temp_index = faiss.IndexFlatIP(self.dimension)
                temp_index.add(candidate_embeddings)

                k1 = min(10, len(candidate_embeddings))
                distances1, indices1 = temp_index.search(query_vector, k1)

                # Level 0検索
                level0_candidates = []
                for local_idx in indices1[0]:
                    if local_idx < len(level1_candidates):
                        global_idx = level1_candidates[local_idx]
                        children = self.parent_to_children.get((1, global_idx), [])
                        level0_candidates.extend(
                            [child[1] for child in children if child[0] == 0]
                        )

                # 最終的なエピソード検索
                if level0_candidates:
                    candidate_embeddings = np.array(
                        [
                            self.levels[0].nodes[i]["embedding"]
                            for i in level0_candidates
                        ],
                        dtype=np.float32,
                    )

                    temp_index = faiss.IndexFlatIP(self.dimension)
                    temp_index.add(candidate_embeddings)

                    k0 = min(k, len(candidate_embeddings))
                    distances0, indices0 = temp_index.search(query_vector, k0)

                    for local_idx, dist in zip(indices0[0], distances0[0]):
                        if local_idx < len(level0_candidates):
                            global_idx = level0_candidates[local_idx]
                            node = self.levels[0].nodes[global_idx]
                            results.append(
                                {
                                    "index": global_idx,
                                    "text": node["text"],
                                    "similarity": float(dist),
                                    "level": 0,
                                    "path": self._get_path_to_root(0, global_idx),
                                }
                            )

        return results[:k]

    def _get_path_to_root(self, level: int, idx: int) -> List[Tuple[int, int]]:
        """ノードからルートへのパスを取得"""
        path = [(level, idx)]
        current = (level, idx)

        while current in self.child_to_parent:
            parent = self.child_to_parent[current]
            path.append(parent)
            current = parent

        return path

    def add_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        新しいドキュメントを階層構造に追加

        動的な再構築を最小限に抑えながら追加
        """
        # Level 0に追加
        new_idx = len(self.levels[0].nodes)
        node = {
            "id": new_idx,
            "embedding": document["embedding"],
            "text": document.get("text", ""),
            "metadata": document.get("metadata", {}),
        }
        self.levels[0].nodes.append(node)

        # FAISSインデックス更新
        if self.levels[0].index:
            self.levels[0].index.add(np.array([node["embedding"]], dtype=np.float32))

        # 最も近いクラスタを見つける
        if self.levels[1].index and self.levels[1].nodes:
            query = np.array([node["embedding"]], dtype=np.float32)
            distances, indices = self.levels[1].index.search(query, 1)

            if indices[0][0] < len(self.levels[1].nodes):
                cluster_idx = indices[0][0]

                # クラスタに追加
                self.child_to_parent[(0, new_idx)] = (1, cluster_idx)
                self.parent_to_children[(1, cluster_idx)].append((0, new_idx))

                # クラスタのメタデータ更新
                self.levels[1].nodes[cluster_idx]["metadata"]["size"] += 1
                self.levels[1].nodes[cluster_idx]["metadata"]["members"].append(new_idx)

                # クラスタが大きくなりすぎたら分割を検討
                if (
                    self.levels[1].nodes[cluster_idx]["metadata"]["size"]
                    > self.cluster_size * 2
                ):
                    logger.info(
                        f"Cluster {cluster_idx} is too large, consider rebalancing"
                    )

        return {
            "success": True,
            "level_0_idx": new_idx,
            "cluster_assigned": cluster_idx if "cluster_idx" in locals() else None,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """階層構造の統計情報"""
        stats = {
            "levels": len(self.levels),
            "nodes_per_level": self.stats["total_nodes"],
            "edges_per_level": self.stats["total_edges"],
            "compression_ratios": [],
        }

        # 圧縮率計算
        for i in range(len(self.levels) - 1):
            if self.stats["total_nodes"][i] > 0:
                ratio = self.stats["total_nodes"][i] / max(
                    1, self.stats["total_nodes"][i + 1]
                )
                stats["compression_ratios"].append(ratio)

        # クラスタサイズ分布
        if self.levels[1].nodes:
            cluster_sizes = [node["metadata"]["size"] for node in self.levels[1].nodes]
            stats["cluster_size_distribution"] = {
                "mean": np.mean(cluster_sizes),
                "std": np.std(cluster_sizes),
                "min": min(cluster_sizes),
                "max": max(cluster_sizes),
            }

        return stats
