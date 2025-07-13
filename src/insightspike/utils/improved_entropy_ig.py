"""
Improved Entropy-based Information Gain Calculation
==================================================

構造化されたエントロピー計算でより正確なIG測定を実現
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class ImprovedEntropyIG:
    """
    改良版エントロピーベースIG計算

    主な改善点：
    1. クラスタ分布のエントロピー
    2. 主成分のエントロピー
    3. グラフ構造を考慮
    4. 埋め込みの意味的変化を捉える
    """

    def __init__(
        self, n_clusters: int = 5, n_components: int = 10, structure_weight: float = 0.3
    ):
        """
        Args:
            n_clusters: クラスタリング数
            n_components: PCA成分数
            structure_weight: 構造エントロピーの重み
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.structure_weight = structure_weight

    def calculate_ig(self, graph1: Any, graph2: Any) -> float:
        """
        改良版IG計算

        Args:
            graph1: 前の状態のグラフ
            graph2: 後の状態のグラフ

        Returns:
            float: 情報利得（正の値は情報増加）
        """
        try:
            # Extract embeddings
            emb1 = self._extract_embeddings(graph1)
            emb2 = self._extract_embeddings(graph2)

            if emb1 is None or emb2 is None:
                return 0.0

            # 1. Clustering-based entropy
            cluster_entropy1 = self._cluster_entropy(emb1)
            cluster_entropy2 = self._cluster_entropy(emb2)
            cluster_ig = cluster_entropy2 - cluster_entropy1

            # 2. PCA-based entropy
            pca_entropy1 = self._pca_entropy(emb1)
            pca_entropy2 = self._pca_entropy(emb2)
            pca_ig = pca_entropy2 - pca_entropy1

            # 3. Structure-based entropy (if graph structure available)
            struct_entropy1 = self._structure_entropy(graph1)
            struct_entropy2 = self._structure_entropy(graph2)
            struct_ig = struct_entropy2 - struct_entropy1

            # Combine different entropy measures
            total_ig = (1 - self.structure_weight) * (
                cluster_ig + pca_ig
            ) / 2 + self.structure_weight * struct_ig

            logger.debug(
                f"Improved IG: cluster={cluster_ig:.3f}, "
                f"pca={pca_ig:.3f}, struct={struct_ig:.3f}, "
                f"total={total_ig:.3f}"
            )

            return float(total_ig)

        except Exception as e:
            logger.error(f"Improved IG calculation failed: {e}")
            return 0.0

    def _extract_embeddings(self, graph: Any) -> Optional[np.ndarray]:
        """グラフから埋め込みを抽出"""
        if graph is None:
            return None

        if hasattr(graph, "x"):
            if isinstance(graph.x, torch.Tensor):
                return graph.x.cpu().numpy()
            else:
                return graph.x
        return None

    def _cluster_entropy(self, embeddings: np.ndarray) -> float:
        """クラスタ分布のエントロピー"""
        if len(embeddings) < self.n_clusters:
            # Not enough samples for clustering
            return 0.0

        try:
            # Perform clustering
            kmeans = KMeans(
                n_clusters=min(self.n_clusters, len(embeddings)), random_state=42
            )
            labels = kmeans.fit_predict(embeddings)

            # Calculate cluster distribution
            unique, counts = np.unique(labels, return_counts=True)
            probs = counts / len(labels)

            # Shannon entropy of cluster distribution
            return float(scipy_entropy(probs, base=2))

        except Exception as e:
            logger.warning(f"Cluster entropy failed: {e}")
            return 0.0

    def _pca_entropy(self, embeddings: np.ndarray) -> float:
        """主成分のエントロピー"""
        if len(embeddings) < 2:
            return 0.0

        try:
            # Perform PCA
            n_comp = min(self.n_components, len(embeddings) - 1, embeddings.shape[1])
            pca = PCA(n_components=n_comp)
            transformed = pca.fit_transform(embeddings)

            # Use explained variance as probability distribution
            explained_var = pca.explained_variance_ratio_

            # Remove zero values and normalize
            explained_var = explained_var[explained_var > 0]
            if len(explained_var) == 0:
                return 0.0

            # Shannon entropy of variance distribution
            return float(scipy_entropy(explained_var, base=2))

        except Exception as e:
            logger.warning(f"PCA entropy failed: {e}")
            return 0.0

    def _structure_entropy(self, graph: Any) -> float:
        """グラフ構造のエントロピー"""
        if not hasattr(graph, "edge_index"):
            return 0.0

        try:
            # Calculate degree distribution
            if hasattr(graph, "num_nodes"):
                num_nodes = graph.num_nodes
            elif hasattr(graph, "x"):
                num_nodes = graph.x.shape[0]
            else:
                return 0.0

            if num_nodes == 0:
                return 0.0

            # Count degrees
            edge_index = graph.edge_index
            degrees = torch.zeros(num_nodes)

            if edge_index.shape[1] > 0:
                src = edge_index[0]
                for node in src:
                    degrees[node] += 1

            # Convert to probability distribution
            degree_counts = torch.bincount(degrees.long())
            degree_probs = degree_counts.float() / degree_counts.sum()

            # Remove zeros
            degree_probs = degree_probs[degree_probs > 0]

            if len(degree_probs) == 0:
                return 0.0

            # Calculate entropy
            return float(scipy_entropy(degree_probs.numpy(), base=2))

        except Exception as e:
            logger.warning(f"Structure entropy failed: {e}")
            return 0.0

    def normalize_ig(
        self, ig_value: float, graph1_size: int, graph2_size: int
    ) -> float:
        """
        IGを正規化（サイズ変化を考慮）

        Args:
            ig_value: 計算されたIG値
            graph1_size: 前のグラフのノード数
            graph2_size: 後のグラフのノード数

        Returns:
            float: 正規化されたIG値
        """
        # Size change factor
        size_factor = np.log(max(graph2_size, 1)) - np.log(max(graph1_size, 1))

        # Normalize by size change
        if abs(size_factor) > 0.1:
            normalized = ig_value / abs(size_factor)
        else:
            normalized = ig_value

        return float(normalized)
