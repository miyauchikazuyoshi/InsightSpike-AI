"""Advanced entropy calculation methods for Information Gain.

This module provides improved entropy calculation methods including
clustering-based, PCA-based, and combined approaches.
"""

import logging

import numpy as np

try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None

logger = logging.getLogger(__name__)


class ImprovedEntropyMethods:
    """Advanced entropy calculation methods integrated from graph/metrics/entropy_ig.py."""

    @staticmethod
    def cluster_entropy(embeddings: np.ndarray, n_clusters: int = 5) -> float:
        """Calculate entropy of cluster distribution.

        Args:
            embeddings: Input embeddings array.
            n_clusters: Number of clusters for k-means.

        Returns:
            Shannon entropy of cluster distribution.
        """
        if not SKLEARN_AVAILABLE or len(embeddings) < n_clusters:
            return 0.0

        try:
            from scipy.stats import entropy as scipy_entropy

            # Perform clustering
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(embeddings)), random_state=42
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

    @staticmethod
    def pca_entropy(embeddings: np.ndarray, n_components: int = 10) -> float:
        """Calculate entropy of PCA components.

        Args:
            embeddings: Input embeddings array.
            n_components: Number of PCA components.

        Returns:
            Entropy based on explained variance ratio.
        """
        if len(embeddings) < 2:
            return 0.0

        try:
            from scipy.stats import entropy as scipy_entropy
            from sklearn.decomposition import PCA

            # Perform PCA
            n_comp = min(n_components, len(embeddings) - 1, embeddings.shape[1])
            pca = PCA(n_components=n_comp)
            pca.fit_transform(embeddings)

            # Use explained variance as probability distribution
            explained_var = pca.explained_variance_ratio_
            explained_var = explained_var[explained_var > 0]

            if len(explained_var) == 0:
                return 0.0

            # Normalize and calculate entropy
            explained_var = explained_var / np.sum(explained_var)
            return float(scipy_entropy(explained_var, base=2))

        except Exception as e:
            logger.warning(f"PCA entropy failed: {e}")
            return 0.0

    @staticmethod
    def combined_entropy_ig(
        embeddings1: np.ndarray, embeddings2: np.ndarray, structure_weight: float = 0.3
    ) -> float:
        """Calculate combined information gain using multiple entropy measures.

        Args:
            embeddings1: First embeddings array (before state).
            embeddings2: Second embeddings array (after state).
            structure_weight: Weight for structural component (unused, for API compat).

        Returns:
            Combined information gain value.
        """
        # Clustering-based entropy
        cluster_entropy1 = ImprovedEntropyMethods.cluster_entropy(embeddings1)
        cluster_entropy2 = ImprovedEntropyMethods.cluster_entropy(embeddings2)
        cluster_ig = cluster_entropy2 - cluster_entropy1

        # PCA-based entropy
        pca_entropy1 = ImprovedEntropyMethods.pca_entropy(embeddings1)
        pca_entropy2 = ImprovedEntropyMethods.pca_entropy(embeddings2)
        pca_ig = pca_entropy2 - pca_entropy1

        # Combine measures
        total_ig = (cluster_ig + pca_ig) / 2

        logger.debug(
            f"Combined IG: cluster={cluster_ig:.3f}, pca={pca_ig:.3f}, total={total_ig:.3f}"
        )

        return float(total_ig)


__all__ = ["ImprovedEntropyMethods"]
