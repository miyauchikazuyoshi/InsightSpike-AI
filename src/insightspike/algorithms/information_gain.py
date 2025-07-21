"""
Information Gain (IG) Algorithm Implementation
=============================================

Implementation of Information Gain calculation for InsightSpike-AI's geDIG technology.
This module provides the core ΔIG computation for detecting learning and insight moments.

Mathematical Foundation:
    IG(S, A) = H(S) - Σ(|Sv|/|S|) × H(Sv)
    
    Where H(S) = -Σ p(x) log₂ p(x) is Shannon entropy
    
    ΔIG = H(before) - H(after) = entropy_before - entropy_after

Key Insight Detection:
    - Positive ΔIG values indicate information gain (entropy reduction)
    - ΔIG ≥ 0.2 threshold typically indicates EurekaSpike
    - Combined with ΔGED ≤ -0.5 for full geDIG detection

References:
    - Shannon, C. E. (1948). A mathematical theory of communication.
    - Quinlan, J. R. (1986). Induction of decision trees.
    - Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Mock implementations for graceful fallback
    KMeans = None
    silhouette_score = None
    mutual_info_classif = None

# Import structural entropy measures
try:
    from .structural_entropy import degree_distribution_entropy
    from .structural_entropy import structural_entropy as calc_structural_entropy
    from .structural_entropy import von_neumann_entropy

    STRUCTURAL_ENTROPY_AVAILABLE = True
except ImportError:
    STRUCTURAL_ENTROPY_AVAILABLE = False
    degree_distribution_entropy = None
    von_neumann_entropy = None
    calc_structural_entropy = None

logger = logging.getLogger(__name__)

__all__ = [
    "InformationGain",
    "EntropyMethod",
    "IGResult",
    "compute_shannon_entropy",
    "compute_information_gain",
    "compute_delta_ig",
]


class EntropyMethod(Enum):
    """Methods for calculating entropy."""

    SHANNON = "shannon"  # Classic Shannon entropy
    CLUSTERING = "clustering"  # Clustering-based entropy using silhouette score
    MUTUAL_INFO = "mutual_info"  # Mutual information-based
    FEATURE_ENTROPY = "feature_entropy"  # Feature distribution entropy
    STRUCTURAL = "structural"  # Graph structural entropy
    DEGREE_DISTRIBUTION = "degree_distribution"  # Degree distribution entropy
    VON_NEUMANN = "von_neumann"  # Von Neumann spectral entropy


@dataclass
class IGResult:
    """Result of Information Gain calculation with metadata."""

    ig_value: float
    entropy_before: float
    entropy_after: float
    computation_time: float
    method: EntropyMethod
    sample_count: int
    feature_count: int
    approximation_used: bool = False

    @property
    def information_gain_rate(self) -> float:
        """Relative information gain rate."""
        if self.entropy_before == 0:
            return 0.0
        return self.ig_value / self.entropy_before

    @property
    def is_significant(self) -> bool:
        """Check if information gain is statistically significant."""
        return self.ig_value > 0.1 and self.sample_count >= 10


class InformationGain:
    """
    Information Gain calculator for measuring learning progress.

    This implementation supports multiple entropy calculation methods:
    - SHANNON: Classic Shannon entropy for categorical data
    - CLUSTERING: Clustering-based entropy for vector data
    - MUTUAL_INFO: Mutual information between features
    - FEATURE_ENTROPY: Entropy of feature distributions
    """

    def __init__(
        self,
        method: Union[str, EntropyMethod] = EntropyMethod.CLUSTERING,
        k_clusters: int = 8,
        min_samples: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize Information Gain calculator.

        Args:
            method: Entropy calculation method
            k_clusters: Number of clusters for clustering-based methods
            min_samples: Minimum samples required for reliable calculation
            random_state: Random seed for reproducibility
        """
        if isinstance(method, str):
            method = EntropyMethod(method.lower())
        self.method = method

        self.k_clusters = k_clusters
        self.min_samples = min_samples
        self.random_state = random_state

        # Statistics tracking
        self.calculation_count = 0
        self.total_computation_time = 0.0
        self.approximation_count = 0

        logger.info(
            f"IG Calculator initialized: {method.value} method, "
            f"k_clusters={k_clusters}, min_samples={min_samples}"
        )

    def calculate_from_vectors(
        self, vectors: np.ndarray, method: str = "clustering", k: int = None
    ) -> float:
        """
        Calculate entropy from vector embeddings.

        Args:
            vectors: Input vector embeddings
            method: Calculation method ('clustering', 'pca', 'combined')
            k: Number of clusters (overrides self.k_clusters if provided)

        Returns:
            float: Entropy value
        """
        if vectors is None or len(vectors) == 0:
            return 0.0

        k = k or self.k_clusters

        if method == "clustering":
            return self._clustering_entropy(vectors)
        elif method == "pca":
            return ImprovedEntropyMethods.pca_entropy(vectors)
        elif method == "combined":
            # For combined, we need two sets of vectors, so just use cluster entropy
            return self._clustering_entropy(vectors)
        else:
            # Default to clustering method
            return self._clustering_entropy(vectors)

    def calculate(self, data_before: Any, data_after: Any) -> IGResult:
        """
        Calculate Information Gain between two data states.

        Args:
            data_before: Initial data state (vectors, labels, or distributions)
            data_after: Final data state (vectors, labels, or distributions)

        Returns:
            IGResult: Detailed calculation result
        """
        start_time = time.time()
        self.calculation_count += 1

        try:
            # Calculate entropy for both states
            entropy_before = self._calculate_entropy(data_before)
            entropy_after = self._calculate_entropy(data_after)

            # Information gain is the reduction in entropy
            # Positive IG means entropy decreased (gained information)
            ig_value = entropy_before - entropy_after

            computation_time = time.time() - start_time
            self.total_computation_time += computation_time

            # Determine data characteristics
            sample_count = self._get_sample_count(data_before, data_after)
            feature_count = self._get_feature_count(data_before, data_after)

            result = IGResult(
                ig_value=ig_value,
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                computation_time=computation_time,
                method=self.method,
                sample_count=sample_count,
                feature_count=feature_count,
            )

            logger.debug(
                f"IG calculation completed: {ig_value:.3f} "
                f"(before: {entropy_before:.3f}, after: {entropy_after:.3f})"
            )

            return result

        except Exception as e:
            computation_time = time.time() - start_time
            logger.error(f"IG calculation failed: {e}")

            # Return fallback result
            return IGResult(
                ig_value=0.0,
                entropy_before=0.0,
                entropy_after=0.0,
                computation_time=computation_time,
                method=self.method,
                sample_count=1,
                feature_count=1,
                approximation_used=True,
            )

    def compute_delta_ig(self, state_before: Any, state_after: Any) -> float:
        """
        Compute ΔIG for insight detection.

        Args:
            state_before: Initial state representation
            state_after: Final state representation

        Returns:
            float: ΔIG value (positive indicates information gain/entropy reduction)
        """
        result = self.calculate(state_before, state_after)
        logger.debug(f"ΔIG calculated: {result.ig_value:.3f}")
        return result.ig_value

    def _calculate_entropy(self, data: Any) -> float:
        """Calculate entropy based on selected method."""
        if data is None:
            return 0.0

        if self.method == EntropyMethod.SHANNON:
            return self._shannon_entropy(data)
        elif self.method == EntropyMethod.CLUSTERING:
            return self._clustering_entropy(data)
        elif self.method == EntropyMethod.MUTUAL_INFO:
            return self._mutual_info_entropy(data)
        elif self.method == EntropyMethod.FEATURE_ENTROPY:
            return self._feature_entropy(data)
        elif self.method == EntropyMethod.STRUCTURAL:
            return self._structural_entropy(data)
        elif self.method == EntropyMethod.DEGREE_DISTRIBUTION:
            return self._degree_distribution_entropy(data)
        elif self.method == EntropyMethod.VON_NEUMANN:
            return self._von_neumann_entropy(data)
        else:
            logger.warning(f"Unknown method {self.method}, using Shannon")
            return self._shannon_entropy(data)

    def _shannon_entropy(self, data: Any) -> float:
        """Calculate Shannon entropy for categorical data."""
        try:
            # Handle different data types
            if isinstance(data, np.ndarray) and data.ndim > 1:
                # For matrix data, calculate entropy of flattened distribution
                data = data.flatten()

            if hasattr(data, "__iter__") and not isinstance(data, str):
                # Convert to list for consistent handling
                data = list(data)
            else:
                data = [data]

            # Calculate probabilities
            counter = Counter(data)
            total_count = len(data)

            if total_count <= 1:
                return 0.0

            # Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
            entropy = 0.0
            for count in counter.values():
                if count > 0:
                    p = count / total_count
                    entropy -= p * np.log2(p)

            return float(entropy)

        except Exception as e:
            logger.warning(f"Shannon entropy calculation failed: {e}")
            return 0.0

    def _clustering_entropy(self, data: Any) -> float:
        """Calculate entropy using similarity-based approach."""
        try:
            # Convert data to numpy array
            if hasattr(data, "numpy"):  # PyTorch tensor
                vectors = data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                vectors = data
            elif hasattr(data, "__iter__"):
                vectors = np.array(list(data))
            else:
                return 0.0

            # Ensure 2D array
            if vectors.ndim == 1:
                vectors = vectors.reshape(-1, 1)

            n_samples = vectors.shape[0]

            if n_samples < self.min_samples:
                return 0.0

            # Import and use improved similarity-based entropy with sigmoid normalization
            try:
                from .improved_similarity_entropy import (
                    NormalizationMethod,
                    calculate_similarity_entropy,
                )

                # Use sigmoid normalization for better sensitivity
                return calculate_similarity_entropy(
                    vectors, method=NormalizationMethod.SIGMOID, steepness=5.0
                )
            except ImportError:
                # Fallback to original implementation
                from .similarity_entropy import calculate_similarity_entropy

                return calculate_similarity_entropy(vectors)

        except Exception as e:
            logger.warning(f"Clustering entropy calculation failed: {e}")
            return self._shannon_entropy(data)

    def _mutual_info_entropy(self, data: Any) -> float:
        """Calculate entropy using mutual information."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Sklearn unavailable, using Shannon entropy fallback")
            return self._shannon_entropy(data)

        try:
            # This is a simplified version - in practice, you'd need feature/target separation
            if isinstance(data, np.ndarray) and data.ndim == 2:
                # Use first feature as target, rest as features
                if data.shape[1] > 1:
                    X = data[:, 1:]
                    y = data[:, 0]

                    # Discretize continuous target if needed
                    if len(np.unique(y)) > 10:
                        y = np.digitize(y, np.percentile(y, [25, 50, 75]))

                    mi = mutual_info_classif(X, y, random_state=self.random_state)
                    return float(np.mean(mi))
                else:
                    return self._shannon_entropy(data)
            else:
                return self._shannon_entropy(data)

        except Exception as e:
            logger.warning(f"Mutual info entropy calculation failed: {e}")
            return self._shannon_entropy(data)

    def _feature_entropy(self, data: Any) -> float:
        """Calculate entropy of feature distributions."""
        try:
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    return self._shannon_entropy(data)

                # Calculate entropy for each feature dimension
                total_entropy = 0.0
                for i in range(data.shape[1]):
                    feature_column = data[:, i]

                    # Discretize continuous features into bins
                    if len(np.unique(feature_column)) > 10:
                        hist, _ = np.histogram(feature_column, bins=10)
                        probs = hist / np.sum(hist + 1e-10)
                    else:
                        # Already discrete
                        counter = Counter(feature_column)
                        total = sum(counter.values())
                        probs = np.array([count / total for count in counter.values()])

                    # Calculate Shannon entropy for this feature
                    probs = probs[probs > 0]  # Remove zero probabilities
                    if len(probs) > 1:
                        feature_entropy = -np.sum(probs * np.log2(probs))
                        total_entropy += feature_entropy

                return float(total_entropy)
            else:
                return self._shannon_entropy(data)

        except Exception as e:
            logger.warning(f"Feature entropy calculation failed: {e}")
            return self._shannon_entropy(data)

    def _get_sample_count(self, data_before: Any, data_after: Any) -> int:
        """Get total sample count from both datasets."""
        try:
            count_before = len(data_before) if hasattr(data_before, "__len__") else 1
            count_after = len(data_after) if hasattr(data_after, "__len__") else 1
            return count_before + count_after
        except:
            return 2

    def _get_feature_count(self, data_before: Any, data_after: Any) -> int:
        """Get feature count from datasets."""
        try:
            for data in [data_before, data_after]:
                if hasattr(data, "shape") and len(data.shape) >= 2:
                    return data.shape[1]
                elif hasattr(data, "__iter__") and not isinstance(data, str):
                    first_item = next(iter(data), None)
                    if hasattr(first_item, "__len__"):
                        return len(first_item)
            return 1
        except:
            return 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator performance statistics."""
        avg_time = self.total_computation_time / max(self.calculation_count, 1)
        approximation_rate = self.approximation_count / max(self.calculation_count, 1)

        return {
            "total_calculations": self.calculation_count,
            "total_computation_time": self.total_computation_time,
            "average_computation_time": avg_time,
            "approximation_count": self.approximation_count,
            "approximation_rate": approximation_rate,
            "method": self.method.value,
            "k_clusters": self.k_clusters,
            "min_samples": self.min_samples,
        }

    def compute(self, data_before: Any, data_after: Any) -> float:
        """
        Compute Information Gain between two data states.
        Alias for calculate() method to maintain API consistency.

        Args:
            data_before: Initial data state
            data_after: Final data state

        Returns:
            float: Information gain value
        """
        return self.calculate(data_before, data_after).ig_value

    def _structural_entropy(self, data: Any) -> float:
        """Calculate combined structural entropy measures."""
        if not STRUCTURAL_ENTROPY_AVAILABLE:
            logger.warning("Structural entropy unavailable, using Shannon fallback")
            return self._shannon_entropy(data)

        try:
            measures = calc_structural_entropy(data)
            return float(measures.get("combined", 0.0))
        except Exception as e:
            logger.error(f"Structural entropy calculation failed: {e}")
            return self._shannon_entropy(data)

    def _degree_distribution_entropy(self, data: Any) -> float:
        """Calculate degree distribution entropy."""
        if not STRUCTURAL_ENTROPY_AVAILABLE:
            logger.warning(
                "Degree distribution entropy unavailable, using Shannon fallback"
            )
            return self._shannon_entropy(data)

        try:
            return float(degree_distribution_entropy(data))
        except Exception as e:
            logger.error(f"Degree distribution entropy calculation failed: {e}")
            return self._shannon_entropy(data)

    def _von_neumann_entropy(self, data: Any) -> float:
        """Calculate Von Neumann entropy."""
        if not STRUCTURAL_ENTROPY_AVAILABLE:
            logger.warning("Von Neumann entropy unavailable, using Shannon fallback")
            return self._shannon_entropy(data)

        try:
            return float(von_neumann_entropy(data))
        except Exception as e:
            logger.error(f"Von Neumann entropy calculation failed: {e}")
            return self._shannon_entropy(data)

    def calculate_from_vectors(
        self, vectors: np.ndarray, method: str = "clustering", k: int = None
    ) -> float:
        """Calculate entropy from vector embeddings.

        Args:
            vectors: Input vector embeddings
            method: Calculation method ('clustering', 'pca', 'combined')
            k: Number of clusters (overrides self.k_clusters if provided)

        Returns:
            float: Entropy value
        """
        if vectors is None or len(vectors) == 0:
            return 0.0

        k = k or self.k_clusters

        if method == "clustering":
            return ImprovedEntropyMethods.cluster_entropy(vectors, n_clusters=k)
        elif method == "pca":
            return ImprovedEntropyMethods.pca_entropy(vectors)
        elif method == "combined":
            # For combined, we need two sets of vectors, so just use cluster entropy
            return ImprovedEntropyMethods.cluster_entropy(vectors, n_clusters=k)
        else:
            # Default to clustering method
            return self._clustering_entropy(vectors)


# Convenience functions for external API
def compute_shannon_entropy(data: Any) -> float:
    """
    Compute Shannon entropy for data.

    Args:
        data: Input data (categorical or continuous)

    Returns:
        float: Shannon entropy value
    """
    calculator = InformationGain(method=EntropyMethod.SHANNON)
    return calculator._shannon_entropy(data)


def compute_information_gain(
    data_before: Any, data_after: Any, method: str = "clustering", **kwargs
) -> float:
    """
    Compute Information Gain between two data states.

    Args:
        data_before: Initial data state
        data_after: Final data state
        method: Calculation method ("shannon", "clustering", "mutual_info", "feature_entropy")
        **kwargs: Additional parameters for InformationGain constructor

    Returns:
        float: Information gain value
    """
    calculator = InformationGain(method=method, **kwargs)
    result = calculator.calculate(data_before, data_after)
    return result.ig_value


def compute_delta_ig(
    state_before: Any, state_after: Any, method: str = "clustering", **kwargs
) -> float:
    """
    Compute ΔIG for insight detection.

    Args:
        state_before: Initial state representation
        state_after: Final state representation
        method: Calculation method
        **kwargs: Additional parameters for InformationGain constructor

    Returns:
        float: ΔIG value (positive indicates information gain/entropy reduction)
    """
    calculator = InformationGain(method=method, **kwargs)
    return calculator.compute_delta_ig(state_before, state_after)


# Additional methods from entropy_ig.py for improved entropy calculation
class ImprovedEntropyMethods:
    """Advanced entropy calculation methods integrated from graph/metrics/entropy_ig.py."""

    @staticmethod
    def cluster_entropy(embeddings: np.ndarray, n_clusters: int = 5) -> float:
        """Calculate entropy of cluster distribution."""
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
        """Calculate entropy of PCA components."""
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
        """Calculate combined information gain using multiple entropy measures."""
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
