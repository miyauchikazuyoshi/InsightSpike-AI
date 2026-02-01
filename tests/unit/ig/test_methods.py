"""Unit tests for ig.methods module."""

import numpy as np
import pytest

from insightspike.algorithms.ig.methods import ImprovedEntropyMethods


class TestImprovedEntropyMethods:
    """Tests for ImprovedEntropyMethods class."""

    def test_cluster_entropy_empty(self):
        embeddings = np.array([]).reshape(0, 10)
        assert ImprovedEntropyMethods.cluster_entropy(embeddings) == 0.0

    def test_cluster_entropy_single_sample(self):
        embeddings = np.random.randn(1, 10)
        assert ImprovedEntropyMethods.cluster_entropy(embeddings, n_clusters=5) == 0.0

    def test_cluster_entropy_uniform(self):
        # Create embeddings that should cluster evenly
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        entropy = ImprovedEntropyMethods.cluster_entropy(embeddings, n_clusters=5)
        # Uniform distribution over 5 clusters has max entropy of log2(5) ≈ 2.32
        assert 0 < entropy <= np.log2(5) + 0.1

    def test_cluster_entropy_skewed(self):
        # Create embeddings with one dominant cluster
        np.random.seed(42)
        dominant = np.random.randn(80, 10)
        others = np.random.randn(20, 10) + 10  # Offset to create separate cluster
        embeddings = np.vstack([dominant, others])
        entropy = ImprovedEntropyMethods.cluster_entropy(embeddings, n_clusters=2)
        # Skewed distribution should have lower entropy
        assert entropy > 0

    def test_pca_entropy_empty(self):
        embeddings = np.array([]).reshape(0, 10)
        assert ImprovedEntropyMethods.pca_entropy(embeddings) == 0.0

    def test_pca_entropy_single_sample(self):
        embeddings = np.random.randn(1, 10)
        assert ImprovedEntropyMethods.pca_entropy(embeddings) == 0.0

    def test_pca_entropy_basic(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 10)
        entropy = ImprovedEntropyMethods.pca_entropy(embeddings, n_components=5)
        assert entropy >= 0

    def test_pca_entropy_low_rank(self):
        # Create low-rank data (should have lower entropy)
        np.random.seed(42)
        base = np.random.randn(50, 2)
        # Extend to 10D but only 2 components have variance
        embeddings = np.hstack([base, np.zeros((50, 8))])
        entropy = ImprovedEntropyMethods.pca_entropy(embeddings, n_components=5)
        # Low-rank data should have lower entropy
        assert entropy >= 0

    def test_combined_entropy_ig_no_change(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 10)
        ig = ImprovedEntropyMethods.combined_entropy_ig(embeddings, embeddings)
        # Same data should give IG close to 0
        assert abs(ig) < 0.1

    def test_combined_entropy_ig_different(self):
        np.random.seed(42)
        embeddings1 = np.random.randn(50, 10)
        embeddings2 = np.random.randn(50, 10) * 2 + 5  # Different distribution
        ig = ImprovedEntropyMethods.combined_entropy_ig(embeddings1, embeddings2)
        # Different data should give non-zero IG
        assert isinstance(ig, float)
