"""
Hybrid Metrics Calculation
=========================

複数のアプローチを組み合わせた統合的なメトリクス計算
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HybridMetrics:
    """
    ハイブリッドメトリクス計算

    3つのアプローチを統合：
    1. 実際の埋め込みの質を考慮
    2. 構造化されたエントロピー計算
    3. NetworkX構造変化 + 改良エントロピー意味変化
    """

    def __init__(self, config=None):
        self.config = config

        # Component weights
        self.weights = {
            "structure": 0.4,  # グラフ構造の変化
            "semantic": 0.4,  # 意味的変化
            "quality": 0.2,  # 埋め込みの質
        }

        # Initialize components
        self._init_components()

    def _init_components(self):
        """コンポーネントの初期化"""
        # NetworkX for structure
        try:
            from ..algorithms.graph_edit_distance import GraphEditDistance

            self.ged_calculator = GraphEditDistance(optimization_level="standard")
            self.has_networkx = True
        except:
            self.has_networkx = False

        # Improved entropy for semantics
        try:
            from .improved_entropy_ig import ImprovedEntropyIG

            self.entropy_calculator = ImprovedEntropyIG()
            self.has_improved_entropy = True
        except:
            self.has_improved_entropy = False

    def calculate_hybrid_metrics(
        self, graph1: Any, graph2: Any, query: Optional[str] = None
    ) -> Dict[str, float]:
        """
        ハイブリッドメトリクスを計算

        Returns:
            Dict containing:
            - hybrid_ged: 統合GED
            - hybrid_ig: 統合IG
            - structure_score: 構造変化スコア
            - semantic_score: 意味変化スコア
            - quality_score: 埋め込み品質スコア
            - spike_confidence: スパイク検出の信頼度
        """
        results = {}

        # 1. Structure analysis (NetworkX GED)
        structure_ged, structure_ig = self._analyze_structure(graph1, graph2)
        results["structure_ged"] = structure_ged
        results["structure_ig"] = structure_ig

        # 2. Semantic analysis (Improved Entropy)
        semantic_ged, semantic_ig = self._analyze_semantics(graph1, graph2)
        results["semantic_ged"] = semantic_ged
        results["semantic_ig"] = semantic_ig

        # 3. Embedding quality analysis
        quality_score = self._analyze_embedding_quality(graph1, graph2)
        results["quality_score"] = quality_score

        # 4. Query relevance (if provided)
        if query:
            relevance_score = self._analyze_query_relevance(graph1, graph2, query)
            results["relevance_score"] = relevance_score
        else:
            relevance_score = 1.0

        # Combine metrics with quality weighting
        effective_quality = max(0.1, quality_score)  # Avoid zero weight

        # Hybrid GED (negative means simplification/insight)
        results["hybrid_ged"] = (
            self.weights["structure"] * structure_ged
            + self.weights["semantic"] * semantic_ged
        ) * effective_quality

        # Hybrid IG (positive means information gain)
        results["hybrid_ig"] = (
            (
                self.weights["structure"] * structure_ig
                + self.weights["semantic"] * semantic_ig
            )
            * effective_quality
            * relevance_score
        )

        # Spike detection confidence
        results["spike_confidence"] = self._calculate_spike_confidence(results)

        # Additional scores
        results["structure_score"] = self._normalize_score(structure_ged, structure_ig)
        results["semantic_score"] = self._normalize_score(semantic_ged, semantic_ig)

        logger.debug(
            f"Hybrid metrics: GED={results['hybrid_ged']:.3f}, "
            f"IG={results['hybrid_ig']:.3f}, "
            f"Confidence={results['spike_confidence']:.3f}"
        )

        return results

    def _analyze_structure(self, graph1: Any, graph2: Any) -> Tuple[float, float]:
        """構造変化の分析"""
        if not self.has_networkx:
            # Fallback to simple structure analysis
            size1 = getattr(graph1, "num_nodes", 0)
            size2 = getattr(graph2, "num_nodes", 0)

            # Simple GED approximation
            ged = float(abs(size2 - size1))
            if size2 < size1:  # Simplification
                ged = -ged

            # Simple IG based on size change
            ig = np.log(max(size2, 1)) - np.log(max(size1, 1))

            return ged, ig

        try:
            # Use NetworkX for accurate structure analysis
            from ..algorithms.pyg_adapter import PyGAdapter

            nx1 = PyGAdapter.pyg_to_networkx(graph1)
            nx2 = PyGAdapter.pyg_to_networkx(graph2)

            # Calculate GED
            result = self.ged_calculator.calculate(nx1, nx2)
            ged = result.ged_value

            # Negative for simplification
            if nx2.number_of_nodes() < nx1.number_of_nodes():
                ged = -ged

            # Structure-based IG
            density1 = nx1.number_of_edges() / max(
                nx1.number_of_nodes() * (nx1.number_of_nodes() - 1) / 2, 1
            )
            density2 = nx2.number_of_edges() / max(
                nx2.number_of_nodes() * (nx2.number_of_nodes() - 1) / 2, 1
            )

            ig = density2 - density1  # Increase in connectivity

            return float(ged), float(ig)

        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            return 0.0, 0.0

    def _analyze_semantics(self, graph1: Any, graph2: Any) -> Tuple[float, float]:
        """意味変化の分析"""
        if not self.has_improved_entropy:
            # Fallback to simple semantic analysis
            return 0.0, 0.0

        try:
            # Use improved entropy
            ig = self.entropy_calculator.calculate_ig(graph1, graph2)

            # Derive GED from IG (heuristic)
            # Large positive IG suggests expansion (positive GED)
            # Large negative IG suggests simplification (negative GED)
            ged = ig * 10.0  # Scale factor

            return float(ged), float(ig)

        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return 0.0, 0.0

    def _analyze_embedding_quality(self, graph1: Any, graph2: Any) -> float:
        """埋め込みの質を分析"""
        try:
            quality_scores = []

            for graph in [graph1, graph2]:
                if hasattr(graph, "x") and graph.x is not None:
                    embeddings = (
                        graph.x.cpu().numpy()
                        if hasattr(graph.x, "cpu")
                        else graph.x.numpy()
                    )

                    # Check if embeddings are meaningful (not random)
                    # 1. Variance check
                    var_per_dim = np.var(embeddings, axis=0)
                    meaningful_dims = np.sum(var_per_dim > 0.5)
                    var_score = meaningful_dims / embeddings.shape[1]

                    # 2. Clustering tendency
                    from sklearn.metrics import silhouette_score

                    if len(embeddings) > 2:
                        try:
                            from sklearn.cluster import KMeans

                            kmeans = KMeans(
                                n_clusters=min(3, len(embeddings)), random_state=42
                            )
                            labels = kmeans.fit_predict(embeddings)
                            sil_score = silhouette_score(embeddings, labels)
                            cluster_score = (sil_score + 1) / 2  # Normalize to [0, 1]
                        except:
                            cluster_score = 0.5
                    else:
                        cluster_score = 0.5

                    quality_scores.append(var_score * 0.5 + cluster_score * 0.5)
                else:
                    quality_scores.append(0.0)

            return float(np.mean(quality_scores))

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return 0.5  # Default medium quality

    def _analyze_query_relevance(self, graph1: Any, graph2: Any, query: str) -> float:
        """クエリとの関連性を分析"""
        # TODO: Implement query-aware analysis
        # For now, return 1.0 (fully relevant)
        return 1.0

    def _calculate_spike_confidence(self, metrics: Dict[str, float]) -> float:
        """スパイク検出の信頼度を計算"""
        confidence = 0.0

        # Strong negative GED (simplification) is good
        if metrics["hybrid_ged"] < -0.5:
            confidence += 0.3

        # Positive IG is good
        if metrics["hybrid_ig"] > 0.2:
            confidence += 0.3

        # Agreement between structure and semantic
        if (metrics["structure_ged"] < 0) == (metrics["semantic_ged"] < 0):
            confidence += 0.2

        # High quality embeddings
        confidence += metrics["quality_score"] * 0.2

        return min(1.0, confidence)

    def _normalize_score(self, ged: float, ig: float) -> float:
        """GEDとIGを統合した正規化スコア"""
        # Negative GED and positive IG are both good
        ged_score = 1.0 / (1.0 + np.exp(ged))  # Sigmoid, high for negative GED
        ig_score = 1.0 / (1.0 + np.exp(-ig))  # Sigmoid, high for positive IG

        return float((ged_score + ig_score) / 2)
