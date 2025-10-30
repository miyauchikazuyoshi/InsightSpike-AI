"""
Insight Fact Registration System
===============================

Comprehensive system for registering, evaluating, and utilizing discovered insights
with graph optimization analysis using GED/IG metrics.

Key Features:
- Automatic insight extraction from agent responses
- Quality assessment using multiple criteria
- Graph optimization evaluation with GED/IG
- Structured storage and retrieval
- Integration with existing learning systems
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InsightFact:
    """Represents a discovered insight fact"""

    id: str
    text: str
    source_concepts: List[str]
    target_concepts: List[str]
    confidence: float
    quality_score: float
    ged_optimization: float  # Graph improvement via GED
    ig_improvement: float  # Information gain improvement
    discovery_context: str
    generated_at: float
    validation_status: str  # 'pending', 'verified', 'rejected'
    relationship_type: str  # 'causal', 'structural', 'analogical', 'synthetic'
    usage_count: int = 0
    last_accessed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InsightFact":
        return cls(**data)


@dataclass
class GraphOptimizationResult:
    """Results of graph optimization analysis"""

    before_ged: float
    after_ged: float
    ged_improvement: float
    before_ig: float
    after_ig: float
    ig_improvement: float
    structural_efficiency: float
    connectivity_improvement: float


class InsightFactRegistry:
    """
    Manages registration, evaluation, and utilization of discovered insights.

    This system completes the learning cycle:
    Question → Analysis → Insight → Registration → Future Utilization
    """

    # Quality thresholds for insight registration
    MIN_QUALITY_SCORE = 0.6
    MIN_GED_IMPROVEMENT = 0.1
    MIN_IG_IMPROVEMENT = 0.05
    MAX_INSIGHTS = 5000  # Prevent unbounded growth

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the insight fact registry"""
        if db_path is None:
            from ..config import get_config

            config = get_config()
            db_path = Path("data") / "insight_facts.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache for fast access
        self.insight_cache: Dict[str, InsightFact] = {}
        self.concept_index: Dict[str, Set[str]] = defaultdict(
            set
        )  # concept -> insight_ids

        # Initialize database
        self._init_database()
        self._load_from_database()

        logger.info(
            f"InsightFactRegistry initialized with {len(self.insight_cache)} existing insights"
        )

    @property
    def insights(self) -> Dict[str, InsightFact]:
        """Access to insights cache for compatibility"""
        return self.insight_cache

    def _init_database(self):
        """Initialize SQLite database for insight storage"""
        with sqlite3.connect(self.db_path) as conn:
            # Main insights table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source_concepts TEXT NOT NULL,
                    target_concepts TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    ged_optimization REAL NOT NULL,
                    ig_improvement REAL NOT NULL,
                    discovery_context TEXT,
                    generated_at REAL NOT NULL,
                    updated_at REAL DEFAULT 0,
                    validation_status TEXT DEFAULT 'pending',
                    relationship_type TEXT,
                    usage_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0
                )
            """
            )

            # Concept-insight relationships for fast lookup
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_insights (
                    concept TEXT NOT NULL,
                    insight_id TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    PRIMARY KEY (concept, insight_id),
                    FOREIGN KEY (insight_id) REFERENCES insights(id)
                )
            """
            )

            # Graph optimization history
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_history (
                    insight_id TEXT NOT NULL,
                    before_ged REAL,
                    after_ged REAL,
                    before_ig REAL,
                    after_ig REAL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (insight_id) REFERENCES insights(id)
                )
            """
            )

            # Create indexes for performance
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_quality ON insights(quality_score)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_concepts ON concept_insights(concept)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_validation ON insights(validation_status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_generated_at ON insights(generated_at)"
                )
            except:
                pass

            # Migration: Add updated_at column if it doesn't exist
            try:
                conn.execute(
                    "ALTER TABLE insights ADD COLUMN updated_at REAL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists

            conn.commit()

    def _load_from_database(self):
        """Load existing insights from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, text, source_concepts, target_concepts, confidence,
                       quality_score, ged_optimization, ig_improvement,
                       discovery_context, generated_at, validation_status,
                       relationship_type, usage_count, last_accessed
                FROM insights
            """
            )

            for row in cursor.fetchall():
                insight = InsightFact(
                    id=row[0],
                    text=row[1],
                    source_concepts=json.loads(row[2]),
                    target_concepts=json.loads(row[3]),
                    confidence=row[4],
                    quality_score=row[5],
                    ged_optimization=row[6],
                    ig_improvement=row[7],
                    discovery_context=row[8],
                    generated_at=row[9],
                    validation_status=row[10],
                    relationship_type=row[11],
                    usage_count=row[12],
                    last_accessed=row[13],
                )

                self.insight_cache[insight.id] = insight

                # Build concept index
                for concept in insight.source_concepts + insight.target_concepts:
                    self.concept_index[concept.lower()].add(insight.id)

    def extract_insights_from_response(
        self,
        question: str,
        response: str,
        l1_analysis: Any,
        reasoning_quality: float,
        graph_before: Optional[nx.Graph] = None,
        graph_after: Optional[nx.Graph] = None,
    ) -> List[InsightFact]:
        """
        Extract potential insights from an agent response and evaluate them.

        Args:
            question: Original question that triggered the insight
            response: Agent's response containing potential insights
            l1_analysis: Layer1 analysis results
            reasoning_quality: Quality score of the reasoning
            graph_before: Knowledge graph before insight
            graph_after: Knowledge graph after insight

        Returns:
            List of validated insight facts
        """
        potential_insights = []

        # 1. Extract potential insights using heuristics
        insight_candidates = self._extract_insight_candidates(response, l1_analysis)

        for candidate in insight_candidates:
            # 2. Evaluate quality
            quality_score = self._evaluate_insight_quality(
                candidate, question, reasoning_quality
            )

            if quality_score < self.MIN_QUALITY_SCORE:
                continue

            # 3. Evaluate graph optimization if graphs provided
            optimization_result = None
            if graph_before and graph_after:
                optimization_result = self._evaluate_graph_optimization(
                    graph_before, graph_after, candidate
                )

                # Filter by optimization thresholds
                if (
                    optimization_result.ged_improvement < self.MIN_GED_IMPROVEMENT
                    or optimization_result.ig_improvement < self.MIN_IG_IMPROVEMENT
                ):
                    continue

            # 4. Create insight fact
            insight_id = self._generate_insight_id(candidate["text"])

            insight = InsightFact(
                id=insight_id,
                text=candidate["text"],
                source_concepts=candidate["source_concepts"],
                target_concepts=candidate["target_concepts"],
                confidence=candidate["confidence"],
                quality_score=quality_score,
                ged_optimization=optimization_result.ged_improvement
                if optimization_result
                else 0.0,
                ig_improvement=optimization_result.ig_improvement
                if optimization_result
                else 0.0,
                discovery_context=f"Q: {question[:100]}...",
                generated_at=time.time(),
                validation_status="pending",
                relationship_type=candidate["relationship_type"],
            )

            potential_insights.append(insight)

        # 5. Register high-quality insights
        registered_insights = []
        for insight in potential_insights:
            if self.register_insight(insight):
                registered_insights.append(insight)

        logger.info(
            f"Extracted and registered {len(registered_insights)} insights from response"
        )
        return registered_insights

    def _extract_insight_candidates(
        self, response: str, l1_analysis: Any
    ) -> List[Dict[str, Any]]:
        """Extract potential insight statements from response text"""
        candidates = []

        # Insight indicators in text
        insight_patterns = [
            # Connection patterns
            r"(.+?)\s+(?:connects?|relates?|links?)\s+(?:to|with)\s+(.+)",
            # Causal patterns
            r"(.+?)\s+(?:causes?|leads? to|results? in)\s+(.+)",
            # Structural patterns
            r"(.+?)\s+(?:is|are)\s+(?:essentially|fundamentally|actually)\s+(.+)",
            # Analogical patterns
            r"(.+?)\s+(?:is like|resembles|mirrors)\s+(.+)",
            # Synthesis patterns
            r"(?:by combining|integrating|synthesizing)\s+(.+?)\s+(?:and|with)\s+(.+?),?\s+we get\s+(.+)",
        ]

        import re

        sentences = response.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue

            for pattern in insight_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    groups = match.groups()

                    if len(groups) >= 2:
                        source_concepts = [groups[0].strip()]
                        target_concepts = [groups[1].strip()]
                        relationship_type = self._classify_relationship_type(sentence)

                        candidates.append(
                            {
                                "text": sentence,
                                "source_concepts": source_concepts,
                                "target_concepts": target_concepts,
                                "confidence": 0.7,  # Base confidence for pattern matches
                                "relationship_type": relationship_type,
                            }
                        )
                        break

        # Also extract insights from L1 analysis unknown elements
        if hasattr(l1_analysis, "unknown_elements") and l1_analysis.unknown_elements:
            # Look for synthesis of unknown elements
            synthesis_text = (
                f"Synthesis of {', '.join(l1_analysis.unknown_elements[:3])}"
            )
            if len(synthesis_text) > 20:
                candidates.append(
                    {
                        "text": synthesis_text,
                        "source_concepts": l1_analysis.unknown_elements[:2],
                        "target_concepts": l1_analysis.unknown_elements[2:3]
                        if len(l1_analysis.unknown_elements) > 2
                        else [],
                        "confidence": 0.6,
                        "relationship_type": "synthetic",
                    }
                )

        return candidates

    def _classify_relationship_type(self, text: str) -> str:
        """Classify the type of relationship expressed in the insight"""
        text_lower = text.lower()

        if any(
            word in text_lower
            for word in ["cause", "leads to", "results in", "because"]
        ):
            return "causal"
        elif any(
            word in text_lower for word in ["structure", "composed of", "consists of"]
        ):
            return "structural"
        elif any(
            word in text_lower for word in ["like", "similar", "resembles", "analogy"]
        ):
            return "analogical"
        elif any(
            word in text_lower for word in ["combining", "synthesis", "integration"]
        ):
            return "synthetic"
        else:
            return "general"

    def _evaluate_insight_quality(
        self, candidate: Dict[str, Any], question: str, reasoning_quality: float
    ) -> float:
        """Evaluate the quality of an insight candidate"""
        quality_factors = []

        # 1. Text quality (length, complexity)
        text_quality = min(1.0, len(candidate["text"]) / 100.0)
        quality_factors.append(text_quality * 0.2)

        # 2. Concept richness
        total_concepts = len(candidate["source_concepts"]) + len(
            candidate["target_concepts"]
        )
        concept_quality = min(1.0, total_concepts / 5.0)
        quality_factors.append(concept_quality * 0.3)

        # 3. Base confidence
        quality_factors.append(candidate["confidence"] * 0.2)

        # 4. Reasoning quality from agent
        quality_factors.append(reasoning_quality * 0.2)

        # 5. Novelty check (not already registered)
        novelty_score = 1.0
        insight_text_lower = candidate["text"].lower()
        for existing_insight in self.insight_cache.values():
            if (
                self._calculate_text_similarity(
                    insight_text_lower, existing_insight.text.lower()
                )
                > 0.8
            ):
                novelty_score = 0.3  # Significantly reduce quality for duplicates
                break
        quality_factors.append(novelty_score * 0.1)

        return sum(quality_factors)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _evaluate_graph_optimization(
        self,
        graph_before: nx.Graph,
        graph_after: nx.Graph,
        insight_candidate: Dict[str, Any],
    ) -> GraphOptimizationResult:
        """Evaluate how the insight optimizes the knowledge graph"""

        # Calculate GED between before/after graphs
        try:
            # For large graphs, use approximation
            if len(graph_before.nodes()) > 50 or len(graph_after.nodes()) > 50:
                ged_before = self._approximate_ged_baseline(graph_before)
                ged_after = self._approximate_ged_baseline(graph_after)
            else:
                ged_before = nx.graph_edit_distance(graph_before, nx.Graph())
                ged_after = nx.graph_edit_distance(graph_after, nx.Graph())

            ged_improvement = (ged_before - ged_after) / max(ged_before, 1.0)
        except:
            ged_before = ged_after = 0.0
            ged_improvement = 0.0

        # Calculate Information Gain
        ig_before = self._calculate_graph_information_content(graph_before)
        ig_after = self._calculate_graph_information_content(graph_after)
        ig_improvement = ig_after - ig_before

        # Structural efficiency metrics
        efficiency_before = self._calculate_structural_efficiency(graph_before)
        efficiency_after = self._calculate_structural_efficiency(graph_after)
        structural_efficiency = efficiency_after - efficiency_before

        # Connectivity improvement
        connectivity_before = (
            nx.average_clustering(graph_before) if graph_before.nodes() else 0
        )
        connectivity_after = (
            nx.average_clustering(graph_after) if graph_after.nodes() else 0
        )
        connectivity_improvement = connectivity_after - connectivity_before

        return GraphOptimizationResult(
            before_ged=ged_before,
            after_ged=ged_after,
            ged_improvement=ged_improvement,
            before_ig=ig_before,
            after_ig=ig_after,
            ig_improvement=ig_improvement,
            structural_efficiency=structural_efficiency,
            connectivity_improvement=connectivity_improvement,
        )

    def _approximate_ged_baseline(self, graph: nx.Graph) -> float:
        """Approximate GED for large graphs"""
        # Use graph properties as GED proxy
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())
        avg_degree = sum(dict(graph.degree()).values()) / max(n_nodes, 1)
        clustering = nx.average_clustering(graph) if n_nodes > 0 else 0

        # Combine metrics into GED approximation
        return n_nodes * 0.4 + n_edges * 0.3 + avg_degree * 0.2 + clustering * 0.1

    def _calculate_graph_information_content(self, graph: nx.Graph) -> float:
        """Calculate information content of a graph"""
        if not graph.nodes():
            return 0.0

        # Use entropy-based measure
        degrees = [d for n, d in graph.degree()]
        if not degrees:
            return 0.0

        # Degree distribution entropy
        from collections import Counter

        degree_counts = Counter(degrees)
        total_nodes = len(degrees)

        entropy = 0.0
        for count in degree_counts.values():
            prob = count / total_nodes
            if prob > 0:
                entropy -= prob * np.log2(prob)

        return entropy

    def _calculate_structural_efficiency(self, graph: nx.Graph) -> float:
        """Calculate structural efficiency of the graph"""
        if not graph.nodes() or len(graph.nodes()) < 2:
            return 0.0

        try:
            # Global efficiency
            return nx.global_efficiency(graph)
        except:
            # Fallback: average shortest path efficiency
            try:
                avg_path_length = nx.average_shortest_path_length(graph)
                return 1.0 / avg_path_length if avg_path_length > 0 else 0.0
            except:
                return 0.0

    def register_insight(self, insight: InsightFact) -> bool:
        """Register a new insight fact"""

        # Check if already exists
        if insight.id in self.insight_cache:
            logger.debug(f"Insight {insight.id} already registered")
            return False

        # Check capacity
        if len(self.insight_cache) >= self.MAX_INSIGHTS:
            logger.warning(
                f"Maximum insights reached ({self.MAX_INSIGHTS}), cleaning up old insights"
            )
            self._cleanup_old_insights()

        # Store in cache and database
        self.insight_cache[insight.id] = insight

        # Update concept index
        for concept in insight.source_concepts + insight.target_concepts:
            self.concept_index[concept.lower()].add(insight.id)

        # Store in database
        self._store_insight_in_db(insight)

        logger.info(
            f"Registered new insight: {insight.text[:50]}... (Quality: {insight.quality_score:.3f})"
        )
        return True

    def _generate_insight_id(self, text: str) -> str:
        """Generate unique ID for insight"""
        # Use content hash for consistent IDs
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _store_insight_in_db(self, insight: InsightFact):
        """Store insight in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO insights 
                (id, text, source_concepts, target_concepts, confidence,
                 quality_score, ged_optimization, ig_improvement, discovery_context,
                 generated_at, validation_status, relationship_type, usage_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    insight.id,
                    insight.text,
                    json.dumps(insight.source_concepts),
                    json.dumps(insight.target_concepts),
                    insight.confidence,
                    insight.quality_score,
                    insight.ged_optimization,
                    insight.ig_improvement,
                    insight.discovery_context,
                    insight.generated_at,
                    insight.validation_status,
                    insight.relationship_type,
                    insight.usage_count,
                    insight.last_accessed,
                ),
            )

            # Store concept relationships
            for concept in insight.source_concepts + insight.target_concepts:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO concept_insights (concept, insight_id, relevance_score)
                    VALUES (?, ?, ?)
                """,
                    (concept.lower(), insight.id, 1.0),
                )

            conn.commit()

    def find_relevant_insights(
        self, concepts: List[str], limit: int = 10
    ) -> List[InsightFact]:
        """Find insights relevant to given concepts"""
        relevant_ids = set()

        for concept in concepts:
            concept_lower = concept.lower()
            if concept_lower in self.concept_index:
                relevant_ids.update(self.concept_index[concept_lower])

        # Get insights and sort by quality
        insights = [
            self.insight_cache[id] for id in relevant_ids if id in self.insight_cache
        ]
        insights.sort(key=lambda x: x.quality_score, reverse=True)

        return insights[:limit]

    def get_insights_by_type(
        self, relationship_type: str, limit: int = 10
    ) -> List[InsightFact]:
        """Get insights by relationship type"""
        matching_insights = [
            insight
            for insight in self.insights.values()
            if insight.relationship_type == relationship_type
        ]
        # Sort by quality score and return top insights
        return sorted(matching_insights, key=lambda x: x.quality_score, reverse=True)[
            :limit
        ]

    def get_recent_insights(self, limit: int = 50) -> List[InsightFact]:
        """Get recent insights sorted by generation time"""
        all_insights = list(self.insights.values())
        # Sort by generated_at timestamp in descending order (most recent first)
        return sorted(all_insights, key=lambda x: x.generated_at, reverse=True)[:limit]

    def search_insights_by_concept(
        self, concept: str, limit: int = 10
    ) -> List[InsightFact]:
        """Search insights that contain the given concept"""
        concept_lower = concept.lower()
        matching_insights = []

        for insight in self.insights.values():
            # Check if concept appears in source or target concepts
            if (
                any(concept_lower in c.lower() for c in insight.source_concepts)
                or any(concept_lower in c.lower() for c in insight.target_concepts)
                or concept_lower in insight.text.lower()
            ):
                matching_insights.append(insight)

        # Sort by quality score and return top insights
        return sorted(matching_insights, key=lambda x: x.quality_score, reverse=True)[
            :limit
        ]

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about graph optimization from insights"""
        insights = list(self.insight_cache.values())

        if not insights:
            return {"total_insights": 0}

        ged_improvements = [
            i.ged_optimization for i in insights if i.ged_optimization > 0
        ]
        ig_improvements = [i.ig_improvement for i in insights if i.ig_improvement > 0]

        return {
            "total_insights": len(insights),
            "validated_insights": len(
                [i for i in insights if i.validation_status == "verified"]
            ),
            "avg_quality_score": sum(i.quality_score for i in insights) / len(insights),
            "avg_ged_improvement": sum(ged_improvements) / len(ged_improvements)
            if ged_improvements
            else 0,
            "avg_ig_improvement": sum(ig_improvements) / len(ig_improvements)
            if ig_improvements
            else 0,
            "max_ged_improvement": max(ged_improvements) if ged_improvements else 0,
            "max_ig_improvement": max(ig_improvements) if ig_improvements else 0,
            "relationship_types": {
                rtype: len([i for i in insights if i.relationship_type == rtype])
                for rtype in set(i.relationship_type for i in insights)
            },
        }

    def _cleanup_old_insights(self):
        """Clean up old, low-quality insights to maintain capacity"""
        insights = list(self.insight_cache.values())

        # Sort by quality and age, remove bottom 10%
        insights.sort(key=lambda x: (x.quality_score, -x.generated_at))
        to_remove = insights[: len(insights) // 10]

        for insight in to_remove:
            # Remove from cache
            del self.insight_cache[insight.id]

            # Remove from concept index
            for concept in insight.source_concepts + insight.target_concepts:
                self.concept_index[concept.lower()].discard(insight.id)

            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM insights WHERE id = ?", (insight.id,))
                conn.execute(
                    "DELETE FROM concept_insights WHERE insight_id = ?", (insight.id,)
                )
                conn.commit()

        logger.info(f"Cleaned up {len(to_remove)} old insights")

    def _update_validation_status(self, insight_id: str, status: str):
        """Update validation status for an insight"""
        if insight_id in self.insight_cache:
            self.insight_cache[insight_id].validation_status = status

            # Update in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE insights 
                    SET validation_status = ?, updated_at = ?
                    WHERE id = ?
                """,
                    (status, time.time(), insight_id),
                )
                conn.commit()

    def _remove_insight_from_db(self, insight_id: str):
        """Remove insight from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM insights WHERE id = ?", (insight_id,))
            conn.execute(
                "DELETE FROM concept_insights WHERE insight_id = ?", (insight_id,)
            )
            conn.commit()


# Global instance for easy access
_global_registry: Optional[InsightFactRegistry] = None


def get_insight_registry() -> InsightFactRegistry:
    """Get or create global insight registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = InsightFactRegistry()
    return _global_registry


def shutdown_insight_registry():
    """Shutdown global insight registry"""
    global _global_registry
    if _global_registry is not None:
        _global_registry = None
