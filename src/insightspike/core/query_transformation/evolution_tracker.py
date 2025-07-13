"""
Query Evolution Tracking System
==============================

Phase 3: Track and learn from query transformation patterns.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .query_state import QueryState, QueryTransformationHistory

logger = logging.getLogger(__name__)


@dataclass
class EvolutionPattern:
    """Represents a successful query evolution pattern"""

    pattern_id: str
    initial_query_type: str  # e.g., "comparison", "causal", "exploratory"
    transformation_path: List[str]  # Sequence of stages
    absorbed_concepts: List[str]
    final_insights: List[str]
    success_score: float  # Quality of the final insight
    avg_confidence_gain: float
    num_hops: int
    branches_used: List[str]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionPattern":
        return cls(**data)


class PatternDatabase:
    """SQLite database for storing evolution patterns"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evolution_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    initial_query_type TEXT,
                    transformation_path TEXT,
                    absorbed_concepts TEXT,
                    final_insights TEXT,
                    success_score REAL,
                    avg_confidence_gain REAL,
                    num_hops INTEGER,
                    branches_used TEXT,
                    timestamp TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_embeddings (
                    query_id TEXT PRIMARY KEY,
                    pattern_id TEXT,
                    stage TEXT,
                    embedding BLOB,
                    confidence REAL,
                    timestamp TEXT,
                    FOREIGN KEY (pattern_id) REFERENCES evolution_patterns(pattern_id)
                )
            """
            )

            # Index for fast pattern lookup
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_type 
                ON evolution_patterns(initial_query_type)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_success_score 
                ON evolution_patterns(success_score DESC)
            """
            )

    def save_pattern(self, pattern: EvolutionPattern):
        """Save an evolution pattern"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evolution_patterns VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pattern.pattern_id,
                    pattern.initial_query_type,
                    json.dumps(pattern.transformation_path),
                    json.dumps(pattern.absorbed_concepts),
                    json.dumps(pattern.final_insights),
                    pattern.success_score,
                    pattern.avg_confidence_gain,
                    pattern.num_hops,
                    json.dumps(pattern.branches_used),
                    pattern.timestamp,
                ),
            )

    def get_similar_patterns(
        self, query_type: str, min_score: float = 0.7, limit: int = 5
    ) -> List[EvolutionPattern]:
        """Retrieve similar successful patterns"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM evolution_patterns 
                WHERE initial_query_type = ? AND success_score >= ?
                ORDER BY success_score DESC
                LIMIT ?
            """,
                (query_type, min_score, limit),
            )

            patterns = []
            for row in cursor:
                data = {
                    "pattern_id": row[0],
                    "initial_query_type": row[1],
                    "transformation_path": json.loads(row[2]),
                    "absorbed_concepts": json.loads(row[3]),
                    "final_insights": json.loads(row[4]),
                    "success_score": row[5],
                    "avg_confidence_gain": row[6],
                    "num_hops": row[7],
                    "branches_used": json.loads(row[8]),
                    "timestamp": row[9],
                }
                patterns.append(EvolutionPattern.from_dict(data))

            return patterns

    def save_embedding(
        self,
        query_id: str,
        pattern_id: str,
        stage: str,
        embedding: torch.Tensor,
        confidence: float,
    ):
        """Save query embedding for trajectory analysis"""
        with sqlite3.connect(self.db_path) as conn:
            embedding_bytes = embedding.cpu().numpy().tobytes()
            conn.execute(
                """
                INSERT INTO query_embeddings VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    query_id,
                    pattern_id,
                    stage,
                    embedding_bytes,
                    confidence,
                    datetime.now().isoformat(),
                ),
            )


class QueryTypeClassifier:
    """Classify queries into types for pattern matching"""

    QUERY_TYPES = {
        "comparison": ["vs", "versus", "compare", "difference", "similar"],
        "causal": ["why", "cause", "because", "lead", "result"],
        "relational": ["relate", "connect", "link", "between", "through"],
        "definitional": ["what is", "define", "meaning", "explain"],
        "procedural": ["how to", "steps", "process", "method"],
        "exploratory": ["explore", "discover", "find", "search"],
    }

    @classmethod
    def classify(cls, query: str) -> str:
        """Classify query into a type"""
        query_lower = query.lower()

        for query_type, keywords in cls.QUERY_TYPES.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type

        return "exploratory"  # Default


class EvolutionTracker:
    """Track and learn from query evolution patterns"""

    def __init__(self, db_path: Optional[Path] = None, enable_learning: bool = True):
        self.db_path = (
            db_path or Path.home() / ".insightspike" / "evolution_patterns.db"
        )
        self.pattern_db = PatternDatabase(self.db_path)
        self.enable_learning = enable_learning

        # Pattern success metrics
        self.pattern_performance = defaultdict(lambda: {"success": 0, "total": 0})

        # Learned optimization strategies
        self.optimization_rules = self._load_optimization_rules()

    def track_evolution(
        self,
        initial_query: str,
        transformation_history: QueryTransformationHistory,
        branches: Optional[List] = None,
        final_score: float = 0.0,
    ) -> EvolutionPattern:
        """Track a complete query evolution"""

        # Classify query type
        query_type = QueryTypeClassifier.classify(initial_query)

        # Extract pattern information
        states = transformation_history.states
        transformation_path = [s.stage for s in states]

        # Collect absorbed concepts
        all_concepts = []
        for state in states:
            all_concepts.extend(state.absorbed_concepts)
        absorbed_concepts = list(set(all_concepts))

        # Collect insights
        final_insights = transformation_history.get_total_insights()

        # Calculate metrics
        confidence_gains = [
            states[i + 1].confidence - states[i].confidence
            for i in range(len(states) - 1)
        ]
        avg_confidence_gain = np.mean(confidence_gains) if confidence_gains else 0

        # Count hops
        num_hops = len([s for s in states if s.stage in ["exploring", "transforming"]])

        # Branch information
        branch_info = []
        if branches:
            branch_info = [b.exploration_direction for b in branches]

        # Create pattern
        pattern = EvolutionPattern(
            pattern_id=f"{query_type}_{datetime.now().timestamp()}",
            initial_query_type=query_type,
            transformation_path=transformation_path,
            absorbed_concepts=absorbed_concepts[:10],  # Top 10
            final_insights=final_insights[:5],  # Top 5
            success_score=final_score,
            avg_confidence_gain=avg_confidence_gain,
            num_hops=num_hops,
            branches_used=branch_info,
        )

        # Save pattern if successful
        if final_score > 0.7 and self.enable_learning:
            self.pattern_db.save_pattern(pattern)
            self._update_pattern_performance(query_type, success=True)

            # Save embeddings for trajectory analysis
            for i, state in enumerate(states):
                if state.embedding is not None:
                    self.pattern_db.save_embedding(
                        query_id=f"{pattern.pattern_id}_state_{i}",
                        pattern_id=pattern.pattern_id,
                        stage=state.stage,
                        embedding=state.embedding,
                        confidence=state.confidence,
                    )
        else:
            self._update_pattern_performance(query_type, success=False)

        return pattern

    def suggest_exploration_strategy(
        self, query: str, current_state: QueryState
    ) -> Dict[str, Any]:
        """Suggest exploration strategy based on learned patterns"""

        query_type = QueryTypeClassifier.classify(query)

        # Get similar successful patterns
        similar_patterns = self.pattern_db.get_similar_patterns(query_type)

        if not similar_patterns:
            return self._default_strategy()

        # Analyze successful patterns
        suggestions = {
            "recommended_concepts": [],
            "recommended_branches": [],
            "expected_hops": 0,
            "confidence_threshold": 0.7,
            "exploration_temperature": 1.0,
        }

        # Aggregate recommendations from successful patterns
        concept_counts = defaultdict(int)
        branch_counts = defaultdict(int)
        hop_counts = []

        for pattern in similar_patterns:
            # Weight by success score
            weight = pattern.success_score

            for concept in pattern.absorbed_concepts:
                concept_counts[concept] += weight

            for branch in pattern.branches_used:
                branch_counts[branch] += weight

            hop_counts.append(pattern.num_hops)

        # Top recommendations
        suggestions["recommended_concepts"] = [
            concept
            for concept, _ in sorted(
                concept_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

        suggestions["recommended_branches"] = [
            branch
            for branch, _ in sorted(
                branch_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
        ]

        suggestions["expected_hops"] = int(np.median(hop_counts)) if hop_counts else 3

        # Adjust exploration temperature based on pattern performance
        performance = self.pattern_performance[query_type]
        if performance["total"] > 0:
            success_rate = performance["success"] / performance["total"]
            # Lower temperature for well-understood patterns
            suggestions["exploration_temperature"] = 0.5 + 0.5 * (1 - success_rate)

        return suggestions

    def get_optimization_hints(
        self,
        current_state: QueryState,
        transformation_history: QueryTransformationHistory,
    ) -> List[str]:
        """Get optimization hints based on current progress"""

        hints = []

        # Check if stuck in exploration
        recent_states = transformation_history.states[-3:]
        if len(recent_states) >= 3:
            confidence_changes = [
                recent_states[i + 1].confidence - recent_states[i].confidence
                for i in range(len(recent_states) - 1)
            ]

            if all(change < 0.05 for change in confidence_changes):
                hints.append("Consider branching to explore different perspectives")

        # Check if missing key concepts
        if (
            current_state.stage == "exploring"
            and len(current_state.absorbed_concepts) < 2
        ):
            hints.append("Try to connect with more fundamental concepts")

        # Suggest multi-hop if single-hop isn't working
        if (
            current_state.confidence < 0.3
            and current_state.transformation_magnitude < 0.2
        ):
            hints.append("Enable multi-hop reasoning for deeper connections")

        return hints

    def _update_pattern_performance(self, query_type: str, success: bool):
        """Update pattern performance metrics"""
        self.pattern_performance[query_type]["total"] += 1
        if success:
            self.pattern_performance[query_type]["success"] += 1

    def _default_strategy(self) -> Dict[str, Any]:
        """Default exploration strategy"""
        return {
            "recommended_concepts": [],
            "recommended_branches": ["general", "theoretical", "practical"],
            "expected_hops": 3,
            "confidence_threshold": 0.7,
            "exploration_temperature": 1.0,
        }

    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load learned optimization rules"""
        # In a real implementation, these would be loaded from storage
        return {
            "min_concepts_for_insight": 3,
            "optimal_branch_count": 2,
            "early_stopping_confidence": 0.9,
            "exploration_decay_rate": 0.8,
        }

    def export_patterns(self, output_path: Path):
        """Export successful patterns for analysis"""
        patterns = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM evolution_patterns 
                WHERE success_score > 0.7
                ORDER BY timestamp DESC
            """
            )

            for row in cursor:
                data = {
                    "pattern_id": row[0],
                    "initial_query_type": row[1],
                    "transformation_path": json.loads(row[2]),
                    "absorbed_concepts": json.loads(row[3]),
                    "final_insights": json.loads(row[4]),
                    "success_score": row[5],
                    "avg_confidence_gain": row[6],
                    "num_hops": row[7],
                    "branches_used": json.loads(row[8]),
                    "timestamp": row[9],
                }
                patterns.append(data)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "patterns": patterns,
                    "performance_metrics": dict(self.pattern_performance),
                    "export_date": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"Exported {len(patterns)} patterns to {output_path}")


class TrajectoryAnalyzer:
    """Analyze query evolution trajectories for insights"""

    def __init__(self, pattern_db: PatternDatabase):
        self.pattern_db = pattern_db

    def analyze_trajectory(self, pattern_id: str) -> Dict[str, Any]:
        """Analyze a specific evolution trajectory"""

        # Load embeddings for the pattern
        with sqlite3.connect(self.pattern_db.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT stage, embedding, confidence 
                FROM query_embeddings 
                WHERE pattern_id = ?
                ORDER BY timestamp
            """,
                (pattern_id,),
            )

            stages = []
            embeddings = []
            confidences = []

            for row in cursor:
                stages.append(row[0])
                # Reconstruct embedding from bytes
                emb_array = np.frombuffer(row[1], dtype=np.float32)
                embeddings.append(emb_array)
                confidences.append(row[2])

        if not embeddings:
            return {}

        # Calculate trajectory metrics
        embeddings = np.array(embeddings)

        # Total distance traveled
        distances = []
        for i in range(1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[i - 1])
            distances.append(dist)

        total_distance = sum(distances)

        # Trajectory smoothness (lower is smoother)
        if len(distances) > 1:
            smoothness = np.std(distances)
        else:
            smoothness = 0.0

        # Direction changes
        direction_changes = 0
        if len(embeddings) > 2:
            for i in range(1, len(embeddings) - 1):
                v1 = embeddings[i] - embeddings[i - 1]
                v2 = embeddings[i + 1] - embeddings[i]

                # Normalize
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

                # Angle between vectors
                cos_angle = np.dot(v1_norm, v2_norm)
                angle = np.arccos(np.clip(cos_angle, -1, 1))

                if angle > np.pi / 4:  # More than 45 degrees
                    direction_changes += 1

        return {
            "total_distance": float(total_distance),
            "avg_step_size": float(np.mean(distances)) if distances else 0,
            "trajectory_smoothness": float(smoothness),
            "direction_changes": direction_changes,
            "confidence_correlation": float(
                np.corrcoef(range(len(confidences)), confidences)[0, 1]
            )
            if len(confidences) > 1
            else 0,
            "final_confidence": confidences[-1] if confidences else 0,
            "num_steps": len(stages),
        }
