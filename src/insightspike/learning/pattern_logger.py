"""
Pattern Logger for Learning Mechanism
====================================

Logs successful reasoning patterns and rewards for strategy adjustment.

IMPORT INSTRUMENTATION: lightweight prints for diagnosing pytest collection hang.
Safe to remove after root cause isolation.
"""
print('[pattern_logger] module import start', flush=True)

import json
import logging
import time
print('[pattern_logger] basic stdlib imported', flush=True)
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
print('[pattern_logger] numpy imported', flush=True)

logger = logging.getLogger(__name__)


@dataclass
class ReasoningPattern:
    """Represents a successful reasoning pattern"""
    
    question: str
    question_embedding: Optional[List[float]]  # For similarity matching
    retrieved_docs: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]  # ΔGED, ΔIG
    reward: float
    spike_detected: bool
    response_quality: float
    timestamp: float
    
    # Strategy parameters used
    similarity_threshold: float
    hop_limit: int
    path_decay: float
    
    # Outcome metrics
    user_feedback: Optional[float] = None  # Future: user satisfaction
    concepts_discovered: List[str] = None
    insights_generated: List[str] = None


class PatternLogger:
    """
    Logs and analyzes successful reasoning patterns for learning.
    
    Features:
    - Pattern storage with rewards
    - Strategy effectiveness tracking
    - Pattern similarity matching
    - Strategy recommendation
    """
    
    def __init__(self, config=None, log_dir: Optional[str] = None):
        """Initialize PatternLogger (instrumented)."""
        print('[pattern_logger] __init__ start', flush=True)
        self.config = config
        self.log_dir = Path(log_dir or "/tmp/insightspike/patterns")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print('[pattern_logger] log_dir ready', flush=True)

        # In-memory pattern cache
        self.patterns: List[ReasoningPattern] = []
        self.max_patterns = 1000  # Limit memory usage

        # Strategy effectiveness tracking
        self.strategy_rewards: Dict[str, List[float]] = {
            "high_threshold": [],  # similarity_threshold > 0.4
            "low_threshold": [],   # similarity_threshold <= 0.4
            "single_hop": [],      # hop_limit = 1
            "multi_hop": [],       # hop_limit > 1
            "high_decay": [],      # path_decay > 0.7
            "low_decay": [],       # path_decay <= 0.7
        }

        # Load existing patterns
        try:
            self._load_patterns()
        except Exception as e:  # pragma: no cover
            logger.debug(f'Pattern load skipped: {e}')
        print('[pattern_logger] __init__ complete', flush=True)
    
    def log_pattern(
        self,
        question: str,
        context: Dict[str, Any],
        result: Any,
        config_snapshot: Dict[str, Any]
    ) -> None:
        """Log a reasoning pattern with its outcome"""
        
        try:
            # Extract key information
            graph_analysis = context.get("graph_analysis", {})
            metrics = graph_analysis.get("metrics", {})
            reward_info = graph_analysis.get("reward", {})
            
            # Create pattern record
            pattern = ReasoningPattern(
                question=question,
                question_embedding=self._get_embedding_list(context),
                retrieved_docs=self._sanitize_docs(context.get("retrieved_documents", [])),
                graph_metrics={
                    "delta_ged": metrics.get("delta_ged", 0.0),
                    "delta_ig": metrics.get("delta_ig", 0.0),
                },
                reward=reward_info.get("total", 0.0),
                spike_detected=graph_analysis.get("spike_detected", False),
                response_quality=context.get("reasoning_quality", 0.5),
                timestamp=time.time(),
                similarity_threshold=config_snapshot.get("similarity_threshold", 0.3),
                hop_limit=config_snapshot.get("hop_limit", 2),
                path_decay=config_snapshot.get("path_decay", 0.7),
                concepts_discovered=self._extract_concepts(result),
                insights_generated=self._extract_insights(result),
            )
            
            # Add to cache
            self.patterns.append(pattern)
            if len(self.patterns) > self.max_patterns:
                self.patterns.pop(0)  # Remove oldest
            
            # Update strategy rewards
            self._update_strategy_rewards(pattern)
            
            # Persist if significant
            if pattern.reward > 0.5 or pattern.spike_detected:
                self._save_pattern(pattern)
            
            logger.debug(f"Logged pattern with reward {pattern.reward:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to log pattern: {e}")
    
    def find_similar_patterns(
        self,
        question: str,
        embedding: Optional[np.ndarray] = None,
        top_k: int = 5
    ) -> List[Tuple[ReasoningPattern, float]]:
        """Find patterns similar to the current question"""
        
        if not self.patterns:
            return []
        
        # Simple text similarity for now
        # TODO: Use embeddings when available
        similarities = []
        
        for pattern in self.patterns:
            # Calculate similarity (simple word overlap for now)
            q_words = set(question.lower().split())
            p_words = set(pattern.question.lower().split())
            
            if len(q_words) == 0:
                similarity = 0.0
            else:
                similarity = len(q_words & p_words) / len(q_words)
            
            similarities.append((pattern, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def recommend_strategy(
        self,
        question: str,
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend strategy adjustments based on similar patterns"""
        
        # Find similar successful patterns
        similar_patterns = self.find_similar_patterns(question, top_k=10)
        
        if not similar_patterns:
            return current_config  # No changes
        
        # Analyze what worked for similar questions
        successful_patterns = [
            (p, s) for p, s in similar_patterns 
            if p.reward > 0.3 or p.spike_detected
        ]
        
        if not successful_patterns:
            return current_config
        
        # Average successful strategies (weighted by similarity)
        total_weight = sum(s for _, s in successful_patterns)
        if total_weight == 0:
            return current_config
        
        avg_threshold = sum(
            p.similarity_threshold * s for p, s in successful_patterns
        ) / total_weight
        
        avg_hop_limit = sum(
            p.hop_limit * s for p, s in successful_patterns
        ) / total_weight
        
        avg_decay = sum(
            p.path_decay * s for p, s in successful_patterns
        ) / total_weight
        
        # Create recommended config
        recommended = current_config.copy()
        
        # Adjust with some smoothing
        alpha = 0.3  # Learning rate
        recommended["similarity_threshold"] = (
            (1 - alpha) * current_config.get("similarity_threshold", 0.3) +
            alpha * avg_threshold
        )
        recommended["hop_limit"] = int(round(
            (1 - alpha) * current_config.get("hop_limit", 2) +
            alpha * avg_hop_limit
        ))
        recommended["path_decay"] = (
            (1 - alpha) * current_config.get("path_decay", 0.7) +
            alpha * avg_decay
        )
        
        logger.info(
            f"Strategy recommendation: threshold={recommended['similarity_threshold']:.3f}, "
            f"hops={recommended['hop_limit']}, decay={recommended['path_decay']:.3f}"
        )
        
        return recommended
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for different strategies"""
        
        performance = {}
        
        for strategy, rewards in self.strategy_rewards.items():
            if rewards:
                performance[strategy] = {
                    "avg_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "max_reward": max(rewards),
                    "count": len(rewards),
                    "spike_rate": sum(1 for r in rewards if r > 0.5) / len(rewards),
                }
            else:
                performance[strategy] = {
                    "avg_reward": 0.0,
                    "std_reward": 0.0,
                    "max_reward": 0.0,
                    "count": 0,
                    "spike_rate": 0.0,
                }
        
        return performance
    
    def _update_strategy_rewards(self, pattern: ReasoningPattern) -> None:
        """Update strategy performance tracking"""
        
        reward = pattern.reward
        
        # Categorize by threshold
        if pattern.similarity_threshold > 0.4:
            self.strategy_rewards["high_threshold"].append(reward)
        else:
            self.strategy_rewards["low_threshold"].append(reward)
        
        # Categorize by hop limit
        if pattern.hop_limit == 1:
            self.strategy_rewards["single_hop"].append(reward)
        else:
            self.strategy_rewards["multi_hop"].append(reward)
        
        # Categorize by decay
        if pattern.path_decay > 0.7:
            self.strategy_rewards["high_decay"].append(reward)
        else:
            self.strategy_rewards["low_decay"].append(reward)
        
        # Keep only recent rewards (last 100)
        for strategy in self.strategy_rewards:
            if len(self.strategy_rewards[strategy]) > 100:
                self.strategy_rewards[strategy].pop(0)
    
    def _get_embedding_list(self, context: Dict[str, Any]) -> Optional[List[float]]:
        """Extract question embedding if available"""
        # TODO: Implement when embeddings are accessible
        return None
    
    def _sanitize_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove embeddings from docs to save space"""
        sanitized = []
        for doc in docs[:5]:  # Keep only top 5
            clean_doc = {
                "text": doc.get("text", "")[:100],  # Truncate
                "similarity": doc.get("similarity", 0.0),
                "hop": doc.get("hop", 0),
            }
            sanitized.append(clean_doc)
        return sanitized
    
    def _extract_concepts(self, result: Any) -> List[str]:
        """Extract discovered concepts from result"""
        if hasattr(result, "concepts_discovered"):
            return result.concepts_discovered[:10]
        return []
    
    def _extract_insights(self, result: Any) -> List[str]:
        """Extract generated insights from result"""
        if hasattr(result, "insights_generated"):
            return result.insights_generated[:5]
        return []
    
    def _save_pattern(self, pattern: ReasoningPattern) -> None:
        """Save significant pattern to disk"""
        try:
            filename = f"pattern_{int(pattern.timestamp)}.json"
            filepath = self.log_dir / filename
            
            with open(filepath, "w") as f:
                json.dump(asdict(pattern), f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
    
    def _load_patterns(self) -> None:
        """Load patterns from disk"""
        try:
            pattern_files = sorted(self.log_dir.glob("pattern_*.json"))
            
            # Load most recent patterns
            for filepath in pattern_files[-self.max_patterns:]:
                with open(filepath) as f:
                    data = json.load(f)
                    pattern = ReasoningPattern(**data)
                    self.patterns.append(pattern)
                    self._update_strategy_rewards(pattern)
            
            logger.info(f"Loaded {len(self.patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")