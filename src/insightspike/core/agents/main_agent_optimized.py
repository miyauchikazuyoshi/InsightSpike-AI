"""
Optimized Main Agent with Full Query Transformation Integration
=============================================================

Phase 4: Production-ready agent with performance optimization and learning.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np

from ..config import Config
from ..query_transformation import QueryState, QueryTransformationHistory
from ..query_transformation.enhanced_query_transformer import EnhancedQueryTransformer
from ..query_transformation.evolution_tracker import (
    EvolutionTracker,
    QueryTypeClassifier,
    TrajectoryAnalyzer,
)
from .main_agent_advanced import MainAgentAdvanced, AdvancedTransformationResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Performance metrics for optimization"""

    query_processing_time: float
    transformation_cycles: int
    branches_explored: int
    memory_usage_mb: float
    cache_hits: int
    cache_misses: int
    gpu_utilization: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class QueryCache:
    """Intelligent query result caching"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.access_counts: Dict[str, int] = {}

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        if query in self.cache:
            result, timestamp = self.cache[query]
            if time.time() - timestamp < self.ttl_seconds:
                self.access_counts[query] = self.access_counts.get(query, 0) + 1
                return result
            else:
                # Expired
                del self.cache[query]
        return None

    def put(self, query: str, result: Dict[str, Any]):
        """Cache a query result"""
        # Evict least recently used if at capacity
        if len(self.cache) >= self.max_size:
            # Find least accessed query
            lru_query = min(
                self.cache.keys(), key=lambda k: self.access_counts.get(k, 0)
            )
            del self.cache[lru_query]
            if lru_query in self.access_counts:
                del self.access_counts[lru_query]

        self.cache[query] = (result, time.time())
        self.access_counts[query] = 1

    def invalidate_similar(self, query: str, threshold: float = 0.8):
        """Invalidate cached results similar to given query"""
        # In production, use embedding similarity
        # For now, use simple string matching
        to_invalidate = []
        query_words = set(query.lower().split())

        for cached_query in self.cache:
            cached_words = set(cached_query.lower().split())
            similarity = len(query_words & cached_words) / len(
                query_words | cached_words
            )
            if similarity > threshold:
                to_invalidate.append(cached_query)

        for q in to_invalidate:
            del self.cache[q]


class MainAgentOptimized(MainAgentAdvanced):
    """
    Production-optimized agent with learning and caching.

    Features:
    - Query result caching
    - Parallel branch exploration
    - GPU acceleration for embeddings
    - Evolution pattern learning
    - Performance monitoring
    - Async processing support
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        enable_cache: bool = True,
        enable_learning: bool = True,
        enable_async: bool = True,
        max_workers: int = 4,
    ):
        super().__init__(config)

        # Optimization components
        self.enable_cache = enable_cache
        self.query_cache = QueryCache() if enable_cache else None

        # Evolution tracking
        self.enable_learning = enable_learning
        self.evolution_tracker = EvolutionTracker(enable_learning=enable_learning)
        self.trajectory_analyzer = TrajectoryAnalyzer(self.evolution_tracker.pattern_db)

        # Async processing
        self.enable_async = enable_async
        self.executor = (
            ThreadPoolExecutor(max_workers=max_workers) if enable_async else None
        )

        # Performance monitoring
        self.metrics_history: List[OptimizationMetrics] = []

        # GPU optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name()}")

    async def process_question_async(self, question: str) -> Dict[str, Any]:
        """Async processing with optimization"""

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self.process_question_optimized, question
        )
        return result

    def process_question_optimized(self, question: str) -> Dict[str, Any]:
        """Optimized question processing with caching and learning"""

        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Check cache first
        cache_hits = 0
        cache_misses = 0

        if self.enable_cache and self.query_cache:
            cached_result = self.query_cache.get(question)
            if cached_result:
                cache_hits += 1
                logger.info(f"Cache hit for query: {question[:50]}...")

                # Still track metrics
                self._record_metrics(
                    processing_time=time.time() - start_time,
                    cycles=0,
                    branches=0,
                    memory_usage=self._get_memory_usage() - start_memory,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                )

                return cached_result

        cache_misses += 1

        # Get optimization suggestions from learned patterns
        query_type = QueryTypeClassifier.classify(question)
        initial_state = self._create_initial_state(question)

        if self.enable_learning:
            strategy = self.evolution_tracker.suggest_exploration_strategy(
                question, initial_state
            )
            self._apply_optimization_strategy(strategy)

        # Process with advanced features
        main_history = QueryTransformationHistory(initial_query=question)
        main_history.add_state(initial_state)

        # Parallel branch exploration if enabled
        if self.enable_async and self.executor:
            result = self._process_with_parallel_branches(question, main_history)
        else:
            result = self._process_with_advanced_transformation(question, main_history)

        # Track evolution pattern
        if self.enable_learning:
            final_score = self._calculate_result_quality(result)
            pattern = self.evolution_tracker.track_evolution(
                initial_query=question,
                transformation_history=main_history,
                branches=result.get("branches", []),
                final_score=final_score,
            )

            # Add pattern info to result
            result["evolution_pattern"] = pattern.to_dict()

            # Analyze trajectory
            if pattern.pattern_id:
                trajectory_metrics = self.trajectory_analyzer.analyze_trajectory(
                    pattern.pattern_id
                )
                result["trajectory_analysis"] = trajectory_metrics

        # Cache successful results
        if self.enable_cache and result.get("response"):
            self.query_cache.put(question, result)

        # Record performance metrics
        processing_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - start_memory

        self._record_metrics(
            processing_time=processing_time,
            cycles=len(main_history.states),
            branches=len(result.get("branches", [])),
            memory_usage=memory_usage,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

        # Add optimization info to result
        result["optimization_metrics"] = {
            "processing_time": processing_time,
            "cache_hit": cache_hits > 0,
            "gpu_accelerated": self.device.type == "cuda",
            "parallel_branches": self.enable_async,
        }

        return result

    def _process_with_parallel_branches(
        self, question: str, main_history: QueryTransformationHistory
    ) -> Dict[str, Any]:
        """Process with parallel branch exploration"""

        # Initial setup
        current_state = main_history.get_current_state()

        # Create branches
        branches = self._create_optimized_branches(current_state)

        if not branches:
            # Fallback to sequential
            return self._process_with_advanced_transformation(question, main_history)

        # Process branches in parallel
        import concurrent.futures

        branch_futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(branches)
        ) as executor:
            for branch in branches:
                future = executor.submit(
                    self._explore_branch_optimized,
                    branch,
                    self._get_current_knowledge_graph(),
                    [],  # Documents will be fetched per branch
                )
                branch_futures.append((branch, future))

        # Collect results
        branch_results = []
        for branch, future in branch_futures:
            try:
                insights = future.result(timeout=10.0)
                branch_results.append((branch, insights))
            except Exception as e:
                logger.error(f"Branch {branch.branch_id} failed: {e}")

        # Synthesize results
        return self._synthesize_parallel_results(question, main_history, branch_results)

    def _create_optimized_branches(self, current_state: QueryState) -> List:
        """Create branches optimized based on learned patterns"""

        # Get recommended branches from evolution tracker
        if self.enable_learning:
            suggestions = self.evolution_tracker.suggest_exploration_strategy(
                current_state.text, current_state
            )
            recommended = suggestions.get("recommended_branches", [])
        else:
            recommended = ["general", "theoretical", "practical"]

        # Create branches efficiently
        branches = []
        for i, direction in enumerate(recommended[:3]):  # Max 3 parallel branches
            branch = self._create_single_branch(
                current_state, direction, priority=1.0 / (i + 1)
            )
            branches.append(branch)

        return branches

    def _apply_optimization_strategy(self, strategy: Dict[str, Any]):
        """Apply learned optimization strategy"""

        # Set exploration temperature
        if "exploration_temperature" in strategy:
            self.query_transformer.adaptive_explorer.temperature = strategy[
                "exploration_temperature"
            ]

        # Set expected hops
        if "expected_hops" in strategy:
            self.max_cycles = min(self.max_cycles, strategy["expected_hops"] + 1)

        # Pre-warm cache with recommended concepts
        if "recommended_concepts" in strategy and self.l2_memory:
            for concept in strategy["recommended_concepts"][:3]:
                # Prefetch related episodes
                self.l2_memory.search_episodes(concept, k=3)

    def _calculate_result_quality(self, result: Dict[str, Any]) -> float:
        """Calculate quality score for learning"""

        score = 0.0

        # Response quality
        if result.get("response"):
            score += 0.3
            if len(result["response"]) > 100:
                score += 0.1

        # Insight quality
        insights = result.get("synthesis", {}).get("total_insights", [])
        score += min(len(insights) * 0.1, 0.3)

        # Confidence
        if result.get("transformation_history"):
            final_confidence = (
                result["transformation_history"]
                .get("states", [{}])[-1]
                .get("confidence", 0)
            )
            score += final_confidence * 0.2

        # Spike detection
        if result.get("spike_detected"):
            score += 0.2

        return min(score, 1.0)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _record_metrics(self, **kwargs):
        """Record performance metrics"""

        gpu_util = 0.0
        if self.device.type == "cuda":
            gpu_util = torch.cuda.utilization()

        metrics = OptimizationMetrics(
            query_processing_time=kwargs.get("processing_time", 0),
            transformation_cycles=kwargs.get("cycles", 0),
            branches_explored=kwargs.get("branches", 0),
            memory_usage_mb=kwargs.get("memory_usage", 0),
            cache_hits=kwargs.get("cache_hits", 0),
            cache_misses=kwargs.get("cache_misses", 0),
            gpu_utilization=gpu_util,
        )

        self.metrics_history.append(metrics)

        # Keep only recent metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""

        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-100:]  # Last 100 queries

        processing_times = [m.query_processing_time for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        cache_hit_rates = [m.cache_hit_rate for m in recent_metrics]

        return {
            "avg_processing_time": np.mean(processing_times),
            "p95_processing_time": np.percentile(processing_times, 95),
            "avg_memory_usage_mb": np.mean(memory_usage),
            "cache_hit_rate": np.mean(cache_hit_rates),
            "total_queries_processed": len(self.metrics_history),
            "gpu_accelerated": self.device.type == "cuda",
            "parallel_processing": self.enable_async,
        }

    def export_learned_patterns(self, output_path: str):
        """Export learned patterns for analysis"""
        from pathlib import Path

        if self.enable_learning:
            self.evolution_tracker.export_patterns(Path(output_path))

    def _create_initial_state(self, question: str) -> QueryState:
        """Create initial query state with GPU optimization"""

        # Generate embedding on GPU if available
        embedding = self.query_transformer.embedding_model.encode(
            question, convert_to_tensor=True, device=self.device
        )

        return QueryState(
            text=question, embedding=embedding, stage="initial", confidence=0.1
        )

    def _explore_branch_optimized(self, branch, knowledge_graph, documents):
        """Optimized branch exploration"""

        # This would be called in parallel
        try:
            # Quick exploration with early stopping
            insights = []
            for step in range(2):
                # Transform state
                new_state = self.query_transformer.transform_query(
                    branch.current_state, knowledge_graph, documents
                )

                branch.current_state = new_state

                # Early stopping if high confidence
                if new_state.confidence > 0.8:
                    insights.extend(new_state.insights)
                    break

                if new_state.insights:
                    insights.extend(new_state.insights[-1:])

            return insights

        except Exception as e:
            logger.error(f"Branch exploration failed: {e}")
            return []

    def _synthesize_parallel_results(
        self,
        question: str,
        main_history: QueryTransformationHistory,
        branch_results: List[Tuple],
    ) -> Dict[str, Any]:
        """Synthesize results from parallel branches"""

        # Aggregate insights
        all_insights = []
        successful_branches = []

        for branch, insights in branch_results:
            if insights:
                all_insights.extend(insights)
                successful_branches.append(branch)

        # Create final state
        final_state = main_history.get_current_state()
        final_state.stage = "insight" if all_insights else "exploring"
        final_state.confidence = min(0.9, 0.3 + 0.2 * len(all_insights))

        for insight in all_insights:
            final_state.add_insight(insight)

        main_history.add_state(final_state)

        # Build result
        result = {
            "response": all_insights[0] if all_insights else "No insights found",
            "transformation_history": main_history.to_dict(),
            "branches": successful_branches,
            "synthesis": {
                "total_insights": all_insights,
                "num_branches_explored": len(branch_results),
                "successful_branches": len(successful_branches),
            },
        }

        return result

    def _create_single_branch(self, parent_state, direction, priority):
        """Create a single branch efficiently"""
        from ..query_transformation.enhanced_query_transformer import QueryBranch

        branch_state = QueryState(
            text=parent_state.text,
            embedding=parent_state.embedding.clone()
            if parent_state.embedding is not None
            else None,
            stage="exploring",
            confidence=parent_state.confidence * 0.8,
            absorbed_concepts=parent_state.absorbed_concepts.copy(),
        )

        return QueryBranch(
            branch_id=f"branch_{direction}_{time.time()}",
            parent_state=parent_state,
            current_state=branch_state,
            exploration_direction=direction,
            priority=priority,
        )

    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # Override base process_question to use optimized version
    def process_question(self, question: str) -> Dict[str, Any]:
        """Process question with full optimization"""
        return self.process_question_optimized(question)
