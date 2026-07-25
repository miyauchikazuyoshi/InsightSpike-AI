"""Pure selection and aggregation for ``MainAgent`` cycle results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class CycleAggregationStats:
    """Scalar agent state captured before the current query is recorded."""

    memory_episodes: int
    total_processed: int


@dataclass(frozen=True)
class CycleAggregation(Generic[T]):
    """Selected source cycle and newly-created public result."""

    selected: T
    result: T


@dataclass(frozen=True)
class CycleAggregationDraft(Generic[T]):
    """Selected cycle plus aggregate graph metadata before side effects."""

    selected: T
    graph_analysis: dict


class CycleResultAggregator(Generic[T]):
    """Select the highest-quality cycle and attach aggregate metadata."""

    def __init__(self, result_factory: Callable[..., T]):
        self._result_factory = result_factory

    def aggregate(
        self,
        results: Sequence[T],
        *,
        converged: bool,
        include_history: bool,
        stats: CycleAggregationStats,
    ) -> CycleAggregation[T]:
        """Prepare and immediately materialize an aggregate result."""

        draft = self.prepare(
            results,
            converged=converged,
            include_history=include_history,
            stats=stats,
        )
        return CycleAggregation(
            selected=draft.selected,
            result=self.build_result(draft),
        )

    def prepare(
        self,
        results: Sequence[T],
        *,
        converged: bool,
        include_history: bool,
        stats: CycleAggregationStats,
    ) -> CycleAggregationDraft[T]:
        """Select a cycle and create metadata without constructing a result."""

        if not results:
            raise ValueError("results must not be empty")

        # ``max`` deliberately keeps the first result when qualities tie.
        selected = max(
            results,
            key=lambda result: result.reasoning_quality,
        )
        graph_analysis = selected.graph_analysis.copy()
        graph_analysis.update(
            {
                "total_cycles": len(results),
                "converged": converged,
                "cycle_history": (
                    [result.to_dict() for result in results]
                    if include_history
                    else []
                ),
                "agent_stats": {
                    "memory_episodes": stats.memory_episodes,
                    "total_processed": stats.total_processed,
                },
            }
        )

        return CycleAggregationDraft(
            selected=selected,
            graph_analysis=graph_analysis,
        )

    def build_result(
        self,
        draft: CycleAggregationDraft[T],
    ) -> T:
        """Materialize from the selected cycle's current public fields."""

        selected = draft.selected
        return self._result_factory(
            question=selected.question,
            retrieved_documents=selected.retrieved_documents,
            graph_analysis=draft.graph_analysis,
            response=selected.response,
            reasoning_quality=selected.reasoning_quality,
            spike_detected=selected.spike_detected,
            error_state=selected.error_state,
            cycle_number=selected.cycle_number,
            success=selected.success,
        )


__all__ = [
    "CycleAggregation",
    "CycleAggregationDraft",
    "CycleAggregationStats",
    "CycleResultAggregator",
]
