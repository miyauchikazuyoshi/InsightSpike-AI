"""Base class for RAG baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_loader import HotpotQAExample


@dataclass
class RAGResult:
    """Result from a RAG system."""

    answer: str
    retrieved_facts: list[tuple[str, int]]  # (title, sent_id)
    latency_ms: float
    metadata: dict


class BaseRAG(ABC):
    """Abstract base class for RAG systems."""

    name: str = "BaseRAG"

    @abstractmethod
    def setup(self, examples: list[HotpotQAExample]) -> None:
        """Setup the RAG system with the full context.

        This is called once before processing queries.
        For static systems, this builds the index.
        For dynamic systems, this initializes the empty graph.
        """
        pass

    @abstractmethod
    def process(self, example: HotpotQAExample) -> RAGResult:
        """Process a single question and return the result."""
        pass

    def reset(self) -> None:
        """Reset the system state (for dynamic systems)."""
        pass
