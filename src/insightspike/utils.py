"""Generic helpers"""
import re
from pathlib import Path
from typing import Iterable

__all__ = ["iter_text", "clean_text", "create_mock_components"]


def iter_text(root: Path, suffix: str = ".txt") -> Iterable[Path]:
    """Yield all text files under *root*."""
    yield from root.rglob(f"*{suffix}")


def clean_text(text: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", text).strip()


def create_mock_components():
    """Create mock components for CI/LITE mode compatibility."""

    class MockGraphEditDistance:
        def calculate_distance(self, g1, g2):
            return 1.0

    class MockInformationGain:
        def calculate_gain(self, *args):
            return 0.5

    class MockInsightDetector:
        def detect_insights(self, *args):
            return []

    return MockGraphEditDistance, MockInformationGain, MockInsightDetector
