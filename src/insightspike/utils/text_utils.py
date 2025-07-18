"""Text processing utilities"""
import re
from pathlib import Path
from typing import Iterable

__all__ = ["iter_text", "clean_text", "jaccard_similarity"]


def iter_text(root: Path, suffix: str = ".txt") -> Iterable[Path]:
    """Yield all text files under *root*."""
    yield from root.rglob(f"*{suffix}")


def clean_text(text: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", text).strip()


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    The Jaccard similarity coefficient is defined as the size of the intersection
    divided by the size of the union of the sample sets.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Tokenize by splitting on whitespace and converting to lowercase
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Handle edge cases
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union > 0 else 0.0
