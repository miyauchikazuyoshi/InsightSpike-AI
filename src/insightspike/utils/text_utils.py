"""Text processing utilities"""
from pathlib import Path
from typing import Iterable
import re

__all__ = ["iter_text", "clean_text"]


def iter_text(root: Path, suffix: str = ".txt") -> Iterable[Path]:
    """Yield all text files under *root*."""
    yield from root.rglob(f"*{suffix}")


def clean_text(text: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", text).strip()
