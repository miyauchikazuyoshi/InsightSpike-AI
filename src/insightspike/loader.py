"""Corpus loader"""
from pathlib import Path
from typing import List

from .config import DATA_DIR
from .utils import iter_text, clean_text

__all__ = ["load_corpus"]

def load_corpus(root: Path | None = None) -> List[str]:
    root = root or DATA_DIR
    docs = [clean_text(p.read_text("utf-8")) for p in iter_text(root)]
    if not docs:
        raise RuntimeError(f"No docs under {root}")
    return docs
