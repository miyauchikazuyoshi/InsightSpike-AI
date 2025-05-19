"""Corpus loader"""
from pathlib import Path
from typing import List

from .config import DATA_DIR
from .utils import iter_text, clean_text

__all__ = ["load_corpus"]

def load_corpus(path: Path | None = None) -> List[str]:
    p = Path(path)
    if p.is_file():
        # 各行を1文書として扱う
        return [line.strip() for line in p.read_text("utf-8").splitlines() if line.strip()]
    root = path or DATA_DIR
    docs = [clean_text(p.read_text("utf-8")) for p in iter_text(root)]
    if not docs:
        raise RuntimeError(f"No docs under {root}")
    return docs
