"""Corpus loader"""
from pathlib import Path
from typing import List

from .config import DATA_DIR
from .utils import iter_text, clean_text

__all__ = ["load_corpus"]

def load_corpus(path: Path | None = None) -> List[str]:
    p = Path(path)
    if p.is_file():
        try:
            return [line.strip() for line in p.open(encoding="utf-8") if line.strip()]
        except FileNotFoundError:
            print(f"ファイルが見つかりません: {path}")
            return []
        except UnicodeDecodeError:
            print(f"文字コードエラー: {path} はUTF-8で保存してください")
            return []
    root = path or DATA_DIR
    docs = [clean_text(p.read_text("utf-8")) for p in iter_text(root)]
    if not docs:
        raise RuntimeError(f"No docs under {root}")
    return docs
