"""Corpus loader"""
from pathlib import Path
from typing import List

from ..core.config import get_config
from ..utils.text_utils import iter_text, clean_text

__all__ = ["load_corpus"]

def load_corpus(path: Path | None = None) -> List[str]:
    config = get_config()
    root = path or config.paths.data_dir
    p = Path(root)
    if p.is_file():
        try:
            return [line.strip() for line in p.open(encoding="utf-8") if line.strip()]
        except FileNotFoundError:
            print(f"ファイルが見つかりません: {path}")
            return []
        except UnicodeDecodeError:
            print(f"文字コードエラー: {path} はUTF-8で保存してください")
            return []
    config = get_config()
    root = path or config.paths.data_dir
    docs = []
    for file in iter_text(root):
        try:
            with file.open(encoding="utf-8") as f:
                docs.extend([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"ファイルが見つかりません: {file}")
        except UnicodeDecodeError:
            print(f"文字コードエラー: {file} はUTF-8で保存してください")
    if not docs:
        raise RuntimeError(f"No docs under {root}")
    return docs
