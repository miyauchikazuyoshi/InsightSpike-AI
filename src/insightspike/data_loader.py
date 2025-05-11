import os
from typing import List

def load_raw_documents(raw_dir: str) -> List[str]:
    """
    指定したディレクトリ内のテキストファイル (*.txt) をすべて読み、
    各ファイルの中身を文字列としてリストで返す。

    raw_dir: "data/raw" のようなパス
    return: [ "ドキュメント1の本文", "ドキュメント2の本文", ... ]
    """
    docs = []
    # 拡張子 .txt のみ対象
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(raw_dir, fname)
        with open(path, encoding="utf-8") as f:
            docs.append(f.read().strip())
    return docs