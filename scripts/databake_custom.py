#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np

from insightspike.layer2_memory_manager import Memory

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/databake_custom.py <your_text_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    # 1. テキストファイルから文章を読み込む
    with open(input_path, "r") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(sentences)} sentences from {input_path}")

    # 2. Memoryを構築して保存
    print("Building Memory and saving FAISS index & metadata...")
    mem = Memory.build(sentences)
    mem.save()  # デフォルトのINDEX_FILEに保存

    # 3. embeddingベクトルをnpyで保存
    out_dir = Path("data/embedding")
    out_dir.mkdir(parents=True, exist_ok=True)
    vecs = np.vstack([e.vec for e in mem.episodes])
    np.save(out_dir / "input.npy", vecs)

    print("Done! Memory, index, and embedding vectors saved.")

if __name__ == "__main__":
    main()