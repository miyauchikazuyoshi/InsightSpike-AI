#!/usr/bin/env python
import os
from pathlib import Path
from datasets import load_dataset
import nltk

from insightspike.layer2_memory_manager import Memory

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_FILE = RAW_DIR / "wiki_sentences.txt"

# 1. Wikipediaから文章を取得
print("Downloading Wikipedia sentences...")
dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]", trust_remote_code=True)

# nltkで文分割
nltk.download('punkt')
nltk.download('punkt_tab')

sentences = []
sources = []
for item in dataset:
    # item["text"] を文単位で分割
    sents = [s.strip() for s in nltk.tokenize.sent_tokenize(item["text"]) if s.strip()]
    sentences.extend(sents)
    sources.extend([item.get("title", "unknown")] * len(sents))  # 取得元タイトルを記録

# テキストファイルとして保存
with open(RAW_FILE, "w") as f:
    for s in sentences:
        f.write(s.replace("\n", " ") + "\n")

print(f"Saved {len(sentences)} sentences to {RAW_FILE}")

# 2. Memoryを構築して保存
print("Building Memory and saving FAISS index & metadata...")
mem = Memory.build(sentences)
mem.save()  # デフォルトのINDEX_FILEに保存

# 3. embeddingベクトルとsourcesをnpyで保存
import numpy as np
EMBED_DIR = Path("data/embedding")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
vecs = np.vstack([e.vec for e in mem.episodes])
np.save(EMBED_DIR / "input.npy", vecs)
np.save(EMBED_DIR / "sources.npy", np.array(sources))

print("Done! Memory, index, embedding vectors, and sources saved.")
