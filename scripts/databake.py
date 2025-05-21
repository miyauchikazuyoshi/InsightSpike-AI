'#!/usr/bin/env python'
import os
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from insightspike.layer2_memory_manager import Memory

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_FILE = RAW_DIR / "wiki_sentences.txt"

# 1. Wikipediaから文章を取得
print("Downloading Wikipedia sentences...")
dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]", trust_remote_code=True)
sentences = [item["text"] for item in dataset if item["text"].strip()]
with open(RAW_FILE, "w") as f:
    for s in sentences:
        f.write(s.replace("\n", " ") + "\n")

# 2. Memoryを構築して保存
print("Building Memory and saving FAISS index & metadata...")
mem = Memory.build(sentences)
mem.save()  # デフォルトのINDEX_FILEに保存

print("Done! Memory and index saved.")
