#!/usr/bin/env python
import os
from pathlib import Path
from datasets import load_dataset
import nltk
import nltk.tokenize

from insightspike.layer2_memory_manager import Memory

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_FILE = RAW_DIR / "wiki_sentences.txt"

print("Downloading Wikipedia articles...")
dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)

nltk.download('punkt')

sentences = []
sources = []
count = 0
max_articles = 100

for item in dataset:
    title = item.get("title", "").lower()
    if "physic" in title:  # "physics", "physicist", etc.
        sents = [s.strip() for s in nltk.tokenize.sent_tokenize(item["text"]) if s.strip()]
        if sents:
            sentences.extend(sents)
            sources.extend([item.get("title", "unknown")] * len(sents))
            count += 1
    if count >= max_articles:
        break

with open(RAW_FILE, "w") as f:
    for s in sentences:
        f.write(s.replace("\n", " ") + "\n")

print(f"Saved {len(sentences)} sentences from {count} physics-related articles to {RAW_FILE}")

print("Building Memory and saving FAISS index & metadata...")
mem = Memory.build(sentences)
mem.save()

import numpy as np
EMBED_DIR = Path("data/embedding")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
vecs = np.vstack([e.vec for e in mem.episodes])
np.save(EMBED_DIR / "input.npy", vecs)
np.save(EMBED_DIR / "sources.npy", np.array(sources))

print("Done! Memory, index, embedding vectors, and sources saved.")
