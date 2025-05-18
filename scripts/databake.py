import os
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_FILE = RAW_DIR / "wiki_sentences.txt"

# 1. Wikipediaから文章を取得
print("Downloading Wikipedia sentences...")
dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]")
sentences = [item["text"] for item in dataset if item["text"].strip()]
with open(RAW_FILE, "w") as f:
    for s in sentences:
        f.write(s.replace("\n", " ") + "\n")

# 2. sentence-transformersでベクトル化
print("Embedding sentences...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64)
np.save(RAW_DIR / "wiki_embeddings.npy", embeddings)

# 3. faissでインデックス化
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, str(RAW_DIR / "wiki_faiss.index"))

print("Done! Data saved in:", RAW_DIR)