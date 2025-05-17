"""L2 – C‑value memory with IVF‑PQ"""

from __future__ import annotations
from pathlib import Path
from typing import List
import json

import faiss
import numpy as np

from .embedder import get_model
from .config import INDEX_FILE

__all__ = ["Episode", "Memory"]


class Episode:
    """Single memory entry (vector + text + C‑value)."""

    def __init__(self, vec: np.ndarray, text: str, c: float = 0.5):
        self.vec = vec.astype(np.float32)
        self.text = text
        self.c = float(c)


class Memory:
    """IVF‑PQ index with C‑value reinforcement."""

    def __init__(self, dim: int):
        nlist, m = 256, 16
        self.index = faiss.index_factory(dim, f"IVF{nlist},PQ{m}")
        self.episodes: List[Episode] = []
        self.dim = dim

    # ── construction ────────────────────────────────
    @classmethod
    def build(cls, docs: List[str]):
        model = get_model()
        vecs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
        mem = cls(vecs.shape[1])
        for v, t in zip(vecs, docs):
            mem.episodes.append(Episode(v, t))
        mem.train_index()
        return mem

    def train_index(self):
        vecs = np.vstack([e.vec for e in self.episodes])
        self.index.reset()
        self.index.train(vecs)
        self.index.add(vecs)

    # ── persistence ────────────────────────────────
    def save(self, path: Path = INDEX_FILE):
        faiss.write_index(self.index, str(path))
        meta_path = path.with_suffix(".json")
        meta = [{"c": e.c, "text": e.text} for e in self.episodes]
        meta_path.write_text(json.dumps(meta, ensure_ascii=False))
        return meta_path

    @classmethod
    def load(cls, path: Path = INDEX_FILE):
        """Load memory state from disk."""
        index = faiss.read_index(str(path))
        meta = json.loads(path.with_suffix(".json").read_text())
        mem = cls(index.d)
        mem.index = index

        # Reconstruct stored vectors so that the memory can be retrained
        # without losing information after loading.  If reconstruction is
        # unavailable, fall back to zero vectors as before.

        if hasattr(index, "reconstruct_n"):
            vecs = index.reconstruct_n(0, index.ntotal)
        elif hasattr(index, "reconstruct"):
            vecs = [index.reconstruct(i) for i in range(index.ntotal)]
        else:
            vecs = [np.zeros(index.d, dtype=np.float32) for _ in meta]

        mem.episodes = [Episode(v, m["text"], m["c"]) for v, m in zip(vecs, meta)]
        return mem

    # ── retrieval ──────────────────────────────────
    def search(self, q: np.ndarray, top_k: int = 5, gamma: float = 1.0):
        D, indices = self.index.search(q.astype(np.float32), top_k * 5)
        scored: list[tuple[float, int]] = []
        for d, i in zip(D[0], indices[0]):
            c = self.episodes[i].c
            scored.append((float(d) * (c**gamma), i))
        scored.sort(reverse=True)
        return scored[:top_k]

    # ── C‑value update ─────────────────────────────
    def update_c(self, idxs: List[int], reward: float, eta: float = 0.1):
        for i in idxs:
            e = self.episodes[i]
            e.c = max(0.0, min(1.0, e.c + eta * reward))
        # ── add new episode from LLM output ────────────────

    def add_episode(self, vec: np.ndarray, text: str, c_init: float = 0.2):
        self.episodes.append(Episode(vec, text, c_init))
        self.train_index()

    # ── merge similar episodes ────────────────────────
    def merge(self, idxs: list[int], gain: float = 0.1):
        if len(idxs) < 2:
            return
        vecs = np.vstack([self.episodes[i].vec for i in idxs])
        texts = [self.episodes[i].text for i in idxs]
        new_vec = vecs.mean(axis=0)
        new_text = " / ".join(texts)
        # 利得: 新ノード C = max(old)+gain, 旧は減衰
        max_c = max(self.episodes[i].c for i in idxs)
        self.add_episode(new_vec, new_text, min(1.0, max_c + gain))
        for i in idxs:
            self.episodes[i].c *= 0.5  # 下げる
        self.train_index()

    # ── split incoherent episode ───────────────────────
    def split(self, idx: int):
        ep = self.episodes[idx]
        # ここでは dummy: vec を 2 コピー + 小ノイズ
        noise = np.random.randn(2, self.dim) * 1e-4
        for v in ep.vec + noise:
            self.add_episode(v, ep.text, ep.c / 2)
        ep.c *= 0.5
        self.train_index()

    # ── prune low-C & inactive episodes ────────────────
    def prune(self, c_thresh: float, inactive_n: int):
        now = getattr(self, "_loop_count", 0)
        self._loop_count = now + 1
        keep = []
        for ep in self.episodes:
            age = getattr(ep, "last", now)
            if ep.c >= c_thresh or (now - age) < inactive_n:
                keep.append(ep)
        if len(keep) != len(self.episodes):
            self.episodes = keep
            self.train_index()
