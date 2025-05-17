"""Build PyG similarity graph"""
from pathlib import Path
import numpy as np, torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

from .config import GRAPH_FILE, SIM_THRESHOLD

__all__ = ["build_graph", "load_graph"]

def build_graph(vectors: np.ndarray, dest: Path | None = None):
    n = len(vectors)
    sims = cosine_similarity(vectors)
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= SIM_THRESHOLD:
                src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst])
    data = Data(x=torch.tensor(vectors, dtype=torch.float32), edge_index=edge_index)
    path = dest or GRAPH_FILE; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data, edge_index

def load_graph(path: Path | None = None):
    import torch
    return torch.load(path or GRAPH_FILE)