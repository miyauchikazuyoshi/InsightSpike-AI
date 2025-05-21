"""Build PyG similarity graph"""
from pathlib import Path
import numpy as np, torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

from .config import GRAPH_FILE, SIM_THRESHOLD

__all__ = ["build_graph", "load_graph", "save_graph"]

def save_graph(data, path: Path | None = None):
    """Save graph data to file."""
    dest = path or GRAPH_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, dest)
    return dest

def load_graph(path: Path | None = None):
    """Load graph data from file."""
    src = path or GRAPH_FILE
    if not src.exists():
        raise FileNotFoundError(f"Graph file not found at {src}")
    return torch.load(src)

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
    
    # 保存処理を save_graph に委譲
    if dest is not None:
        save_graph(data, dest)
    
    return data, edge_index
