"""InsightSpike package metadata"""
class About:
    NAME = "InsightSpike-AI"
    VERSION = "0.7-Eureka"

# 実装をモジュールから分離し、直接定義
def build_graph(vectors, dest=None):
    """グラフ構築のスタブ関数"""
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from sklearn.metrics.pairwise import cosine_similarity
    from .config import SIM_THRESHOLD
    
    # コア機能を最小限の依存関係で実装
    n = len(vectors)
    sims = cosine_similarity(vectors)
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= SIM_THRESHOLD:
                src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst])
    data = Data(x=torch.tensor(vectors, dtype=torch.float32), edge_index=edge_index)
    
    if dest is not None:
        save_graph(data, dest)
    
    return data, edge_index

def load_graph(path=None):
    """グラフ読み込みのスタブ関数"""
    import torch
    from torch_geometric.data import Data
    from pathlib import Path
    from .config import GRAPH_FILE
    
    src = path or GRAPH_FILE
    if not Path(src).exists():
        raise FileNotFoundError(f"Graph file not found at {src}")
    return torch.load(src)

def save_graph(data, path=None):
    """グラフ保存のスタブ関数"""
    import torch
    from pathlib import Path
    from .config import GRAPH_FILE
    
    dest = path or GRAPH_FILE
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, dest)
    return dest

# エクスポートする名前を明示
__all__ = ["About", "build_graph", "load_graph", "save_graph"]
