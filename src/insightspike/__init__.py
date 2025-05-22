"""InsightSpike package metadata"""
class About:
    NAME = "InsightSpike-AI"
    VERSION = "0.7-Eureka"

# 実装をモジュールから分離し、直接定義
def build_base_graph(vectors, sim_threshold):
    # 全組み合わせでcosine_similarity
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    from torch_geometric.data import Data

    sims = cosine_similarity(vectors)
    src, dst = [], []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= sim_threshold:
                src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst])
    data = Data(x=torch.tensor(vectors, dtype=torch.float32), edge_index=edge_index)
    return data, edge_index

def build_graph(vectors, top_k=10):
    # faiss等で近傍探索し、近傍のみでグラフ再編集
    import faiss
    import numpy as np
    import torch
    from torch_geometric.data import Data

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype=np.float32))
    D, I = index.search(np.array(vectors, dtype=np.float32), top_k+1)
    src, dst = [], []
    n = len(vectors)
    for i in range(n):
        for j in I[i][1:]:  # 0番目は自分自身
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst])
    data = Data(x=torch.tensor(vectors, dtype=torch.float32), edge_index=edge_index)
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
__all__ = ["About", "build_base_graph", "build_graph", "load_graph", "save_graph"]
