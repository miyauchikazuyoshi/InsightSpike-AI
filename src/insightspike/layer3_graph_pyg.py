"""Build PyG similarity graph"""
from pathlib import Path  # ここを明示的に最初に
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

from .config import GRAPH_FILE, SIM_THRESHOLD

__all__ = ["build_graph", "load_graph", "save_graph"]

def save_graph(data, path: Optional[Path] = None):  # 型アノテーション変更
    """Save graph data to file."""
    import torch
    dest = path or GRAPH_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Data オブジェクトを分解して保存する
    try:
        # 通常の方法でまず試す
        torch.save(data, dest)
    except Exception as e:
        if hasattr(data, 'edge_index') and hasattr(data, 'x'):
            # エラー発生時: PyG Data オブジェクトを分解
            save_data = {
                'x': data.x.detach().cpu().numpy() if torch.is_tensor(data.x) else data.x,
                'edge_index': data.edge_index.detach().cpu().numpy() if torch.is_tensor(data.edge_index) else data.edge_index
            }
            torch.save(save_data, dest)
        else:
            # その他のオブジェクトの場合はエラーを再発生
            raise e
    
    return dest

def load_graph(path: Optional[Path] = None):  # 型アノテーション変更
    """Load graph data from file."""
    import torch
    from torch_geometric.data import Data
    src = path or GRAPH_FILE
    if not src.exists():
        raise FileNotFoundError(f"Graph file not found at {src}")
    
    loaded = torch.load(src)
    
    # 保存されたデータが辞書形式かチェック
    if isinstance(loaded, dict) and 'x' in loaded and 'edge_index' in loaded:
        # PyTorch Geometric Data オブジェクトに変換
        x = torch.tensor(loaded['x']) if not torch.is_tensor(loaded['x']) else loaded['x']
        edge_index = torch.tensor(loaded['edge_index']) if not torch.is_tensor(loaded['edge_index']) else loaded['edge_index']
        return Data(x=x, edge_index=edge_index)
    
    # そのまま返す
    return loaded

def build_graph(vectors: np.ndarray, dest: Optional[Path] = None):  # 型アノテーション変更
    import torch
    from torch_geometric.data import Data
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
