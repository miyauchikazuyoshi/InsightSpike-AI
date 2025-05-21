"""InsightSpike package metadata"""
class About:
    NAME = "InsightSpike-AI"
    VERSION = "0.7-Eureka"

# 直接関数をスタブとして定義（モジュールのインポートを回避）
def build_graph(vectors, dest=None):
    """スタブ実装：グラフを構築"""
    try:
        # 実際のモジュールが利用可能な場合は、そちらを使用
        from .layer3_graph_pyg import build_graph as real_build_graph
        return real_build_graph(vectors, dest)
    except (ImportError, AttributeError):
        # フォールバック実装（テスト用）
        from torch_geometric.data import Data
        import torch
        return Data(x=torch.tensor(vectors)), None

def load_graph(path=None):
    """スタブ実装：グラフを読み込み"""
    try:
        from .layer3_graph_pyg import load_graph as real_load_graph
        return real_load_graph(path)
    except (ImportError, AttributeError):
        # フォールバック
        return None

def save_graph(data, path=None):
    """スタブ実装：グラフを保存"""
    try:
        from .layer3_graph_pyg import save_graph as real_save_graph
        return real_save_graph(data, path)
    except (ImportError, AttributeError):
        # フォールバック
        return path

# エクスポートする名前を明示
__all__ = ["About", "build_graph", "load_graph", "save_graph"]
