"""InsightSpike package metadata"""
class About:
    NAME = "InsightSpike-AI"
    VERSION = "0.7-Eureka"

# モジュール全体を先にインポート
try:
    from . import layer3_graph_pyg
    # 具体的な関数をモジュールから取得
    build_graph = layer3_graph_pyg.build_graph
    load_graph = layer3_graph_pyg.load_graph
    save_graph = layer3_graph_pyg.save_graph
except ImportError:
    # テスト用のスタブ関数
    def build_graph(*args, **kwargs): pass
    def load_graph(*args, **kwargs): pass
    def save_graph(*args, **kwargs): pass

# エクスポートする名前を指定
__all__ = ["About", "build_graph", "load_graph", "save_graph"]
