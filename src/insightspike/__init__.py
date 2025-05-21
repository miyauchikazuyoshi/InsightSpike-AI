"""InsightSpike package metadata"""
class About:
    NAME = "InsightSpike-AI"
    VERSION = "0.7-Eureka"

from .layer3_graph_pyg import build_graph, load_graph, save_graph

__all__ = ["About", "build_graph", "load_graph", "save_graph"]
