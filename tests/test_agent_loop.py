from unittest.mock import patch
from insightspike.agent_loop import cycle
from insightspike.layer2_memory_manager import Memory
import networkx as nx

def test_cycle_runs():
    with open("data/raw/test_sentences.txt") as f:
        docs = [line.strip() for line in f if line.strip()]
    with patch("insightspike.layer2_memory_manager.Memory.train_index"), \
         patch("insightspike.layer2_memory_manager.Memory.search", return_value=[(0.9, i) for i in range(10)]), \
         patch("insightspike.graph_metrics.KMeans"), \
         patch("insightspike.graph_metrics.silhouette_score", return_value=0.5):
        mem = Memory.build(docs)
        result = cycle(mem, "test question", None)
        assert isinstance(result, nx.Graph)
