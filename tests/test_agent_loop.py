from unittest.mock import patch
from insightspike.agent_loop import cycle
from insightspike.layer2_memory_manager import Memory
import networkx as nx

def test_cycle_runs():
    with patch("insightspike.layer2_memory_manager.Memory.train_index"):
        mem = Memory.build(["a", "b"])
        result = cycle(mem, "test question", None)
        assert isinstance(result, nx.Graph)
