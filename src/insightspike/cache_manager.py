from pathlib import Path
import torch

GRAPH_PATH = Path("data/graph_pyg.pt")
MEMORY_JSON = Path("data/memory.json")

def save_cache(graph, memory):
    torch.save(graph, GRAPH_PATH)
    memory.save(MEMORY_JSON)

def load_cache():
    if GRAPH_PATH.exists() and MEMORY_JSON.exists():
        graph = torch.load(GRAPH_PATH)
        # memory = Memory.load(MEMORY_JSON)  # 実装に合わせて
        return graph #, memory
    else:
        return None

def clear_cache():
    if GRAPH_PATH.exists():
        GRAPH_PATH.unlink()
    if MEMORY_JSON.exists():
        MEMORY_JSON.unlink()