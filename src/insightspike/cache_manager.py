from pathlib import Path

GRAPH_PATH = Path("data/graph_pyg.pt")
MEMORY_JSON = Path("data/memory.json")

# Safe PyTorch import
try:
    import torch
    TORCH_AVAILABLE = hasattr(torch, 'save') and hasattr(torch, 'load')
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

def save_cache(graph, memory):
    if TORCH_AVAILABLE:
        torch.save(graph, GRAPH_PATH)
    else:
        # Fallback: save as pickle or other format
        import pickle
        with open(GRAPH_PATH, 'wb') as f:
            pickle.dump(graph, f)
    
    memory.save(MEMORY_JSON)

def load_cache():
    if GRAPH_PATH.exists() and MEMORY_JSON.exists():
        if TORCH_AVAILABLE:
            graph = torch.load(GRAPH_PATH)
        else:
            # Fallback: load as pickle
            import pickle
            with open(GRAPH_PATH, 'rb') as f:
                graph = pickle.load(f)
        # memory = Memory.load(MEMORY_JSON)  # 実装に合わせて
        return graph #, memoryort Path

GRAPH_PATH = Path("data/graph_pyg.pt")
MEMORY_JSON = Path("data/memory.json")

def save_cache(graph, memory):
    if TORCH_AVAILABLE:
        import torch
        torch.save(graph, GRAPH_PATH)
    else:
        # Fallback to pickle for non-PyTorch environments
        import pickle
        with open(GRAPH_PATH, 'wb') as f:
            pickle.dump(graph, f)
    memory.save(MEMORY_JSON)

def load_cache():
    if GRAPH_PATH.exists() and MEMORY_JSON.exists():
        if TORCH_AVAILABLE:
            import torch
            graph = torch.load(GRAPH_PATH)
        else:
            # Fallback to pickle for non-PyTorch environments
            import pickle
            with open(GRAPH_PATH, 'rb') as f:
                graph = pickle.load(f)
        # memory = Memory.load(MEMORY_JSON)  # 実装に合わせて
        return graph #, memory
    else:
        return None

def clear_cache():
    if GRAPH_PATH.exists():
        GRAPH_PATH.unlink()
    if MEMORY_JSON.exists():
        MEMORY_JSON.unlink()
        