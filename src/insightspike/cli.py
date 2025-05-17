"""CLI entrypoints"""
from pathlib import Path
import typer, numpy as np
from rich import print

from .embedder              import get_model
from .loader                import load_corpus
from .layer2_memory_manager import Memory
from .layer3_graph_pyg      import build_graph
from .agent_loop            import cycle

app = typer.Typer()

@app.command()
def embed(path: Path | None = typer.Option(None)):
    docs = load_corpus(path)
    Memory.build(docs).save()   # 戻り値を無視
    print(f"Embedded {len(docs)} docs.")

@app.command()
def graph():
    mem = Memory.load(); vecs = np.vstack([e.vec for e in mem.episodes])
    build_graph(vecs)
    print("Graph initialised.")

@app.command()
def loop(q: str):
    try:
        mem = Memory.load()
    except FileNotFoundError:
        mem = Memory.build(load_corpus()); mem.save()

    try:
        import torch
        g_old = torch.load("data/graph_loop.pt")
    except (FileNotFoundError, RuntimeError):
        g_old = None

    g_new = cycle(mem, q, g_old)
    import torch
    torch.save(g_new, "data/graph_loop.pt")
    mem.save()

if __name__ == "__main__":
    app()