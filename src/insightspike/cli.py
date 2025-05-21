"""CLI entrypoints"""
from pathlib import Path
import typer, numpy as np
from rich import print

from .embedder              import get_model
from .loader                import load_corpus
from .layer2_memory_manager import Memory
# try-except で両方のインポート方法を試す
try:
    from .layer3_graph_pyg import build_graph, load_graph, save_graph
except (ImportError, AttributeError):
    # 直接モジュール全体をインポート
    from . import layer3_graph_pyg
    build_graph = layer3_graph_pyg.build_graph
    load_graph = layer3_graph_pyg.load_graph 
    save_graph = layer3_graph_pyg.save_graph
from .agent_loop            import cycle
from insightspike import cache_manager
from .graph_metrics import delta_ged, delta_ig

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
def loop(q: str, k: int = 15):
    """質問に対して推論ループを実行"""
    mem = Memory.load()
    
    try:
        g_old = torch.load("data/graph_loop.pt")
    except (FileNotFoundError, RuntimeError):
        g_old = None
    
    # kパラメータをcycle関数に渡す
    g_new = cycle(mem, q, g_old, top_k=k)
    # 残りのコード...

@app.command()
def cache(action: str):
    if action == "save":
        # obtain or build memory
        try:
            memory = Memory.load()
        except FileNotFoundError:
            memory = Memory.build(load_corpus())

        # obtain or build graph
        try:
            graph = load_graph()
        except Exception:  # FileNotFoundError or corrupted graph
            vecs = np.vstack([e.vec for e in memory.episodes])
            graph, _ = build_graph(vecs)

        cache_manager.save_cache(graph, memory)
    elif action == "load":
        cache_manager.load_cache()
    elif action == "clear":
        cache_manager.clear_cache()
    else:
        print("Unknown action")

@app.command()
def list_memory_files():
    import os
    os.system("ls -lh data/")

@app.command()
def reorganize(iterations: int = 5):
    """インプットなしでメモリの再編成とグラフ最適化を実行"""
    from .layer3_graph_pyg import load_graph, save_graph
    from .graph_metrics import delta_ged, delta_ig
    
    mem = Memory.load()
    g = load_graph()
    print(f"Starting memory reorganization for {iterations} iterations")
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        # 1. グラフから重要なノードを特定
        important_nodes = [i for i in range(min(10, len(mem.episodes)))]  # 簡易実装：最初の10ノード
        
        # 2. 重要なノードに対応するエピソードの重要度を上げる
        for node_id in important_nodes:
            mem.update_c([node_id], reward=0.2)  # 自己報酬
        
        # 3. 類似度の高いエピソードをマージ検討
        if len(mem.episodes) > 5:  # 十分なエピソードがある場合
            # 簡易実装：最初の2つをマージ
            mem.merge([0, 1])
        
        # 4. グラフを再構築して変化を評価
        vecs = np.vstack([e.vec for e in mem.episodes])
        new_g = build_graph(vecs)
        
        # 5. GED/IGの変化を計算
        ged_change = delta_ged(g, new_g)
        old_vecs = vecs.copy()  # 簡易比較用
        ig_change = delta_ig(old_vecs, vecs)
        
        # 6. 進捗を出力
        print(f"  ΔGED: {ged_change:.5f}, ΔIG: {ig_change:.5f}")
        
        # 7. グラフを更新
        g = new_g
        
        # 8. メモリインデックスを再訓練
        mem.train_index()
    
    # 最終結果を保存
    mem.save()
    save_graph(g)
    print("Memory reorganization complete")

if __name__ == "__main__":
    app()
