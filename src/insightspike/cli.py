"""CLI entrypoints"""
from pathlib import Path
import typer
import numpy as np
from rich import print
import pathlib
from typing import Optional

from .embedder              import get_model
from .loader                import load_corpus
from .layer2_memory_manager import Memory
# try-except で両方のインポート方法を試す
try:
    from insightspike import build_graph, load_graph, save_graph  # パッケージから直接インポート
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
def embed(path: Optional[pathlib.Path] = typer.Option(None, help="テキストファイルのパス")):
    """Embeddingを実行"""
    if path is None:
        print("[red]Error: --path オプションで入力ファイルを指定してください[/red]")
        raise typer.Exit(code=1)
    print(f"Embedding for: {path}")
    docs = load_corpus(path)
    Memory.build(docs).save()
    print(f"Embedded {len(docs)} docs.")

@app.command()
def basegraph(
    input_path: str = typer.Argument(..., help="入力npyファイルパス"),
    output_path: str = typer.Argument(..., help="出力ptファイルパス"),
    sim_threshold: float = typer.Argument(0.8, help="類似度しきい値")
):
    """全組み合わせで初期グラフを構築し保存"""
    import numpy as np
    from insightspike import build_base_graph
    import torch

    vectors = np.load(input_path)
    data, _ = build_base_graph(vectors, sim_threshold)
    torch.save(data, output_path)
    print(f"Base graph saved to {output_path}")

@app.command()
def graph():
    mem = Memory.load(); vecs = np.vstack([e.vec for e in mem.episodes])
    build_graph(vecs)
    print("Graph initialised.")

@app.command()
def loop(q: str, k: int = 15):
    """質問に対して推論ループを実行"""
    import torch
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
        if len(mem.episodes) > 5:
            mem.merge([0, 1])

        # 4. layer2: C値が低い/非アクティブなエピソードをprune
        mem.prune(c_thresh=0.1, inactive_n=10)

        # 5. layer2: C値が高すぎるノードをsplit（例: c>0.95）
        for idx, ep in enumerate(mem.episodes):
            if ep.c > 0.95:
                mem.split(idx)

        # 6. layer2: 新規エピソード追加例（ダミー）
        #if i == iterations - 1:
            # 最終イテレーションでダミー追加
        #    dummy_vec = np.random.randn(mem.dim).astype(np.float32)
        #    mem.add_episode(dummy_vec, f"Dummy episode {i}", c_init=0.2)

        # 7. グラフを再構築して変化を評価
        vecs = np.vstack([e.vec for e in mem.episodes])
        new_g = build_graph(vecs)

        # 8. GED/IGの変化を計算
        ged_change = delta_ged(g, new_g)
        old_vecs = vecs.copy()
        ig_change = delta_ig(old_vecs, vecs)

        # 9. 進捗を出力
        print(f"  ΔGED: {ged_change:.5f}, ΔIG: {ig_change:.5f}")

        # 10. グラフを更新
        g = new_g

        # 11. メモリインデックスを再訓練
        mem.train_index()

    # 最終結果を保存
    mem.save()
    save_graph(g)
    print("Memory reorganization complete")

@app.command()
def ask_LLM(q: str):
    """質問を直接Layer4 LLMに投げて回答を得る"""
    from .layer4_llm import generate
    answer = generate(q)
    print("[bold magenta]Answer:[/bold magenta]", answer)

@app.command()
def adaptive(q: str, initial_k: int = 5, max_k: int = 50, step_k: int = 5):
    from .layer2_memory_manager import Memory
    mem = Memory.load()
    from .agent_loop import adaptive_loop
    g_new, iteration_count = adaptive_loop(mem, q, initial_k=initial_k, max_k=max_k, step_k=step_k)
    print(f"[green]Adaptive loop finished in {iteration_count} iterations.[/green]")

@app.command()
def add_data_baked(
    baked_path: str = typer.Argument(..., help="追加するembedding npyファイルのパス")
):
    """既存Memoryやグラフにembedding済みデータを追加"""
    import numpy as np
    from insightspike.layer2_memory_manager import Memory

    # 既存Memoryをロード
    mem = Memory.load()
    # 追加embeddingをロード
    new_vecs = np.load(baked_path)
    # 追加データをMemoryにappend（実装によってはadd_episode等を使う）
    for vec in new_vecs:
        mem.add_episode(vec, text="追加データ", c_init=0.2)
    mem.save()
    print(f"追加embedding({len(new_vecs)})件をMemoryに統合しました。")


if __name__ == "__main__":
    app()
