"""Subcortical 4-loop with EurekaSpike"""
from __future__ import annotations
from rich import print
import numpy as np, networkx as nx


# Provide a module-level torch object so tests can patch
try:  # pragma: no cover - optional dependency
    import torch as torch  # attempt real torch
except Exception:  # torch not installed
    class _DummyTorch:
        """Fallback torch stub providing load/save stubs."""

        def load(self, *_, **__):
            raise ModuleNotFoundError("torch is required")

        def save(self, *_, **__):
            raise ModuleNotFoundError("torch is required")

    torch = _DummyTorch()

from typer import Typer, Option

from .embedder              import get_model
from .prompt_builder        import build_prompt
from .layer1_error_monitor  import uncertainty
from .layer2_memory_manager import Memory
from .layer3_graph_pyg      import build_graph
from .graph_metrics         import delta_ged, delta_ig
from .layer4_llm            import generate
from .config                import (TOP_K, SPIKE_GED, SPIKE_IG, ETA_SPIKE, LOG_DIR, MERGE_GED, SPLIT_IG, PRUNE_C, INACTIVE_N, timestamp,)
__all__ = ["cycle"]

app = Typer()

def cycle(memory: Memory, question: str, g_old: nx.Graph | None = None, top_k=TOP_K):
    """Run a single reasoning cycle."""

    time_id = timestamp()
    # --- save question text ---
    (LOG_DIR / f"{time_id}_question.txt").write_text(question, encoding="utf-8")

    model = get_model(); q_vec = model.encode([question], normalize_embeddings=True)[0]
    raw_scores, ids = memory.search(q_vec, top_k)
    scores = list(raw_scores)

    # 変更点: 空のメモリや結果を安全に処理
    if not ids or not memory.episodes:
        # 空の結果を返す場合のハンドリング
        return nx.Graph()

    # 変更点: 安全にアクセスするよう修正
    try:
        vecs_new = np.vstack([memory.episodes[i].vec for i in ids])
        g_pyg, _ = build_graph(vecs_new)
        
        # docs変数の定義を追加 - これが欠けていた
        docs = [memory.episodes[i].text for i in ids]
        
    except (IndexError, AttributeError):
        # エラー時は空のグラフを返す
        return nx.Graph()

    g_new = nx.Graph(); g_new.add_nodes_from(range(g_pyg.num_nodes))
    g_new.add_edges_from(g_pyg.edge_index.numpy().T.tolist())

    d_ged = delta_ged(g_old, g_new) if g_old is not None else 0.0
    d_ig  = delta_ig(vecs_new, vecs_new) if vecs_new is not None else 0.0
    unc   = uncertainty(scores)
    reward = (-d_ged) + d_ig - unc

    eureka = (d_ged < -SPIKE_GED) and (d_ig > SPIKE_IG)
    if eureka:
        print("⚡ [yellow]EurekaSpike![/yellow]")
        memory.update_c(list(ids), reward, eta=ETA_SPIKE)
        memory.train_index()
    else:
        memory.update_c(list(ids), reward)

    # 正しく定義されたdocs変数を使用
    answer = generate(build_prompt(question, docs))

    print(f"ΔGED {-d_ged:.3f}  ΔIG {d_ig:.3f}  Unc {unc:.3f}  R {reward:.3f}")
    print("[bold magenta]Answer:[/bold magenta]", answer, "")

    # --- Merge / Split triggers ---------------------------------
    if d_ged < -MERGE_GED:
        memory.merge(list(ids))
    elif d_ig < SPLIT_IG:
        memory.split(ids[0])          # とりあえず先頭を分裂
    memory.prune(PRUNE_C, INACTIVE_N)

    # --- add new episode from LLM hypothesis --------------------
    answer_vec = model.encode([answer], normalize_embeddings=True)[0]
    memory.add_episode(answer_vec, answer, c_init=0.2)

    # --- save memory snapshot ---
    meta_path = memory.save()  # returns Path
    snapshot_name = LOG_DIR / f"{time_id}_index.json"
    meta_path.replace(snapshot_name)

    # --- output & return ---
    print(f"Snapshot saved → {snapshot_name.name}\n")
    return g_new

def cycle_with_status(memory: Memory, question: str, g_old: nx.Graph | None = None, top_k=TOP_K):
    """cycle関数を拡張して報酬と内発報酬の状態も返す"""
    time_id = timestamp()
    # --- save question text ---
    (LOG_DIR / f"{time_id}_question.txt").write_text(question, encoding="utf-8")
    
    model = get_model(); q_vec = model.encode([question], normalize_embeddings=True)[0]
    raw_scores, ids = memory.search(q_vec, top_k)
    scores = list(raw_scores)
    
    # 空のメモリや結果を安全に処理
    if not ids or not memory.episodes:
        # 空の結果を返す場合
        return nx.Graph(), 0.0, False
    
    # 安全にアクセス
    try:
        vecs_new = np.vstack([memory.episodes[i].vec for i in ids])
        g_pyg, _ = build_graph(vecs_new)
        docs = [memory.episodes[i].text for i in ids]
    except (IndexError, AttributeError):
        # エラー時
        return nx.Graph(), 0.0, False
    
    g_new = nx.Graph(); g_new.add_nodes_from(range(g_pyg.num_nodes))
    g_new.add_edges_from(g_pyg.edge_index.numpy().T.tolist())
    
    d_ged = delta_ged(g_old, g_new) if g_old is not None else 0.0
    d_ig  = delta_ig(vecs_new, vecs_new) if vecs_new is not None else 0.0
    unc   = uncertainty(scores)
    reward = (-d_ged) + d_ig - unc
    
    # 内発報酬の判定
    eureka = (d_ged < -SPIKE_GED) and (d_ig > SPIKE_IG)
    
    # 残りの処理も同様に実装
    if eureka:
        print("⚡ [yellow]EurekaSpike![/yellow]")
        memory.update_c(list(ids), reward, eta=ETA_SPIKE)
        memory.train_index()
    else:
        memory.update_c(list(ids), reward)
        
    answer = generate(build_prompt(question, docs))

    print(f"ΔGED {-d_ged:.3f}  ΔIG {d_ig:.3f}  Unc {unc:.3f}  R {reward:.3f}")
    print("[bold magenta]Answer:[/bold magenta]", answer, "")

    if d_ged < -MERGE_GED:
        memory.merge(list(ids))
    elif d_ig < SPLIT_IG:
        memory.split(ids[0])
    memory.prune(PRUNE_C, INACTIVE_N)

    answer_vec = model.encode([answer], normalize_embeddings=True)[0]
    memory.add_episode(answer_vec, answer, c_init=0.2)

    meta_path = memory.save()
    snapshot_name = LOG_DIR / f"{time_id}_index.json"
    meta_path.replace(snapshot_name)

    print(f"Snapshot saved → {snapshot_name.name}\n")
    
    return g_new, reward, eureka  # 3つの値を返す

@app.command()
def adaptive_loop(memory, question, initial_k=5, max_k=50, step_k=5):

    """Increase search range until an intrinsic reward is triggered."""

    try:
        g_old = torch.load("data/graph_loop.pt")
    except (FileNotFoundError, RuntimeError):
        g_old = None  # 他の例外は捕捉されない
    
    current_k = initial_k
    max_iterations = (max_k - initial_k) // step_k + 1
    
    iteration_count = 0
    for i in range(max_iterations):
        print(f"試行 {i+1}/{max_iterations}: k={current_k}")
        g_new, reward, eureka = cycle_with_status(memory, question, g_old, current_k)
        
        iteration_count += 1  #

        if eureka:
            print("⚡ EurekaSpike検出! 探索終了")
            break
            
        current_k += step_k
        if current_k > max_k:
            print("最大検索範囲に到達")
            break
    
    torch.save(g_new, "data/graph_loop.pt")
    return g_new, iteration_count  # 戻り値を追加
