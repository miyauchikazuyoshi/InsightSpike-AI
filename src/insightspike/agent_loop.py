"""Subcortical 4-loop with EurekaSpike"""
from __future__ import annotations
from rich import print
import numpy as np, networkx as nx

from .embedder              import get_model
from .prompt_builder        import build_prompt
from .layer1_error_monitor  import uncertainty
from .layer2_memory_manager import Memory
from .layer3_graph_pyg      import build_graph
from .graph_metrics         import delta_ged, delta_ig
from .layer4_llm            import generate
from .config                import (TOP_K, SPIKE_GED, SPIKE_IG, ETA_SPIKE, LOG_DIR, MERGE_GED, SPLIT_IG, PRUNE_C, INACTIVE_N, timestamp,)
__all__ = ["cycle"]

TOP_K = 15  # 検索結果として取得するドキュメント数

def cycle(memory: Memory, question: str, g_old: nx.Graph | None = None):
    """Run a single reasoning cycle.

    Parameters
    ----------
    memory : Memory
        Episode memory store.
    question : str
        User question text.
    g_old : nx.Graph | None, optional
        Previous similarity graph for ΔGED calculation.
    """

    time_id = timestamp()
    # --- save question text ---
    (LOG_DIR / f"{time_id}_question.txt").write_text(question, encoding="utf-8")

    model = get_model(); q_vec = model.encode([question], normalize_embeddings=True)[0]
    raw_scores, ids = memory.search(q_vec, TOP_K)
    scores = list(raw_scores)

    vecs_new = np.vstack([memory.episodes[i].vec for i in ids]) if memory.episodes[0].vec.any() else None
    g_pyg, _ = build_graph(vecs_new)
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
