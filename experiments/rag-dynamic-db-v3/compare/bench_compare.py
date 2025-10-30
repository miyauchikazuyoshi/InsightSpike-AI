#!/usr/bin/env python3
"""
MVP: Compare four pipelines on the same corpus/queries under unified logging.

Pipelines:
- static (Plain RAG; no updates)
- cosine (heuristic graphRAG)
- frequency (heuristic graphRAG)
- gedig (proposed geDIG‑graphRAG)

Outputs:
- summary.csv (per‑method aggregate)
- records.csv (per‑query records)
- graphs/{method}.graph.json
- state/{method}.state.json
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_ROOT = REPO_ROOT / "experiments/rag-dynamic-db-v3"
# Import as package 'src.*' so that relative imports (..core) resolve
sys.path.insert(0, str(RAG_ROOT))

# Baselines and utils
from src.baselines.static_rag import StaticRAG
from src.baselines.cosine_rag import CosineOnlyRAG
from src.baselines.frequency_rag import FrequencyBasedRAG
from src.baselines.gedig_rag import GeDIGRAG
from src.core.config import ExperimentConfig, DEFAULT_CONFIG
from src.multidomain_knowledge_base import (
    create_multidomain_knowledge_base,
    create_multidomain_queries,
)
# Import utility from same folder (ensure path)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_embed_gen import build_embedder, build_generator, choose_provider


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_systems(cfg: ExperimentConfig):
    systems = {
        "static": StaticRAG(cfg),
        "cosine": CosineOnlyRAG(cfg),
        "frequency": FrequencyBasedRAG(cfg),
        "gedig": GeDIGRAG(cfg),
    }
    return systems


def add_initial_kb(systems, kb_texts: List[str], max_docs: int | None = None) -> Dict[str, int]:
    n_added = {}
    docs = kb_texts[: (max_docs or len(kb_texts))]
    for name, sys in systems.items():
        n_added[name] = sys.add_initial_knowledge(docs)
    return n_added


def run_once(
    outdir: Path,
    n_queries: int = 30,
    seed: int = 42,
    max_kb_docs: int | None = None,
    embed_mode: str = "auto",
    provider: str | None = None,
    model: str | None = None,
    gedig_threshold_mode: str | None = None,
    gedig_threshold: float | None = None,
    gedig_percentile: float | None = None,
    gedig_k: float | None = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "graphs").mkdir(parents=True, exist_ok=True)
    (outdir / "state").mkdir(parents=True, exist_ok=True)

    # Config (use defaults; can be extended via YAML later)
    cfg = DEFAULT_CONFIG
    rng = np.random.default_rng(seed)

    # Data
    kb_items = create_multidomain_knowledge_base()
    kb_texts = [k.text for k in kb_items]
    queries_all = [q for q, _lvl in create_multidomain_queries()]
    if n_queries and n_queries < len(queries_all):
        idx = rng.choice(len(queries_all), size=n_queries, replace=False)
        queries = [queries_all[i] for i in idx]
    else:
        queries = queries_all

    # Systems (build first, then inject embedder/generator)
    systems = build_systems(cfg)

    # Realistic components with graceful fallbacks
    embedder = build_embedder(kb_texts, queries, mode=embed_mode)
    generator = build_generator(provider_name=provider, model_name=model)
    for sys_name, sys in systems.items():
        sys.embedder = embedder
        sys.generator = generator
        # Optional: override geDIG threshold settings for the gedig system
        if sys_name == "gedig":
            if gedig_threshold_mode:
                sys.threshold_mode = gedig_threshold_mode
            if gedig_threshold is not None:
                sys.gedig_threshold = float(gedig_threshold)
            if gedig_percentile is not None:
                sys.threshold_percentile = float(gedig_percentile)
            if gedig_k is not None and hasattr(sys, "gedig_evaluator"):
                sys.gedig_evaluator.k = float(gedig_k)

    add_initial_kb(systems, kb_texts, max_docs=max_kb_docs)

    # Records
    per_query_rows: List[Dict] = []

    for qi, q in enumerate(queries, 1):
        for name, sys in systems.items():
            resp = sys.process_query(q, query_id=str(qi), session_id=0)
            row = {
                "method": name,
                "query_id": resp.query_id,
                "query": resp.query,
                "total_time": f"{resp.total_time:.4f}",
                "retrieval_time": f"{resp.retrieval_time:.4f}",
                "generation_time": f"{resp.generation_time:.4f}",
                "update_time": f"{resp.update_time:.4f}",
                "updated": int(bool(resp.knowledge_updated)),
                "graph_size_before": resp.graph_size_before,
                "graph_size_after": resp.graph_size_after,
                "update_reason": resp.update_decision.reason if resp.update_decision else "",
            }
            # geDIG extras if available
            if name == "gedig" and resp.update_decision and resp.update_decision.gedig_result:
                r = resp.update_decision.gedig_result
                row.update({
                    "gedig": f"{r.delta_gedig:.6f}",
                    "delta_ged": f"{r.delta_ged:.6f}",
                    "delta_ig": f"{r.delta_ig:.6f}",
                })
            per_query_rows.append(row)

    # Summaries
    summary_rows: List[Dict] = []
    for name, sys in systems.items():
        stats = sys.get_statistics()
        gstats = stats.get("graph_statistics", {})
        # Fallbacks for key names depending on KG implementation
        n_nodes = gstats.get("n_nodes", gstats.get("current_nodes", 0))
        n_edges = gstats.get("n_edges", gstats.get("current_edges", 0))
        # Compute isolated nodes if not present
        n_isolated = gstats.get("n_isolated", 0)
        if not n_isolated and hasattr(sys, "knowledge_graph"):
            try:
                deg = dict(sys.knowledge_graph.graph.degree())
                n_isolated = sum(1 for _, d in deg.items() if d == 0)
            except Exception:
                n_isolated = 0
        summary_rows.append({
            "method": name,
            "queries": stats.get("queries_processed", 0),
            "updates": stats.get("updates_applied", 0),
            "update_rate": f"{stats.get('update_rate', 0.0):.4f}",
            "avg_total_time": f"{stats.get('avg_response_time', 0.0):.4f}",
            "nodes": n_nodes,
            "edges": n_edges,
            "isolated": n_isolated,
        })

        # Save graphs (state saving may include non-JSON-native types; skip for MVP)
        sys.knowledge_graph.save_to_file(outdir / "graphs" / f"{name}.graph.json")
        # sys.save_state(str(outdir / "state" / f"{name}.state.json"))

    # Write CSVs
    # Union of keys to accommodate optional geDIG columns
    fieldnames = []
    if per_query_rows:
        keys = set()
        for r in per_query_rows:
            keys.update(r.keys())
        fieldnames = list(sorted(keys))
    with open(outdir / "records.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_query_rows)
    with open(outdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(REPO_ROOT / "experiments/rag-dynamic-db-v3/results/compare_mvp"))
    ap.add_argument("--n-queries", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-kb-docs", type=int, default=None)
    ap.add_argument("--embed-mode", type=str, default="auto", choices=["auto","st","insightspike","is","tfidf","fallback"], help="Embedding backend preference")
    ap.add_argument("--provider", type=str, default=None, choices=["anthropic","openai","mock"], help="Force provider (default: auto-detect)")
    ap.add_argument("--model", type=str, default=None, help="Override model name for provider (e.g., claude-3-5-sonnet-20240620)")
    # geDIG threshold overrides
    ap.add_argument("--gedig-threshold-mode", type=str, default=None, choices=["fixed","percentile"], help="geDIG decision mode: fixed threshold or percentile")
    ap.add_argument("--gedig-threshold", type=float, default=None, help="Fixed threshold θ for F=ΔGED-kΔIG (accept if F<θ)")
    ap.add_argument("--gedig-percentile", type=float, default=None, help="Percentile for threshold when mode=percentile (e.g., 20 means accept lowest 20% F)")
    ap.add_argument("--gedig-k", type=float, default=None, help="k coefficient in F=ΔGED-k·ΔIG")
    args = ap.parse_args()

    base_out = Path(args.outdir)
    run_dir = base_out / f"run_{_ts()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] output: {run_dir}")
    summary = run_once(
        run_dir,
        n_queries=args.n_queries,
        seed=args.seed,
        max_kb_docs=args.max_kb_docs,
        embed_mode=args.embed_mode,
        provider=args.provider,
        model=args.model,
        gedig_threshold_mode=args.gedig_threshold_mode,
        gedig_threshold=args.gedig_threshold,
        gedig_percentile=args.gedig_percentile,
        gedig_k=args.gedig_k,
    )
    print("[compare] summary:")
    for row in summary:
        print("  ", row)


if __name__ == "__main__":
    main()
