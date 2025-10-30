#!/usr/bin/env python3
"""
RAG PSZ Trial Runner: Generate per‑query metrics needed for PSZ scatter and latency table.

Outputs:
  - data/rag_eval/psz_points.csv      (run_id, query_id, H, k, PER, acceptance, FMR, latency_ms)
  - data/latency/rag_latency.csv      (run_id, H, k, latency_ms)

Notes:
  - PER is approximated by tokenized length ratio (prompt_with_docs / query_only).
  - acceptance/FMR are rule‑based placeholders by default (until human/rule evaluation is wired):
      acceptance = 1 if answer shares ≥1 keyword with retrieved docs or length≥32 chars; else 0
      FMR        = 0 (placeholder)
    You can later replace these with your evaluation rules or human annotations.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Repo paths
REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_SRC = REPO_ROOT / "experiments/rag-dynamic-db-v3/src"
import sys
sys.path.insert(0, str(RAG_SRC))
sys.path.insert(0, str(REPO_ROOT / "src"))

from multidomain_knowledge_base import (
    create_multidomain_knowledge_base,
    create_multidomain_queries,
)
from insightspike.providers.provider_factory import ProviderFactory
from insightspike.implementations.layers.layer4_prompt_builder import L4PromptBuilder


def l2norm(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-9)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def build_embedder_auto(kb_texts: List[str], queries: List[str]):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return lambda texts: l2norm(model.encode(texts))
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        vec.fit(kb_texts + queries)
        return lambda texts: l2norm(vec.transform(texts).astype(np.float64).toarray())


def topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sim.shape[0])
    return np.argpartition(-sim, k - 1)[:k]


def jaccard_keywords(a: str, b: str, min_len: int = 4) -> float:
    import re
    tok = lambda s: {w for w in re.findall(r"\w+", s.lower()) if len(w) >= min_len}
    A, B = tok(a), tok(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def rule_acceptance(answer: str, docs: List[str]) -> int:
    if not answer or len(answer) < 16:
        return 0
    if len(answer) >= 32:
        return 1
    # keyword overlap with any doc
    for t in docs:
        if jaccard_keywords(answer, t) >= 0.05:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--provider", type=str, default="anthropic", choices=["anthropic","openai","mock"])
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--out_psz", type=str, default=str(REPO_ROOT / "data/rag_eval/psz_points.csv"))
    ap.add_argument("--out_lat", type=str, default=str(REPO_ROOT / "data/latency/rag_latency.csv"))
    ap.add_argument("--max_queries", type=int, default=0, help="Limit number of queries (0=all)")
    ap.add_argument("--log_prompt", action="store_true")
    args = ap.parse_args()

    # Load .env if present (and normalize Anthropic key names)
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line or line.strip().startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    if k and (k not in os.environ):
                        os.environ[k] = v
        except Exception:
            pass
    # Map alternative Anthropic key env vars if necessary
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        for alt in ("CLAUDE_API_KEY", "ANTH_API_KEY"):
            if os.getenv(alt):
                os.environ["ANTHROPIC_API_KEY"] = os.environ[alt]
                break

    # Load KB and queries
    kb_items = create_multidomain_knowledge_base()
    kb_texts = [k.text for k in kb_items]
    queries = [q for q, _lvl in create_multidomain_queries()]
    if args.max_queries and args.max_queries > 0:
        queries = queries[: args.max_queries]

    embed = build_embedder_auto(kb_texts, queries)
    E = embed(kb_texts)

    builder = L4PromptBuilder()
    api_key = None
    if args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or os.getenv("ANTH_API_KEY")
    elif args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    prov_cfg = dict(provider=args.provider)
    if api_key:
        prov_cfg["api_key"] = api_key
    provider = ProviderFactory.create(args.provider, config=prov_cfg)

    psz_rows = []
    lat_rows = []
    run_id = f"pszH{args.H}_k{args.k}_{args.provider}"

    for q_idx, q in enumerate(queries):
        qv = embed([q])[0]
        sims = E @ qv
        idx = topk_indices(sims, args.k)
        docs = [kb_texts[i] for i in idx]

        # Build prompt and measure PER (chars as proxy for tokens)
        context = {"retrieved_documents": [{"text": kb_texts[i], "similarity": float(sims[i])} for i in idx]}
        prompt = builder.build_prompt(context, q)
        per = (len(prompt.split()) / max(1, len(q.split()))) * 100.0

        # Generate answer and measure additional latency
        t0 = time.time()
        ans = provider.generate(prompt, model=args.model) if hasattr(provider, "generate") else ""
        latency_ms = (time.time() - t0) * 1000.0

        acc = rule_acceptance(ans, docs)
        fmr = 0.0  # placeholder until human/rule evaluation is wired

        psz_rows.append({
            "run_id": run_id,
            "query_id": q_idx,
            "H": args.H,
            "k": args.k,
            "PER": per,
            "acceptance": acc,
            "FMR": fmr,
            "latency_ms": latency_ms,
        })
        lat_rows.append({"run_id": run_id, "H": args.H, "k": args.k, "latency_ms": latency_ms})

    # Write CSVs
    out_psz = Path(args.out_psz)
    out_psz.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_psz.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "query_id", "H", "k", "PER", "acceptance", "FMR", "latency_ms"])
        for r in psz_rows:
            w.writerow([r["run_id"], r["query_id"], r["H"], r["k"], f"{r['PER']:.2f}", r["acceptance"], f"{r['FMR']:.3f}", f"{r['latency_ms']:.2f}"])

    out_lat = Path(args.out_lat)
    out_lat.parent.mkdir(parents=True, exist_ok=True)
    with out_lat.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "H", "k", "latency_ms"])
        for r in lat_rows:
            w.writerow([r["run_id"], r["H"], r["k"], f"{r['latency_ms']:.2f}"])

    print(f"✅ Wrote {out_psz}\n✅ Wrote {out_lat}")


if __name__ == "__main__":
    raise SystemExit(main())
