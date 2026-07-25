#!/usr/bin/env python3
"""v12: BRIGHT Benchmark — Graph-based Re-ranking Experiments.

Modes:
  bm25_only    : BM25 baseline (no graph re-ranking)
  graph_rerank : BM25 + entity graph re-ranking
  cot_rerank   : BM25 + CoT-augmented entity graph re-ranking

Runs evaluation on one or more BRIGHT domains and computes nDCG@10,
Recall@10, and MRR metrics.

Usage:
  PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
      experiments/hotpotqa_v2/scripts/run_bright.py \
      --mode graph_rerank \
      --domains biology,economics \
      --data-dir experiments/hotpotqa_v2/data/bright/ \
      --output experiments/hotpotqa_v2/results/v12_bright_rerank \
      --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bright_pipeline import (
    BrightPipeline,
    BrightResult,
    build_bm25_index,
    bm25_tokenize,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_mrr,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v12: BRIGHT Graph Re-ranking Experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["bm25_only", "graph_rerank", "cot_rerank", "cot_retrieval", "adaptive_retrieval", "unified", "gedig_routing"],
        required=True,
        help="bm25_only: BM25 baseline; graph_rerank: BM25 + entity graph; "
             "cot_rerank: BM25 + CoT-augmented entity graph",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="biology,economics,stackoverflow",
        help="Comma-separated list of BRIGHT domains",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="experiments/hotpotqa_v2/data/bright/",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max queries per domain")

    # Pipeline parameters
    parser.add_argument("--bm25-engine", type=str, default="rank_bm25",
                        choices=["rank_bm25", "pyserini"],
                        help="BM25 engine: rank_bm25 (Python) or pyserini (Lucene)")
    parser.add_argument("--pyserini-k1", type=float, default=0.9,
                        help="Pyserini BM25 k1 parameter (default: 0.9, BRIGHT paper)")
    parser.add_argument("--pyserini-b", type=float, default=0.4,
                        help="Pyserini BM25 b parameter (default: 0.4, BRIGHT paper)")
    parser.add_argument("--initial-top-k", type=int, default=100,
                        help="BM25 candidates to retrieve")
    parser.add_argument("--graph-top-k", type=int, default=30,
                        help="Top BM25 docs used for graph construction")
    parser.add_argument("--rerank-top-k", type=int, default=10,
                        help="Final output size")
    parser.add_argument("--rerank-alpha", type=float, default=0.5,
                        help="BM25 weight (0=all graph, 1=all BM25)")
    parser.add_argument("--max-para-freq", type=int, default=5,
                        help="Discriminative entity filter threshold")

    # CoT-specific parameters
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model for CoT generation")
    parser.add_argument("--cot-weight", type=float, default=2.0,
                        help="Weight multiplier for CoT-connected nodes")
    # CoT Re-retrieval parameters
    parser.add_argument("--cot-retrieval-top-k", type=int, default=50,
                        help="Number of new docs to retrieve via CoT query")
    parser.add_argument("--cot-retrieval-max-concepts", type=int, default=20,
                        help="Max CoT concepts to use as BM25 query")
    # Adaptive retrieval parameters
    parser.add_argument("--beta-low", type=int, default=3,
                        help="β₀ threshold for Tier 1 (skip CoT)")
    parser.add_argument("--beta-high", type=int, default=10,
                        help="β₀ threshold for Tier 3 (aggressive)")
    parser.add_argument("--aggressive-top-k", type=int, default=100,
                        help="Tier 3: docs to retrieve via CoT query")
    parser.add_argument("--aggressive-max-concepts", type=int, default=40,
                        help="Tier 3: max concepts for BM25 query")
    # Unified pipeline parameters
    parser.add_argument("--dense-index-dir", type=str, default=None,
                        help="Dense index directory (required for unified mode)")
    parser.add_argument("--dense-top-k", type=int, default=100,
                        help="Phase 1b Dense retrieval count")
    parser.add_argument("--dense-cot-top-k", type=int, default=50,
                        help="Phase 2.5b Dense CoT retrieval count")
    parser.add_argument("--dense-sim-threshold", type=float, default=0.5,
                        help="Tier D graph edge threshold")
    parser.add_argument("--llm-rerank", action="store_true",
                        help="Enable LLM listwise reranking")
    parser.add_argument("--llm-rerank-top-k", type=int, default=20,
                        help="LLM rerank candidate pool size")
    # Graph mode (Spec F: D+E integration)
    parser.add_argument("--graph-mode", choices=["sentence", "episode", "hybrid"],
                        default="sentence",
                        help="Graph node text source: sentence (split), episode (LLM decomposed), or hybrid (sentence + episode)")
    # geDIG routing parameters (Spec E)
    parser.add_argument("--episode-index-dir", type=str, default=None,
                        help="Episode index directory (for gedig_routing or --graph-mode episode)")
    parser.add_argument("--gedig-lambda", type=float, default=1.0,
                        help="geDIG lambda weight")
    parser.add_argument("--gedig-max-hops", type=int, default=2,
                        help="geDIG multi-hop depth")
    parser.add_argument("--gedig-sp-beta", type=float, default=0.2,
                        help="geDIG shortest-path weight")
    parser.add_argument("--gedig-tau-dg", type=float, default=-0.3,
                        help="geDIG DG threshold")
    parser.add_argument("--gedig-tau-ag", type=float, default=0.1,
                        help="geDIG AG threshold")
    parser.add_argument("--gedig-k-target", type=int, default=4,
                        help="Query episode cross-edge target count")
    # geDIG scoring parameters (Spec H)
    parser.add_argument("--scoring-mode", choices=["classic", "gedig", "gedig_refine"],
                        default="classic",
                        help="Scoring method: classic (5-component), gedig (geDIG-based), "
                             "or gedig_refine (geDIG graph refinement + classic scoring)")
    parser.add_argument("--gedig-scoring-lambda", type=float, default=1.0,
                        help="geDIG scoring lambda weight (GED vs IG balance)")
    parser.add_argument("--gedig-scoring-sp-beta", type=float, default=0.5,
                        help="geDIG scoring shortest-path component weight")
    parser.add_argument("--gedig-scoring-k-hop", type=int, default=2,
                        help="Local subgraph k-hop radius for per-document geDIG")
    parser.add_argument("--gedig-scoring-mp-iterations", type=int, default=2,
                        help="Message passing iterations for geDIG scoring")
    parser.add_argument("--gedig-scoring-mp-alpha", type=float, default=0.3,
                        help="Query influence weight in message passing")
    # Pointwise LLM reranking parameters (Spec J)
    parser.add_argument("--pointwise-rerank", action="store_true",
                        help="Enable LLM pointwise reasoning reranking (Spec J)")
    parser.add_argument("--pointwise-rerank-top-k", type=int, default=30,
                        help="Number of candidates for pointwise scoring")
    parser.add_argument("--pointwise-batch-size", type=int, default=5,
                        help="Documents per LLM call for pointwise scoring")
    parser.add_argument("--pointwise-blend-weight", type=float, default=0.4,
                        help="Pointwise score weight in blend (0=ignore PW, 1=full PW)")
    # Query decomposition parameters (Spec K)
    parser.add_argument("--query-decomp", action="store_true",
                        help="Enable query decomposition (Spec K)")
    parser.add_argument("--query-decomp-top-k", type=int, default=50,
                        help="BM25 top-k per sub-query")
    parser.add_argument("--query-decomp-max-sub", type=int, default=5,
                        help="Max sub-questions to generate")
    # Reasoning reranking parameters (Spec L)
    parser.add_argument("--reasoning-rerank", action="store_true",
                        help="Enable reasoning LLM reranking (Spec L)")
    parser.add_argument("--rerank-model", type=str, default="",
                        help="Model for reasoning reranking (default: same as --model)")
    parser.add_argument("--reasoning-rerank-top-k", type=int, default=20,
                        help="Number of candidates for reasoning reranking")
    parser.add_argument("--reasoning-rerank-doc-chars", type=int, default=4000,
                        help="Max document chars for reasoning reranking prompt")
    parser.add_argument("--reasoning-rerank-blend-weight", type=float, default=0.7,
                        help="Reasoning score weight in blend (0=ignore, 1=full replace)")
    # RIA iterative expansion parameters (Spec M)
    parser.add_argument("--ria-loop", action="store_true",
                        help="Enable RIA iterative query expansion (Spec M)")
    parser.add_argument("--ria-max-rounds", type=int, default=3,
                        help="Maximum RIA iteration rounds")
    parser.add_argument("--ria-docs-per-round", type=int, default=50,
                        help="New docs to retrieve per RIA round")
    parser.add_argument("--ria-feedback-top-k", type=int, default=5,
                        help="Top-k docs to feed back to LLM per round")
    parser.add_argument("--ria-beta0-target", type=int, default=1,
                        help="Target beta0 for RIA convergence")

    # Token-level graph parameters (Spec N)
    parser.add_argument("--token-graph", action="store_true",
                        help="Enable per-doc token-level graph scoring (Spec N)")
    parser.add_argument("--token-graph-weight", type=float, default=0.15,
                        help="Token graph weight in scoring blend")
    parser.add_argument("--token-graph-max-tokens", type=int, default=500,
                        help="Max tokens per doc for spaCy parsing")
    parser.add_argument("--token-graph-walk-score", action="store_true",
                        help="Use DG/AG weighted shortest paths (geDIG Walk Score)")
    parser.add_argument("--token-graph-dg-penalty", type=float, default=2.0,
                        help="Cost penalty for bridge (DG) edges in walk score")
    parser.add_argument("--token-graph-f-eval", action="store_true",
                        help="Use F-evaluation based DG/AG classification (Spec N.2)")
    parser.add_argument("--token-graph-f-lambda", type=float, default=1.0,
                        help="Lambda for F-evaluation: f = cost - lambda * relevance")
    parser.add_argument("--token-graph-insight", type=str, default="none",
                        choices=["none", "graph_agg", "path_bridge", "both"],
                        help="Insight vector injection mode (Spec N.2)")
    # Early Token Graph for DG-guided CoT/RIA (Spec W)
    parser.add_argument("--early-token-graph", action="store_true",
                        help="Run lightweight Token Graph before CoT to detect DG gaps (Spec W)")
    parser.add_argument("--early-tg-top-k", type=int, default=20,
                        help="Number of BM25 docs for early Token Graph (default 20)")
    # Enhanced Graph Construction (Spec X)
    parser.add_argument("--enhanced-graph", action="store_true",
                        help="spaCy sentencizer + NER entities + lemma matching (Spec X)")
    # Progressive DG Escalation (Spec Y)
    parser.add_argument("--progressive-routing", action="store_true",
                        help="3-tier AG/DG escalation: AG→DG-Shallow→DG-Deep (Spec Y)")
    parser.add_argument("--progressive-ag-threshold", type=float, default=0.70,
                        help="AG score above this → Tier 1 AG, skip CoT/RIA (default 0.70)")
    parser.add_argument("--progressive-dg-shallow-threshold", type=float, default=0.55,
                        help="AG score above this → Tier 2 DG-Shallow, gap-fill only (default 0.55)")
    # Unified Heterogeneous Graph Transformer (Spec Z)
    parser.add_argument("--unified-graph", action="store_true",
                        help="Enable AGHT: unified Sentence-Token graph with QKV attention (Spec Z)")
    parser.add_argument("--aght-lambda", type=float, default=1.0,
                        help="F-eval lambda for AGHT edge evaluation (default 1.0)")
    parser.add_argument("--aght-mp-iterations", type=int, default=2,
                        help="Message passing iterations for AGHT (default 2)")
    parser.add_argument("--aght-mp-alpha", type=float, default=0.3,
                        help="Message passing interpolation weight for AGHT (default 0.3)")
    parser.add_argument("--aght-w-q1", type=float, default=1.0,
                        help="Q direct match weight for AGHT (default 1.0)")
    parser.add_argument("--aght-f-lambda", type=float, default=1.0,
                        help="F-eval lambda for AGHT QKV attention (default 1.0)")
    parser.add_argument("--aght-use-unified", action="store_true",
                        help="Deprecated no-op: AGHT already always uses src/gedig/")
    # Entity graph F-eval parameters (Spec O)
    parser.add_argument("--entity-feval", action="store_true",
                        help="Enable entity graph F-eval cross-doc walk score (Spec O)")
    parser.add_argument("--entity-feval-weight", type=float, default=0.20,
                        help="Blend weight for entity F-eval scores (default 0.20)")
    parser.add_argument("--entity-feval-lambda", type=float, default=1.0,
                        help="Lambda for entity F-eval: f = cost - lambda * relevance")
    parser.add_argument("--entity-feval-version", choices=["v1", "v2"], default="v1",
                        help="v1: percentile threshold (Spec O), v2: structural AG/DG/Bridge with Δβ₁ (Spec Q)")
    parser.add_argument("--ag-threshold", type=float, default=None,
                        help="Fixed AG similarity threshold (default: 0.2). None=use 0.2")
    parser.add_argument("--ag-max-k", type=int, default=None,
                        help="Max query edges (default: None=unlimited)")
    parser.add_argument("--ag-min-k", type=int, default=5,
                        help="Min guaranteed query edges (default: 5)")
    parser.add_argument("--beta1-weight", type=float, default=0.3,
                        help="Weight for Δβ₁ bonus in scoring (default: 0.3)")
    # geDIG CoT Loop (Spec R)
    parser.add_argument("--gedig-loop", action="store_true",
                        help="Enable geDIG CoT Loop (Spec R)")
    parser.add_argument("--gedig-loop-max-rounds", type=int, default=2,
                        help="Max rounds for geDIG CoT Loop")
    parser.add_argument("--gedig-loop-delta-beta1-target", type=int, default=1,
                        help="Δβ₁ target for convergence")
    parser.add_argument("--gedig-loop-directed", action="store_true", default=True,
                        help="Use Token Graph DG gaps for directed CoT (Spec S)")
    parser.add_argument("--gedig-loop-no-directed", action="store_false", dest="gedig_loop_directed",
                        help="Disable directed mode (use blind CoT)")
    parser.add_argument("--gedig-loop-max-new-docs", type=int, default=15,
                        help="Max new docs per loop round")
    # Multi-CoT Ensemble parameters (Spec P)
    parser.add_argument("--n-cot-ensemble", type=int, default=1,
                        help="Number of CoT chains for ensemble scoring (1=no ensemble)")
    parser.add_argument("--cot-cache-dir", type=str, default=None,
                        help="Directory to cache CoT generations for reproducibility")
    parser.add_argument("--cot-temperature", type=float, default=0.7,
                        help="Temperature for ensemble CoT generation (diversity)")

    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "mode": args.mode,
        "domains": domains,
        "initial_top_k": args.initial_top_k,
        "graph_top_k": args.graph_top_k,
        "rerank_top_k": args.rerank_top_k,
        "rerank_alpha": args.rerank_alpha,
        "max_para_freq": args.max_para_freq,
        "limit": args.limit,
    }
    config["graph_mode"] = args.graph_mode
    config["scoring_mode"] = args.scoring_mode
    if args.scoring_mode in ("gedig", "gedig_refine"):
        config["gedig_scoring_lambda"] = args.gedig_scoring_lambda
        config["gedig_scoring_sp_beta"] = args.gedig_scoring_sp_beta
        config["gedig_scoring_k_hop"] = args.gedig_scoring_k_hop
        config["gedig_scoring_mp_iterations"] = args.gedig_scoring_mp_iterations
        config["gedig_scoring_mp_alpha"] = args.gedig_scoring_mp_alpha
    if args.pointwise_rerank:
        config["pointwise_rerank"] = True
        config["pointwise_rerank_top_k"] = args.pointwise_rerank_top_k
        config["pointwise_batch_size"] = args.pointwise_batch_size
        config["pointwise_blend_weight"] = args.pointwise_blend_weight
    if args.reasoning_rerank:
        config["reasoning_rerank"] = True
        config["rerank_model"] = args.rerank_model or args.model
        config["reasoning_rerank_top_k"] = args.reasoning_rerank_top_k
        config["reasoning_rerank_doc_chars"] = args.reasoning_rerank_doc_chars
        config["reasoning_rerank_blend_weight"] = args.reasoning_rerank_blend_weight
    if args.query_decomp:
        config["query_decomp"] = True
        config["query_decomp_top_k"] = args.query_decomp_top_k
        config["query_decomp_max_sub"] = args.query_decomp_max_sub
    if args.ria_loop:
        config["ria_loop"] = True
        config["ria_max_rounds"] = args.ria_max_rounds
        config["ria_docs_per_round"] = args.ria_docs_per_round
        config["ria_feedback_top_k"] = args.ria_feedback_top_k
        config["ria_beta0_target"] = args.ria_beta0_target
    if args.token_graph:
        config["token_graph"] = True
        config["token_graph_weight"] = args.token_graph_weight
        config["token_graph_max_tokens"] = args.token_graph_max_tokens
        config["token_graph_walk_score"] = args.token_graph_walk_score
        config["token_graph_dg_penalty"] = args.token_graph_dg_penalty
        config["token_graph_f_eval"] = args.token_graph_f_eval
        config["token_graph_f_lambda"] = args.token_graph_f_lambda
        config["token_graph_insight"] = args.token_graph_insight
    if args.early_token_graph:
        config["early_token_graph"] = True
        config["early_tg_top_k"] = args.early_tg_top_k
    if args.enhanced_graph:
        config["enhanced_graph"] = True
    if args.progressive_routing:
        config["progressive_routing"] = True
        config["progressive_ag_threshold"] = args.progressive_ag_threshold
        config["progressive_dg_shallow_threshold"] = args.progressive_dg_shallow_threshold
    if args.unified_graph:
        config["unified_graph"] = True
        config["aght_lambda"] = args.aght_lambda
        config["aght_mp_iterations"] = args.aght_mp_iterations
        config["aght_mp_alpha"] = args.aght_mp_alpha
        config["aght_w_q1"] = args.aght_w_q1
        config["aght_f_lambda"] = args.aght_f_lambda
    if args.entity_feval:
        config["entity_feval"] = True
        config["entity_feval_weight"] = args.entity_feval_weight
        config["entity_feval_lambda"] = args.entity_feval_lambda
        config["entity_feval_version"] = args.entity_feval_version
        if args.ag_threshold is not None:
            config["ag_threshold"] = args.ag_threshold
        if args.ag_max_k is not None:
            config["ag_max_k"] = args.ag_max_k
        config["ag_min_k"] = args.ag_min_k
        config["beta1_weight"] = args.beta1_weight
    if args.gedig_loop:
        config["gedig_loop"] = args.gedig_loop
        config["gedig_loop_max_rounds"] = args.gedig_loop_max_rounds
        config["gedig_loop_delta_beta1_target"] = args.gedig_loop_delta_beta1_target
        config["gedig_loop_directed"] = args.gedig_loop_directed
        config["gedig_loop_max_new_docs"] = args.gedig_loop_max_new_docs
    if args.n_cot_ensemble > 1:
        config["n_cot_ensemble"] = args.n_cot_ensemble
        config["cot_cache_dir"] = args.cot_cache_dir
        config["cot_temperature"] = args.cot_temperature
    if args.mode in ("cot_rerank", "cot_retrieval", "adaptive_retrieval", "unified", "gedig_routing"):
        config["model"] = args.model
        config["cot_weight"] = args.cot_weight
    if args.mode in ("cot_retrieval", "adaptive_retrieval", "unified", "gedig_routing"):
        config["cot_retrieval_top_k"] = args.cot_retrieval_top_k
        config["cot_retrieval_max_concepts"] = args.cot_retrieval_max_concepts
    if args.mode == "adaptive_retrieval":
        config["beta_low"] = args.beta_low
        config["beta_high"] = args.beta_high
        config["aggressive_top_k"] = args.aggressive_top_k
        config["aggressive_max_concepts"] = args.aggressive_max_concepts
    if args.mode == "unified":
        config["dense_index_dir"] = args.dense_index_dir
        config["dense_top_k"] = args.dense_top_k
        config["dense_cot_top_k"] = args.dense_cot_top_k
        config["dense_sim_threshold"] = args.dense_sim_threshold
        config["llm_rerank"] = args.llm_rerank
        config["llm_rerank_top_k"] = args.llm_rerank_top_k
    if args.mode == "gedig_routing":
        config["episode_index_dir"] = args.episode_index_dir
        config["dense_index_dir"] = args.dense_index_dir
        config["gedig_lambda"] = args.gedig_lambda
        config["gedig_max_hops"] = args.gedig_max_hops
        config["gedig_sp_beta"] = args.gedig_sp_beta
        config["gedig_tau_dg"] = args.gedig_tau_dg
        config["gedig_tau_ag"] = args.gedig_tau_ag
        config["gedig_k_target"] = args.gedig_k_target
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Initialize pipeline
    dense_retriever = None  # Will be set per-domain for unified mode
    dense_retriever_cot = None  # Will be set for cot_retrieval/adaptive_retrieval modes

    # geDIG routing components (shared across domains)
    gedig_router = None
    episode_index = None

    if args.mode == "gedig_routing":
        from bright_cot_pipeline import BrightCoTPipeline
        from dense_retriever import DenseRetriever
        from episode_graph import EpisodeIndex, EpisodeGraphBuilder
        from gedig_router import GeDIGRouter

        if not args.episode_index_dir:
            print("ERROR: --episode-index-dir required for gedig_routing mode")
            sys.exit(1)

        # Initialize geDIG router
        gedig_router = GeDIGRouter(
            lambda_weight=args.gedig_lambda,
            max_hops=args.gedig_max_hops,
            sp_beta=args.gedig_sp_beta,
            tau_dg=args.gedig_tau_dg,
            tau_ag=args.gedig_tau_ag,
        )

        # Initialize episode index
        episode_index = EpisodeIndex(args.episode_index_dir)

        # Dense retriever (optional but recommended for geDIG routing)
        if args.dense_index_dir:
            dense_retriever = DenseRetriever(index_dir=args.dense_index_dir)
        else:
            dense_retriever = None

        # Pipeline created per-domain (needs episode loading)
        pipeline = None
        print(f"  geDIG Router: λ={args.gedig_lambda}, τ_dg={args.gedig_tau_dg}, τ_ag={args.gedig_tau_ag}")
        print(f"  Episode index: {args.episode_index_dir}")

    elif args.mode == "unified":
        from bright_cot_pipeline import BrightCoTPipeline
        from dense_retriever import DenseRetriever
        if not args.dense_index_dir:
            print("ERROR: --dense-index-dir required for unified mode")
            sys.exit(1)
        # DenseRetriever is shared; index loaded per-domain in the loop below
        dense_retriever = DenseRetriever(index_dir=args.dense_index_dir)
        # Pipeline created per-domain (needs dense_domain)
        pipeline = None

    elif args.mode in ("cot_rerank", "cot_retrieval", "adaptive_retrieval"):
        from bright_cot_pipeline import BrightCoTPipeline

        # Dense retrieval for cot_retrieval mode (Spec I: dense pool expansion)
        dense_retriever_cot = None
        if args.dense_index_dir and args.mode in ("cot_retrieval", "adaptive_retrieval"):
            from dense_retriever import DenseRetriever
            dense_retriever_cot = DenseRetriever(index_dir=args.dense_index_dir)
            config["dense_index_dir"] = args.dense_index_dir
            config["dense_top_k"] = args.dense_top_k
            config["dense_cot_top_k"] = args.dense_cot_top_k
            config["dense_sim_threshold"] = args.dense_sim_threshold

        # Episode index loading (for graph_mode=episode/hybrid, Spec F/G)
        episode_index_for_graph = None
        if args.graph_mode in ("episode", "hybrid"):
            if not args.episode_index_dir:
                print(f"ERROR: --episode-index-dir required for --graph-mode {args.graph_mode}")
                sys.exit(1)
            from episode_graph import EpisodeIndex
            episode_index_for_graph = EpisodeIndex(args.episode_index_dir)

        if args.graph_mode in ("episode", "hybrid") or dense_retriever_cot is not None:
            # Per-domain pipeline (needs dense_domain for episode/dense lookup)
            pipeline = None
        else:
            pipeline = BrightCoTPipeline(
                model=args.model,
                initial_top_k=args.initial_top_k,
                graph_top_k=args.graph_top_k,
                rerank_top_k=args.rerank_top_k,
                rerank_alpha=args.rerank_alpha,
                max_para_freq=args.max_para_freq,
                cot_weight=args.cot_weight,
                cot_retrieval_top_k=args.cot_retrieval_top_k,
                cot_retrieval_max_concepts=args.cot_retrieval_max_concepts,
                enable_cot_retrieval=(args.mode in ("cot_retrieval", "adaptive_retrieval")),
                enable_adaptive=(args.mode == "adaptive_retrieval"),
                beta_low=args.beta_low,
                beta_high=args.beta_high,
                aggressive_top_k=args.aggressive_top_k,
                aggressive_max_concepts=args.aggressive_max_concepts,
                # geDIG scoring (Spec H)
                scoring_mode=args.scoring_mode,
                gedig_scoring_lambda=args.gedig_scoring_lambda,
                gedig_scoring_sp_beta=args.gedig_scoring_sp_beta,
                gedig_scoring_k_hop=args.gedig_scoring_k_hop,
                gedig_scoring_mp_iterations=args.gedig_scoring_mp_iterations,
                gedig_scoring_mp_alpha=args.gedig_scoring_mp_alpha,
                # Pointwise reranking (Spec J)
                enable_pointwise_rerank=args.pointwise_rerank,
                pointwise_rerank_top_k=args.pointwise_rerank_top_k,
                pointwise_batch_size=args.pointwise_batch_size,
                pointwise_blend_weight=args.pointwise_blend_weight,
                # Query decomposition (Spec K)
                enable_query_decomp=args.query_decomp,
                query_decomp_top_k=args.query_decomp_top_k,
                query_decomp_max_sub=args.query_decomp_max_sub,
                # Reasoning reranking (Spec L)
                enable_reasoning_rerank=args.reasoning_rerank,
                rerank_model=args.rerank_model,
                reasoning_rerank_top_k=args.reasoning_rerank_top_k,
                reasoning_rerank_doc_chars=args.reasoning_rerank_doc_chars,
                reasoning_rerank_blend_weight=args.reasoning_rerank_blend_weight,
                # RIA expansion (Spec M)
                enable_ria_loop=args.ria_loop,
                ria_max_rounds=args.ria_max_rounds,
                ria_docs_per_round=args.ria_docs_per_round,
                ria_feedback_top_k=args.ria_feedback_top_k,
                ria_beta0_target=args.ria_beta0_target,
                enable_token_graph=args.token_graph,
                token_graph_weight=args.token_graph_weight,
                token_graph_max_tokens=args.token_graph_max_tokens,
                token_graph_walk_score=args.token_graph_walk_score,
                token_graph_dg_penalty=args.token_graph_dg_penalty,
                token_graph_f_eval=args.token_graph_f_eval,
                token_graph_f_lambda=args.token_graph_f_lambda,
                token_graph_insight_mode=args.token_graph_insight,
                # Early Token Graph (Spec W)
                enable_early_token_graph=args.early_token_graph,
                early_tg_top_k=args.early_tg_top_k,
                # Enhanced Graph (Spec X)
                enable_enhanced_graph=args.enhanced_graph,
                # Progressive DG Escalation (Spec Y)
                enable_progressive_routing=args.progressive_routing,
                progressive_ag_threshold=args.progressive_ag_threshold,
                progressive_dg_shallow_threshold=args.progressive_dg_shallow_threshold,
                # Unified Heterogeneous Graph Transformer (Spec Z)
                enable_unified_graph=args.unified_graph,
                aght_lambda=args.aght_lambda,
                aght_mp_iterations=args.aght_mp_iterations,
                aght_mp_alpha=args.aght_mp_alpha,
                aght_w_q1=args.aght_w_q1,
                aght_f_lambda=args.aght_f_lambda,
                aght_use_unified_feval=getattr(args, "aght_use_unified", False),
                enable_entity_feval=args.entity_feval,
                entity_feval_weight=args.entity_feval_weight,
                entity_feval_lambda=args.entity_feval_lambda,
                entity_feval_version=args.entity_feval_version,
                entity_feval_ag_threshold=args.ag_threshold,
                entity_feval_ag_max_k=args.ag_max_k,
                entity_feval_ag_min_k=args.ag_min_k,
                entity_feval_beta1_weight=args.beta1_weight,
                # geDIG CoT Loop (Spec R)
                enable_gedig_loop=args.gedig_loop,
                gedig_loop_max_rounds=args.gedig_loop_max_rounds,
                gedig_loop_delta_beta1_target=args.gedig_loop_delta_beta1_target,
                gedig_loop_directed=args.gedig_loop_directed,
                gedig_loop_max_new_docs=args.gedig_loop_max_new_docs,
                # Multi-CoT Ensemble (Spec P)
                n_cot_ensemble=args.n_cot_ensemble,
                cot_cache_dir=args.cot_cache_dir,
                cot_temperature=args.cot_temperature,
            )
    elif args.mode == "graph_rerank":
        pipeline = BrightPipeline(
            initial_top_k=args.initial_top_k,
            graph_top_k=args.graph_top_k,
            rerank_top_k=args.rerank_top_k,
            rerank_alpha=args.rerank_alpha,
            max_para_freq=args.max_para_freq,
        )
    else:
        # bm25_only: use pipeline with alpha=1.0 (100% BM25)
        pipeline = BrightPipeline(
            initial_top_k=args.initial_top_k,
            graph_top_k=0,  # no graph
            rerank_top_k=args.rerank_top_k,
            rerank_alpha=1.0,
            max_para_freq=args.max_para_freq,
        )

    # Per-domain results
    all_domain_summaries = {}
    all_records = []

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        docs_path = data_dir / f"{domain}_docs.jsonl"
        queries_path = data_dir / f"{domain}_queries.jsonl"

        if not docs_path.exists():
            print(f"  ERROR: {docs_path} not found. Skipping.")
            continue
        if not queries_path.exists():
            print(f"  ERROR: {queries_path} not found. Skipping.")
            continue

        # Load queries
        queries = []
        with open(queries_path) as f:
            for line in f:
                queries.append(json.loads(line))
        total = len(queries) if args.limit is None else min(args.limit, len(queries))
        queries = queries[:total]
        print(f"  Queries: {total}")

        # Build BM25 index
        engine_label = f"({args.bm25_engine})" if args.bm25_engine != "rank_bm25" else ""
        print(f"  Building BM25 index {engine_label} from {docs_path}...")
        t_idx = time.time()
        lucene_idx = str(docs_path).replace(".jsonl", "_lucene_index") if args.bm25_engine == "pyserini" else None
        bm25_index, docs = build_bm25_index(
            str(docs_path),
            engine=args.bm25_engine,
            lucene_index_path=lucene_idx,
            pyserini_k1=args.pyserini_k1,
            pyserini_b=args.pyserini_b,
        )
        idx_time = time.time() - t_idx
        print(f"  Index built: {len(docs)} docs in {idx_time:.1f}s")

        # Unified mode: load dense index and create pipeline per-domain
        if args.mode == "unified":
            print(f"  Loading dense index for {domain}...")
            t_dense = time.time()
            dense_retriever.load_index(domain)
            print(f"  Dense index loaded in {time.time() - t_dense:.1f}s")
            pipeline = BrightCoTPipeline(
                model=args.model,
                initial_top_k=args.initial_top_k,
                graph_top_k=args.graph_top_k,
                rerank_top_k=args.rerank_top_k,
                rerank_alpha=args.rerank_alpha,
                max_para_freq=args.max_para_freq,
                cot_weight=args.cot_weight,
                cot_retrieval_top_k=args.aggressive_top_k,
                cot_retrieval_max_concepts=args.aggressive_max_concepts,
                enable_cot_retrieval=True,
                enable_adaptive=False,
                dense_retriever=dense_retriever,
                dense_domain=domain,
                dense_top_k=args.dense_top_k,
                dense_cot_top_k=args.dense_cot_top_k,
                dense_sim_threshold=args.dense_sim_threshold,
                enable_llm_rerank=args.llm_rerank,
                llm_rerank_top_k=args.llm_rerank_top_k,
                # Multi-CoT Ensemble (Spec P)
                n_cot_ensemble=args.n_cot_ensemble,
                cot_cache_dir=args.cot_cache_dir,
                cot_temperature=args.cot_temperature,
            )

        # geDIG routing mode: load episodes and create pipeline per-domain
        if args.mode == "gedig_routing":
            print(f"  Loading episode index for {domain}...")
            episode_index.load_domain(domain)

            # Load dense index if available
            dr_for_domain = None
            if dense_retriever is not None:
                print(f"  Loading dense index for {domain}...")
                t_dense = time.time()
                dense_retriever.load_index(domain)
                print(f"  Dense index loaded in {time.time() - t_dense:.1f}s")
                dr_for_domain = dense_retriever

            # Create episode graph builder
            from episode_graph import EpisodeGraphBuilder
            ep_builder = EpisodeGraphBuilder(
                k_target=args.gedig_k_target,
                dense_retriever=dr_for_domain,
                dense_domain=domain,
            )

            pipeline = BrightCoTPipeline(
                model=args.model,
                initial_top_k=args.initial_top_k,
                graph_top_k=args.graph_top_k,
                rerank_top_k=args.rerank_top_k,
                rerank_alpha=args.rerank_alpha,
                max_para_freq=args.max_para_freq,
                cot_weight=args.cot_weight,
                cot_retrieval_top_k=args.cot_retrieval_top_k,
                cot_retrieval_max_concepts=args.cot_retrieval_max_concepts,
                enable_cot_retrieval=True,
                enable_adaptive=False,
                dense_retriever=dr_for_domain,
                dense_domain=domain,
                dense_top_k=args.dense_top_k if args.dense_index_dir else 0,
                dense_cot_top_k=args.dense_cot_top_k if args.dense_index_dir else 0,
                dense_sim_threshold=args.dense_sim_threshold,
                # geDIG routing components
                gedig_router=gedig_router,
                episode_index=episode_index,
                episode_graph_builder=ep_builder,
                # Multi-CoT Ensemble (Spec P)
                n_cot_ensemble=args.n_cot_ensemble,
                cot_cache_dir=args.cot_cache_dir,
                cot_temperature=args.cot_temperature,
            )
            print(f"  geDIG routing pipeline initialized for {domain}")

        # Episode/hybrid graph mode (Spec F/G): load episodes and create pipeline per-domain
        if args.graph_mode in ("episode", "hybrid") and args.mode in ("cot_rerank", "cot_retrieval", "adaptive_retrieval"):
            print(f"  Loading episodes for {domain} (graph_mode={args.graph_mode})...")
            episode_index_for_graph.load_domain(domain)
            pipeline = BrightCoTPipeline(
                model=args.model,
                initial_top_k=args.initial_top_k,
                graph_top_k=args.graph_top_k,
                rerank_top_k=args.rerank_top_k,
                rerank_alpha=args.rerank_alpha,
                max_para_freq=args.max_para_freq,
                cot_weight=args.cot_weight,
                cot_retrieval_top_k=args.cot_retrieval_top_k,
                cot_retrieval_max_concepts=args.cot_retrieval_max_concepts,
                enable_cot_retrieval=(args.mode in ("cot_retrieval", "adaptive_retrieval")),
                enable_adaptive=(args.mode == "adaptive_retrieval"),
                beta_low=args.beta_low,
                beta_high=args.beta_high,
                aggressive_top_k=args.aggressive_top_k,
                aggressive_max_concepts=args.aggressive_max_concepts,
                graph_mode=args.graph_mode,
                episode_index=episode_index_for_graph,
                dense_domain=domain,
                # Dense retrieval (if available)
                dense_retriever=dense_retriever_cot,
                dense_top_k=args.dense_top_k,
                dense_cot_top_k=args.dense_cot_top_k,
                dense_sim_threshold=args.dense_sim_threshold,
                # geDIG scoring (Spec H)
                scoring_mode=args.scoring_mode,
                gedig_scoring_lambda=args.gedig_scoring_lambda,
                gedig_scoring_sp_beta=args.gedig_scoring_sp_beta,
                gedig_scoring_k_hop=args.gedig_scoring_k_hop,
                gedig_scoring_mp_iterations=args.gedig_scoring_mp_iterations,
                gedig_scoring_mp_alpha=args.gedig_scoring_mp_alpha,
                # Pointwise reranking (Spec J)
                enable_pointwise_rerank=args.pointwise_rerank,
                pointwise_rerank_top_k=args.pointwise_rerank_top_k,
                pointwise_batch_size=args.pointwise_batch_size,
                pointwise_blend_weight=args.pointwise_blend_weight,
                # Query decomposition (Spec K)
                enable_query_decomp=args.query_decomp,
                query_decomp_top_k=args.query_decomp_top_k,
                query_decomp_max_sub=args.query_decomp_max_sub,
                # Reasoning reranking (Spec L)
                enable_reasoning_rerank=args.reasoning_rerank,
                rerank_model=args.rerank_model,
                reasoning_rerank_top_k=args.reasoning_rerank_top_k,
                reasoning_rerank_doc_chars=args.reasoning_rerank_doc_chars,
                reasoning_rerank_blend_weight=args.reasoning_rerank_blend_weight,
                # RIA expansion (Spec M)
                enable_ria_loop=args.ria_loop,
                ria_max_rounds=args.ria_max_rounds,
                ria_docs_per_round=args.ria_docs_per_round,
                ria_feedback_top_k=args.ria_feedback_top_k,
                ria_beta0_target=args.ria_beta0_target,
                enable_token_graph=args.token_graph,
                token_graph_weight=args.token_graph_weight,
                token_graph_max_tokens=args.token_graph_max_tokens,
                token_graph_walk_score=args.token_graph_walk_score,
                token_graph_dg_penalty=args.token_graph_dg_penalty,
                token_graph_f_eval=args.token_graph_f_eval,
                token_graph_f_lambda=args.token_graph_f_lambda,
                token_graph_insight_mode=args.token_graph_insight,
                # Early Token Graph (Spec W)
                enable_early_token_graph=args.early_token_graph,
                early_tg_top_k=args.early_tg_top_k,
                # Enhanced Graph (Spec X)
                enable_enhanced_graph=args.enhanced_graph,
                # Progressive DG Escalation (Spec Y)
                enable_progressive_routing=args.progressive_routing,
                progressive_ag_threshold=args.progressive_ag_threshold,
                progressive_dg_shallow_threshold=args.progressive_dg_shallow_threshold,
                # Unified Heterogeneous Graph Transformer (Spec Z)
                enable_unified_graph=args.unified_graph,
                aght_lambda=args.aght_lambda,
                aght_mp_iterations=args.aght_mp_iterations,
                aght_mp_alpha=args.aght_mp_alpha,
                aght_w_q1=args.aght_w_q1,
                aght_f_lambda=args.aght_f_lambda,
                aght_use_unified_feval=getattr(args, "aght_use_unified", False),
                enable_entity_feval=args.entity_feval,
                entity_feval_weight=args.entity_feval_weight,
                entity_feval_lambda=args.entity_feval_lambda,
                entity_feval_version=args.entity_feval_version,
                entity_feval_ag_threshold=args.ag_threshold,
                entity_feval_ag_max_k=args.ag_max_k,
                entity_feval_ag_min_k=args.ag_min_k,
                entity_feval_beta1_weight=args.beta1_weight,
                # geDIG CoT Loop (Spec R)
                enable_gedig_loop=args.gedig_loop,
                gedig_loop_max_rounds=args.gedig_loop_max_rounds,
                gedig_loop_delta_beta1_target=args.gedig_loop_delta_beta1_target,
                gedig_loop_directed=args.gedig_loop_directed,
                gedig_loop_max_new_docs=args.gedig_loop_max_new_docs,
                # Multi-CoT Ensemble (Spec P)
                n_cot_ensemble=args.n_cot_ensemble,
                cot_cache_dir=args.cot_cache_dir,
                cot_temperature=args.cot_temperature,
            )
            print(f"  {args.graph_mode.capitalize()} graph pipeline initialized for {domain}")

        # Dense retrieval per-domain: load index and create pipeline
        elif dense_retriever_cot is not None and args.mode in ("cot_retrieval", "adaptive_retrieval"):
            print(f"  Loading dense index for {domain}...")
            t_dense = time.time()
            dense_retriever_cot.load_index(domain)
            print(f"  Dense index loaded in {time.time() - t_dense:.1f}s")
            pipeline = BrightCoTPipeline(
                model=args.model,
                initial_top_k=args.initial_top_k,
                graph_top_k=args.graph_top_k,
                rerank_top_k=args.rerank_top_k,
                rerank_alpha=args.rerank_alpha,
                max_para_freq=args.max_para_freq,
                cot_weight=args.cot_weight,
                cot_retrieval_top_k=args.cot_retrieval_top_k,
                cot_retrieval_max_concepts=args.cot_retrieval_max_concepts,
                enable_cot_retrieval=True,
                enable_adaptive=(args.mode == "adaptive_retrieval"),
                beta_low=args.beta_low,
                beta_high=args.beta_high,
                aggressive_top_k=args.aggressive_top_k,
                aggressive_max_concepts=args.aggressive_max_concepts,
                # Dense retrieval
                dense_retriever=dense_retriever_cot,
                dense_domain=domain,
                dense_top_k=args.dense_top_k,
                dense_cot_top_k=args.dense_cot_top_k,
                dense_sim_threshold=args.dense_sim_threshold,
                # geDIG scoring (Spec H)
                scoring_mode=args.scoring_mode,
                gedig_scoring_lambda=args.gedig_scoring_lambda,
                gedig_scoring_sp_beta=args.gedig_scoring_sp_beta,
                gedig_scoring_k_hop=args.gedig_scoring_k_hop,
                gedig_scoring_mp_iterations=args.gedig_scoring_mp_iterations,
                gedig_scoring_mp_alpha=args.gedig_scoring_mp_alpha,
                # Pointwise reranking (Spec J)
                enable_pointwise_rerank=args.pointwise_rerank,
                pointwise_rerank_top_k=args.pointwise_rerank_top_k,
                pointwise_batch_size=args.pointwise_batch_size,
                pointwise_blend_weight=args.pointwise_blend_weight,
                # Query decomposition (Spec K)
                enable_query_decomp=args.query_decomp,
                query_decomp_top_k=args.query_decomp_top_k,
                query_decomp_max_sub=args.query_decomp_max_sub,
                # Reasoning reranking (Spec L)
                enable_reasoning_rerank=args.reasoning_rerank,
                rerank_model=args.rerank_model,
                reasoning_rerank_top_k=args.reasoning_rerank_top_k,
                reasoning_rerank_doc_chars=args.reasoning_rerank_doc_chars,
                reasoning_rerank_blend_weight=args.reasoning_rerank_blend_weight,
                # RIA expansion (Spec M)
                enable_ria_loop=args.ria_loop,
                ria_max_rounds=args.ria_max_rounds,
                ria_docs_per_round=args.ria_docs_per_round,
                ria_feedback_top_k=args.ria_feedback_top_k,
                ria_beta0_target=args.ria_beta0_target,
                enable_token_graph=args.token_graph,
                token_graph_weight=args.token_graph_weight,
                token_graph_max_tokens=args.token_graph_max_tokens,
                token_graph_walk_score=args.token_graph_walk_score,
                token_graph_dg_penalty=args.token_graph_dg_penalty,
                token_graph_f_eval=args.token_graph_f_eval,
                token_graph_f_lambda=args.token_graph_f_lambda,
                token_graph_insight_mode=args.token_graph_insight,
                # Early Token Graph (Spec W)
                enable_early_token_graph=args.early_token_graph,
                early_tg_top_k=args.early_tg_top_k,
                # Enhanced Graph (Spec X)
                enable_enhanced_graph=args.enhanced_graph,
                # Progressive DG Escalation (Spec Y)
                enable_progressive_routing=args.progressive_routing,
                progressive_ag_threshold=args.progressive_ag_threshold,
                progressive_dg_shallow_threshold=args.progressive_dg_shallow_threshold,
                # Unified Heterogeneous Graph Transformer (Spec Z)
                enable_unified_graph=args.unified_graph,
                aght_lambda=args.aght_lambda,
                aght_mp_iterations=args.aght_mp_iterations,
                aght_mp_alpha=args.aght_mp_alpha,
                aght_w_q1=args.aght_w_q1,
                aght_f_lambda=args.aght_f_lambda,
                aght_use_unified_feval=getattr(args, "aght_use_unified", False),
                enable_entity_feval=args.entity_feval,
                entity_feval_weight=args.entity_feval_weight,
                entity_feval_lambda=args.entity_feval_lambda,
                entity_feval_version=args.entity_feval_version,
                entity_feval_ag_threshold=args.ag_threshold,
                entity_feval_ag_max_k=args.ag_max_k,
                entity_feval_ag_min_k=args.ag_min_k,
                entity_feval_beta1_weight=args.beta1_weight,
                # geDIG CoT Loop (Spec R)
                enable_gedig_loop=args.gedig_loop,
                gedig_loop_max_rounds=args.gedig_loop_max_rounds,
                gedig_loop_delta_beta1_target=args.gedig_loop_delta_beta1_target,
                gedig_loop_directed=args.gedig_loop_directed,
                gedig_loop_max_new_docs=args.gedig_loop_max_new_docs,
                # Multi-CoT Ensemble (Spec P)
                n_cot_ensemble=args.n_cot_ensemble,
                cot_cache_dir=args.cot_cache_dir,
                cot_temperature=args.cot_temperature,
            )
            print(f"  Dense+CoT pipeline initialized for {domain}")

        # Results file (per-domain)
        domain_results_file = out_dir / f"{domain}_results.jsonl"
        done_ids: set[str] = set()

        if domain_results_file.exists():
            with open(domain_results_file) as f:
                for line in f:
                    r = json.loads(line)
                    done_ids.add(str(r["query_id"]))
            print(f"  Resuming: {len(done_ids)} already done")

        # Run queries
        domain_records = []
        print(f"\n  Running {args.mode} on {total} queries...\n")

        for q_idx, q in enumerate(queries):
            query_id = str(q["id"])
            if query_id in done_ids:
                continue

            query_text = q["query"]
            gold_ids = set(q["gold_ids"]) if q["gold_ids"] else set()
            excluded_raw = q.get("excluded_ids", [])
            excluded_ids = set(excluded_raw) if excluded_raw and excluded_raw != ["N/A"] else set()

            print(f"  [{q_idx+1}/{total}] Q{query_id}: {query_text[:60]}...")

            t0 = time.time()

            try:
                if args.mode == "bm25_only":
                    # Pure BM25: just get top-k without graph
                    result = _bm25_only_rerank(
                        query_text, query_id, bm25_index, docs,
                        top_k=args.initial_top_k,
                        rerank_top_k=args.rerank_top_k,
                        excluded_ids=excluded_ids,
                    )
                else:
                    result = pipeline.rerank(
                        query=query_text,
                        query_id=query_id,
                        bm25_index=bm25_index,
                        docs=docs,
                        gold_ids=gold_ids,
                        excluded_ids=excluded_ids,
                    )

                # Compute metrics
                ndcg_10 = compute_ndcg_at_k(result.ranked_doc_ids, gold_ids, k=10)
                recall_10 = compute_recall_at_k(result.ranked_doc_ids, gold_ids, k=10)
                mrr = compute_mrr(result.ranked_doc_ids, gold_ids)

                # Also compute BM25 baseline metrics for comparison
                bm25_ndcg = compute_ndcg_at_k(result.bm25_doc_ids, gold_ids, k=10)
                bm25_recall = compute_recall_at_k(result.bm25_doc_ids, gold_ids, k=10)
                bm25_mrr = compute_mrr(result.bm25_doc_ids, gold_ids)

                latency = (time.time() - t0) * 1000

                record = {
                    "domain": domain,
                    "query_id": query_id,
                    "query": query_text[:200],
                    "mode": args.mode,
                    "graph_mode": args.graph_mode,
                    "ndcg_10": round(ndcg_10, 4),
                    "recall_10": round(recall_10, 4),
                    "mrr": round(mrr, 4),
                    "bm25_ndcg_10": round(bm25_ndcg, 4),
                    "bm25_recall_10": round(bm25_recall, 4),
                    "bm25_mrr": round(bm25_mrr, 4),
                    "n_gold": len(gold_ids),
                    "beta_0": result.beta_0,
                    "beta_1": result.beta_1,
                    "n_graph_nodes": result.n_graph_nodes,
                    "n_graph_edges": result.n_graph_edges,
                    "n_docs_in_graph": result.n_docs_in_graph,
                    "latency_ms": round(latency, 1),
                    "error": None,
                }

                # Gold hit analysis
                hits_in_top10 = sum(1 for d in result.ranked_doc_ids if d in gold_ids)
                record["gold_hits_top10"] = hits_in_top10

                # CoT-specific metadata
                if args.mode in ("cot_rerank", "cot_retrieval", "adaptive_retrieval", "unified", "gedig_routing") and hasattr(result, "cot_entities"):
                    record["cot_entities"] = result.cot_entities
                    record["n_cot_nodes"] = result.n_cot_nodes_injected
                    record["n_cot_edges"] = result.n_cot_edges_created
                    record["cot_latency_ms"] = round(result.cot_latency_ms, 1)

                # CoT Re-retrieval diagnostics
                if args.mode in ("cot_retrieval", "adaptive_retrieval", "unified", "gedig_routing") and hasattr(result, "n_cot_retrieved"):
                    record["n_cot_retrieved"] = result.n_cot_retrieved
                    record["n_cot_new_gold"] = result.n_cot_new_gold
                    record["n_merged_candidates"] = result.n_merged_candidates
                    record["cot_retrieval_query"] = result.cot_retrieval_query[:200]

                # Adaptive routing diagnostics
                if args.mode == "adaptive_retrieval" and hasattr(result, "routing_tier"):
                    record["pre_beta_0"] = result.pre_beta_0
                    record["routing_tier"] = result.routing_tier
                    record["cot_skipped"] = result.cot_skipped

                # Unified pipeline diagnostics
                if args.mode == "unified" and hasattr(result, "n_dense_retrieved"):
                    record["n_dense_retrieved"] = result.n_dense_retrieved
                    record["n_dense_cot_retrieved"] = result.n_dense_cot_retrieved
                    record["n_dense_new_gold"] = result.n_dense_new_gold
                    record["n_dense_graph_edges"] = result.n_dense_graph_edges
                    record["llm_rerank_applied"] = result.llm_rerank_applied
                    record["routing_tier"] = result.routing_tier
                    record["cot_skipped"] = result.cot_skipped

                # Dense retrieval diagnostics for cot_retrieval mode
                if args.mode in ("cot_retrieval", "adaptive_retrieval") and args.dense_index_dir and hasattr(result, "n_dense_retrieved"):
                    record["n_dense_retrieved"] = result.n_dense_retrieved
                    record["n_dense_cot_retrieved"] = result.n_dense_cot_retrieved
                    record["n_dense_new_gold"] = result.n_dense_new_gold
                    record["n_dense_graph_edges"] = result.n_dense_graph_edges

                # geDIG routing diagnostics
                if args.mode == "gedig_routing" and hasattr(result, "gedig_value"):
                    record["gedig_value"] = round(result.gedig_value, 4)
                    record["gedig_delta_betti_0"] = result.gedig_delta_betti_0
                    record["gedig_ig_value"] = round(result.gedig_ig_value, 4)
                    record["gedig_ged_value"] = round(result.gedig_ged_value, 4)
                    record["gedig_delta_sp_rel"] = round(result.gedig_delta_sp_rel, 4)
                    record["gedig_computation_ms"] = round(result.gedig_computation_ms, 1)
                    record["routing_tier"] = result.routing_tier
                    record["cot_skipped"] = result.cot_skipped
                    record["n_doc_episodes"] = result.n_doc_episodes
                    record["n_query_episodes"] = result.n_query_episodes
                    record["n_episode_cross_edges"] = result.n_episode_cross_edges

                # geDIG scoring diagnostics (Spec H)
                if args.scoring_mode in ("gedig", "gedig_refine") and hasattr(result, "scoring_mode"):
                    record["scoring_mode"] = result.scoring_mode
                    record["gedig_scoring_lambda"] = result.gedig_scoring_lambda
                    record["gedig_scoring_sp_beta"] = result.gedig_scoring_sp_beta
                    record["n_edges_discovered"] = result.n_edges_discovered
                    record["n_edges_removed"] = result.n_edges_removed
                    record["mp_iterations_run"] = result.mp_iterations_run
                    record["avg_gedig_local"] = round(result.avg_gedig_local, 4)

                # Pointwise reranking diagnostics (Spec J)
                if hasattr(result, "pointwise_rerank_applied"):
                    record["pointwise_rerank_applied"] = result.pointwise_rerank_applied
                    record["pointwise_rerank_n_scored"] = result.pointwise_rerank_n_scored
                    record["pointwise_rerank_n_calls"] = result.pointwise_rerank_n_calls
                    record["pointwise_rerank_ms"] = round(result.pointwise_rerank_ms, 1)
                    record["pointwise_rerank_avg_score"] = round(result.pointwise_rerank_avg_score, 4)

                # Query decomposition diagnostics (Spec K)
                if hasattr(result, "query_decomp_applied"):
                    record["query_decomp_applied"] = result.query_decomp_applied
                    record["n_sub_queries"] = result.n_sub_queries
                    record["n_decomp_new_candidates"] = result.n_decomp_new_candidates
                    record["n_decomp_new_gold"] = result.n_decomp_new_gold
                    record["query_decomp_ms"] = round(result.query_decomp_ms, 1)

                # Reasoning reranking diagnostics (Spec L)
                if hasattr(result, "reasoning_rerank_applied"):
                    record["reasoning_rerank_applied"] = result.reasoning_rerank_applied
                    record["reasoning_rerank_model"] = result.reasoning_rerank_model
                    record["reasoning_rerank_n_scored"] = result.reasoning_rerank_n_scored
                    record["reasoning_rerank_n_calls"] = result.reasoning_rerank_n_calls
                    record["reasoning_rerank_ms"] = round(result.reasoning_rerank_ms, 1)
                    record["reasoning_rerank_avg_score"] = round(result.reasoning_rerank_avg_score, 4)

                # RIA diagnostics (Spec M)
                if hasattr(result, "ria_applied"):
                    record["ria_applied"] = result.ria_applied
                    record["ria_rounds"] = result.ria_rounds
                    record["ria_beta0_history"] = result.ria_beta0_history
                    record["ria_new_docs_per_round"] = result.ria_new_docs_per_round
                    record["ria_new_gold_per_round"] = result.ria_new_gold_per_round
                    record["ria_total_new_docs"] = result.ria_total_new_docs
                    record["ria_total_new_gold"] = result.ria_total_new_gold
                    record["ria_ms"] = round(result.ria_ms, 1)

                if hasattr(result, "token_graph_applied") and result.token_graph_applied:
                    record["token_graph_avg_coverage"] = round(result.token_graph_avg_coverage, 4)
                    record["token_graph_avg_proximity"] = round(result.token_graph_avg_proximity, 4)
                    record["token_graph_n_docs"] = result.token_graph_n_docs
                    record["token_graph_ms"] = round(result.token_graph_ms, 1)
                    record["token_graph_spearman_bm25"] = round(result.token_graph_spearman_bm25, 4)
                    record["token_graph_walk_score"] = result.token_graph_walk_score
                    record["token_graph_avg_beta1"] = round(result.token_graph_avg_beta1, 2)
                    if result.token_graph_f_eval:
                        record["token_graph_f_eval"] = True
                        record["token_graph_insight_mode"] = result.token_graph_insight_mode
                        record["token_graph_avg_n_insights"] = round(result.token_graph_avg_n_insights, 2)
                        record["token_graph_avg_f_theta"] = round(result.token_graph_avg_f_theta, 4)
                if result.entity_feval_applied:
                    record["entity_feval_applied"] = True
                    record["entity_feval_n_ag"] = result.entity_feval_n_ag
                    record["entity_feval_n_dg"] = result.entity_feval_n_dg
                    record["entity_feval_f_theta"] = round(result.entity_feval_f_theta, 4)
                    record["entity_feval_avg_convergence"] = round(result.entity_feval_avg_convergence, 4)
                    record["entity_feval_ranking_dg"] = round(result.entity_feval_ranking_dg, 4)
                    record["entity_feval_adaptive_weight"] = round(result.entity_feval_adaptive_weight, 4)
                    record["entity_feval_version"] = result.entity_feval_version
                    if result.entity_feval_version.startswith("v2"):
                        record["entity_feval_n_bridge"] = result.entity_feval_n_bridge
                        record["entity_feval_beta1_global"] = result.entity_feval_beta1_global
                        record["entity_feval_delta_beta1_global"] = result.entity_feval_delta_beta1_global
                        record["entity_feval_avg_delta_beta1"] = result.entity_feval_avg_delta_beta1
                        record["entity_feval_max_delta_beta1"] = result.entity_feval_max_delta_beta1
                        record["entity_feval_ag_threshold"] = result.entity_feval_ag_threshold
                        record["entity_feval_ag_density"] = result.entity_feval_ag_density
                        record["entity_feval_ag_n_query_edges"] = result.entity_feval_ag_n_query_edges

                # Progressive DG Escalation diagnostics (Spec Y)
                if hasattr(result, "progressive_routing") and result.progressive_routing:
                    record["progressive_tier"] = result.progressive_tier
                    record["progressive_dg_ratio"] = round(result.progressive_dg_ratio, 4)
                    record["progressive_shallow_n_new"] = result.progressive_shallow_n_new
                    record["progressive_shallow_gaps_before"] = result.progressive_shallow_gaps_before
                    record["progressive_shallow_gaps_after"] = result.progressive_shallow_gaps_after
                    record["progressive_escalated"] = result.progressive_escalated

                # geDIG CoT Loop diagnostics (Spec R)
                if hasattr(result, "gedig_loop_applied") and result.gedig_loop_applied:
                    record["gedig_loop_applied"] = True
                    record["gedig_loop_rounds"] = result.gedig_loop_rounds
                    record["gedig_loop_delta_beta1_history"] = result.gedig_loop_delta_beta1_history
                    record["gedig_loop_n_bridge_nodes"] = result.gedig_loop_n_bridge_nodes
                    record["gedig_loop_n_new_docs"] = result.gedig_loop_n_new_docs
                    record["gedig_loop_n_new_gold"] = result.gedig_loop_n_new_gold
                    record["gedig_loop_converged"] = result.gedig_loop_converged
                    record["gedig_loop_ms"] = round(result.gedig_loop_ms, 1)

                # Multi-CoT Ensemble diagnostics (Spec P)
                if hasattr(result, "ensemble_applied") and result.ensemble_applied:
                    record["ensemble_applied"] = True
                    record["ensemble_n_cots"] = result.ensemble_n_cots
                    record["ensemble_cot_cache_hit"] = result.ensemble_cot_cache_hit
                    record["ensemble_ms"] = round(result.ensemble_ms, 1)
                    record["ensemble_n_ag_docs"] = result.ensemble_n_ag_docs
                    record["ensemble_n_dg_docs"] = result.ensemble_n_dg_docs
                    record["ensemble_avg_agreement"] = round(result.ensemble_avg_agreement, 4)
                    record["ensemble_score_variance_mean"] = round(result.ensemble_score_variance_mean, 6)

                status = "+" if ndcg_10 > 0 else "-"
                delta = ndcg_10 - bm25_ndcg
                delta_str = f"Δ={delta:+.3f}" if args.mode != "bm25_only" else ""
                extra = ""
                if args.mode in ("cot_retrieval", "adaptive_retrieval", "unified", "gedig_routing") and hasattr(result, "n_cot_retrieved"):
                    extra = f" cot_ret={result.n_cot_retrieved} new_gold={result.n_cot_new_gold}"
                if args.mode == "adaptive_retrieval" and hasattr(result, "routing_tier"):
                    tier_label = {1: "SKIP", 2: "STD", 3: "AGG"}.get(result.routing_tier, "?")
                    extra += f" T{result.routing_tier}({tier_label}) pre_β₀={result.pre_beta_0}"
                if args.mode == "unified" and hasattr(result, "n_dense_retrieved"):
                    extra += f" dense={result.n_dense_retrieved}+{result.n_dense_cot_retrieved}"
                    extra += f" tierD={result.n_dense_graph_edges}"
                    if result.llm_rerank_applied:
                        extra += " LLM-RR"
                if args.mode in ("cot_retrieval", "adaptive_retrieval") and args.dense_index_dir and hasattr(result, "n_dense_retrieved"):
                    extra += f" dense={result.n_dense_retrieved}+{result.n_dense_cot_retrieved}"
                    extra += f" tierD={result.n_dense_graph_edges}"
                if args.mode == "gedig_routing" and hasattr(result, "gedig_value"):
                    tier_label = {1: "DG", 2: "MOD", 3: "AG"}.get(result.routing_tier, "?")
                    extra += f" geDIG={result.gedig_value:.3f} T{result.routing_tier}({tier_label})"
                    extra += f" Δβ₀={result.gedig_delta_betti_0}"
                    extra += f" ep={result.n_doc_episodes}d+{result.n_query_episodes}q"
                if args.scoring_mode == "gedig" and hasattr(result, "n_edges_discovered"):
                    extra += f" geDIG-S:disc={result.n_edges_discovered},rm={result.n_edges_removed}"
                    extra += f" avg_gDIG={result.avg_gedig_local:.3f}"
                if hasattr(result, "pointwise_rerank_applied") and result.pointwise_rerank_applied:
                    extra += f" PW={result.pointwise_rerank_n_scored}docs/{result.pointwise_rerank_n_calls}calls"
                    extra += f" avg={result.pointwise_rerank_avg_score:.2f}"
                if hasattr(result, "query_decomp_applied") and result.query_decomp_applied:
                    extra += f" QD={result.n_sub_queries}sq+{result.n_decomp_new_candidates}new"
                    if result.n_decomp_new_gold > 0:
                        extra += f"(+{result.n_decomp_new_gold}gold!)"
                if hasattr(result, "reasoning_rerank_applied") and result.reasoning_rerank_applied:
                    extra += f" RR={result.reasoning_rerank_n_scored}docs"
                    extra += f" avg={result.reasoning_rerank_avg_score:.2f}"
                    extra += f" {result.reasoning_rerank_ms:.0f}ms"
                if hasattr(result, "ria_applied") and result.ria_applied:
                    extra += f" RIA={result.ria_rounds}rnd"
                    extra += f" +{result.ria_total_new_docs}docs"
                    if result.ria_total_new_gold > 0:
                        extra += f"(+{result.ria_total_new_gold}gold!)"
                    extra += f" β₀={result.ria_beta0_history}"
                if hasattr(result, "token_graph_applied") and result.token_graph_applied:
                    extra += f" TG={result.token_graph_n_docs}docs"
                    extra += f" cov={result.token_graph_avg_coverage:.2f}"
                    extra += f" ρ={result.token_graph_spearman_bm25:.2f}"
                    extra += f" {result.token_graph_ms:.0f}ms"
                    if result.token_graph_walk_score:
                        extra += f" β₁={result.token_graph_avg_beta1:.1f}"
                    if result.token_graph_f_eval:
                        extra += f" F-eval θ={result.token_graph_avg_f_theta:.3f}"
                        if result.token_graph_insight_mode != "none":
                            extra += f" ins={result.token_graph_avg_n_insights:.1f}"
                if result.entity_feval_applied:
                    if result.entity_feval_version == "v2.1":
                        extra += f" EF-v2.1(AG={result.entity_feval_n_ag}/DG={result.entity_feval_n_dg} Δβ₁={result.entity_feval_delta_beta1_global} th={result.entity_feval_ag_threshold:.3f} d={result.entity_feval_ag_density:.4f})"
                    elif result.entity_feval_version == "v2":
                        extra += f" EF-v2(AG={result.entity_feval_n_ag}/DG={result.entity_feval_n_dg}/BR={result.entity_feval_n_bridge} β₁={result.entity_feval_beta1_global} conv={result.entity_feval_avg_convergence:.2f})"
                    else:
                        extra += f" EF(AG={result.entity_feval_n_ag}/DG={result.entity_feval_n_dg} θ={result.entity_feval_f_theta:.3f} conv={result.entity_feval_avg_convergence:.2f} rdg={result.entity_feval_ranking_dg:.3f} aw={result.entity_feval_adaptive_weight:.3f})"
                if hasattr(result, "gedig_loop_applied") and result.gedig_loop_applied:
                    conv_tag = "Y" if result.gedig_loop_converged else "N"
                    extra += f" LOOP(R={result.gedig_loop_rounds} Δβ₁={result.gedig_loop_delta_beta1_history} +{result.gedig_loop_n_new_docs}docs +{result.gedig_loop_n_new_gold}gold conv={conv_tag} {result.gedig_loop_ms:.0f}ms)"
                if hasattr(result, "ensemble_applied") and result.ensemble_applied:
                    cache_tag = "C" if result.ensemble_cot_cache_hit else "G"
                    extra += f" ENS(N={result.ensemble_n_cots}{cache_tag} AG={result.ensemble_n_ag_docs}/DG={result.ensemble_n_dg_docs} agr={result.ensemble_avg_agreement:.2f} var={result.ensemble_score_variance_mean:.4f} {result.ensemble_ms:.0f}ms)"
                print(
                    f"    {status} nDCG@10={ndcg_10:.3f} R@10={recall_10:.3f} "
                    f"MRR={mrr:.3f} {delta_str} "
                    f"β₀={result.beta_0} nodes={result.n_graph_nodes} "
                    f"({latency:.0f}ms){extra}"
                )

            except Exception as e:
                record = {
                    "domain": domain,
                    "query_id": query_id,
                    "query": query_text[:200],
                    "mode": args.mode,
                    "ndcg_10": 0.0,
                    "recall_10": 0.0,
                    "mrr": 0.0,
                    "error": str(e),
                }
                print(f"    ERROR: {e}")

            domain_records.append(record)
            all_records.append(record)

            # Write incrementally
            with open(domain_results_file, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Domain summary
        valid = [r for r in domain_records if r.get("error") is None]
        if valid:
            avg_ndcg = sum(r["ndcg_10"] for r in valid) / len(valid)
            avg_recall = sum(r["recall_10"] for r in valid) / len(valid)
            avg_mrr = sum(r["mrr"] for r in valid) / len(valid)
            avg_latency = sum(r["latency_ms"] for r in valid) / len(valid)

            domain_summary = {
                "domain": domain,
                "mode": args.mode,
                "n": len(valid),
                "n_errors": len(domain_records) - len(valid),
                "ndcg_10": round(avg_ndcg, 4),
                "recall_10": round(avg_recall, 4),
                "mrr": round(avg_mrr, 4),
                "avg_latency_ms": round(avg_latency, 1),
            }

            if args.mode == "graph_rerank":
                avg_bm25_ndcg = sum(r.get("bm25_ndcg_10", 0) for r in valid) / len(valid)
                avg_beta0 = sum(r.get("beta_0", 0) for r in valid) / len(valid)
                avg_beta1 = sum(r.get("beta_1", 0) for r in valid) / len(valid)
                avg_nodes = sum(r.get("n_graph_nodes", 0) for r in valid) / len(valid)
                avg_edges = sum(r.get("n_graph_edges", 0) for r in valid) / len(valid)

                domain_summary["bm25_ndcg_10"] = round(avg_bm25_ndcg, 4)
                domain_summary["ndcg_delta"] = round(avg_ndcg - avg_bm25_ndcg, 4)
                domain_summary["avg_beta_0"] = round(avg_beta0, 1)
                domain_summary["avg_beta_1"] = round(avg_beta1, 1)
                domain_summary["avg_graph_nodes"] = round(avg_nodes, 1)
                domain_summary["avg_graph_edges"] = round(avg_edges, 1)

            all_domain_summaries[domain] = domain_summary

            print(f"\n  --- {domain} Summary ---")
            print(f"    N={len(valid)}, errors={len(domain_records)-len(valid)}")
            print(f"    nDCG@10  = {avg_ndcg:.4f}")
            print(f"    Recall@10= {avg_recall:.4f}")
            print(f"    MRR      = {avg_mrr:.4f}")
            if args.mode == "graph_rerank":
                print(f"    BM25 nDCG= {avg_bm25_ndcg:.4f}")
                print(f"    Δ nDCG   = {avg_ndcg - avg_bm25_ndcg:+.4f}")
                print(f"    avg β₀   = {avg_beta0:.1f}")
                print(f"    avg nodes= {avg_nodes:.0f}")
        else:
            print(f"\n  No valid results for {domain}")

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    all_valid = [r for r in all_records if r.get("error") is None]
    if all_valid:
        overall_ndcg = sum(r["ndcg_10"] for r in all_valid) / len(all_valid)
        overall_recall = sum(r["recall_10"] for r in all_valid) / len(all_valid)
        overall_mrr = sum(r["mrr"] for r in all_valid) / len(all_valid)

        print(f"  Total queries: {len(all_valid)} valid, "
              f"{len(all_records)-len(all_valid)} errors")
        print(f"  Overall nDCG@10  = {overall_ndcg:.4f}")
        print(f"  Overall Recall@10= {overall_recall:.4f}")
        print(f"  Overall MRR      = {overall_mrr:.4f}")

        if args.mode == "graph_rerank":
            overall_bm25_ndcg = sum(
                r.get("bm25_ndcg_10", 0) for r in all_valid
            ) / len(all_valid)
            print(f"  BM25 nDCG@10     = {overall_bm25_ndcg:.4f}")
            print(f"  Δ nDCG           = {overall_ndcg - overall_bm25_ndcg:+.4f}")

        print(f"\n  Per domain:")
        for domain, ds in all_domain_summaries.items():
            delta = f" Δ={ds.get('ndcg_delta', 0):+.4f}" if args.mode == "graph_rerank" else ""
            print(f"    {domain:20s}  nDCG={ds['ndcg_10']:.4f}  "
                  f"R@10={ds['recall_10']:.4f}  MRR={ds['mrr']:.4f}{delta}")

    # Save overall summary
    overall_summary = {
        "mode": args.mode,
        "domains": domains,
        "n_total": len(all_valid),
        "n_errors": len(all_records) - len(all_valid),
        "overall_ndcg_10": round(overall_ndcg, 4) if all_valid else 0.0,
        "overall_recall_10": round(overall_recall, 4) if all_valid else 0.0,
        "overall_mrr": round(overall_mrr, 4) if all_valid else 0.0,
        "per_domain": all_domain_summaries,
        "config": config,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Results: {out_dir}")
    print(f"  Summary: {out_dir / 'summary.json'}")


def _bm25_only_rerank(
    query: str,
    query_id: str,
    bm25_index: object,
    docs: list[dict],
    top_k: int = 100,
    rerank_top_k: int = 10,
    excluded_ids: set[str] | None = None,
) -> BrightResult:
    """Pure BM25 retrieval without graph re-ranking."""
    t0 = time.time()
    excluded = excluded_ids or set()

    query_tokens = bm25_tokenize(query)
    bm25_scores = bm25_index.get_scores(query_tokens)

    scored = [
        (i, float(bm25_scores[i]))
        for i in range(len(docs))
        if docs[i]["id"] not in excluded
    ]
    scored.sort(key=lambda x: -x[1])
    top = scored[:top_k]

    doc_ids = [docs[i]["id"] for i, _ in top[:rerank_top_k]]
    scores = [s for _, s in top[:rerank_top_k]]

    return BrightResult(
        query_id=query_id,
        ranked_doc_ids=doc_ids,
        ranked_scores=scores,
        bm25_doc_ids=doc_ids,  # same as ranked for bm25_only
        beta_0=0,
        beta_1=0,
        n_graph_nodes=0,
        n_graph_edges=0,
        n_docs_in_graph=0,
        latency_ms=(time.time() - t0) * 1000,
    )


if __name__ == "__main__":
    main()
