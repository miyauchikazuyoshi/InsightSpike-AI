"""v12: BRIGHT — CoT × Graph Dynamic Augmentation Pipeline.

Chain-of-Thought reasoning entities are dynamically injected into
the entity graph, bridging the "reasoning gap" between query and
relevant documents.

Pipeline:
  1. BM25 initial retrieval (top-100)
  2. Entity graph construction from top-N documents
  3. LLM CoT reasoning about the query
  4. Extract entities from CoT → inject as virtual nodes into graph
  5. Re-compute graph scores with augmented graph
  6. Combined re-ranking: α·BM25 + (1-α)·augmented_graph

Usage::

    from bright_cot_pipeline import BrightCoTPipeline
    pipeline = BrightCoTPipeline(model="gpt-4o-mini")
    result = pipeline.rerank(query, query_id, bm25_index, docs)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import networkx as nx

from answerer import LLMAnswerer
from entity_graph import extract_entities, build_sentence_graph
from bright_pipeline import (
    BrightResult,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_mrr,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# CoT prompt template
# ---------------------------------------------------------------------------

_COT_PROMPT = """You are a search expert. Given a complex query, think step-by-step about what concepts, terms, and knowledge areas are needed to find relevant documents.

Query: {query}

Think step-by-step about:
1. What is the core question asking?
2. What domain knowledge or concepts are involved?
3. What technical terms, theories, or named entities are relevant?
4. What intermediate reasoning steps connect the query to potential answers?

Provide your reasoning in 3-5 sentences, focusing on specific concepts and terms."""


# ---------------------------------------------------------------------------
# Extended result with CoT info
# ---------------------------------------------------------------------------

@dataclass
class BrightCoTResult(BrightResult):
    """BrightResult with additional CoT metadata."""

    cot_text: str = ""
    cot_entities: list[str] = field(default_factory=list)
    n_cot_nodes_injected: int = 0
    n_cot_edges_created: int = 0
    cot_latency_ms: float = 0.0
    # CoT Re-retrieval fields
    n_cot_retrieved: int = 0
    n_cot_new_gold: int = 0
    n_merged_candidates: int = 0
    cot_retrieval_query: str = ""
    # Adaptive routing fields
    pre_beta_0: int = 0
    routing_tier: int = 0             # 1=skip CoT, 2=standard, 3=aggressive
    cot_skipped: bool = False
    # Unified pipeline (Dense + LLM rerank) fields
    n_dense_retrieved: int = 0
    n_dense_cot_retrieved: int = 0
    n_dense_new_gold: int = 0
    n_dense_graph_edges: int = 0      # Tier D edge count
    llm_rerank_applied: bool = False
    # Graph mode (Spec F: D+E integration)
    graph_mode: str = "sentence"
    # geDIG routing fields (Spec E)
    gedig_value: float = 0.0
    gedig_delta_betti_0: int = 0
    gedig_ig_value: float = 0.0
    gedig_ged_value: float = 0.0
    gedig_delta_sp_rel: float = 0.0
    gedig_computation_ms: float = 0.0
    n_doc_episodes: int = 0
    n_query_episodes: int = 0
    n_episode_cross_edges: int = 0
    # geDIG scoring fields (Spec H)
    scoring_mode: str = "classic"
    gedig_scoring_lambda: float = 0.0
    gedig_scoring_sp_beta: float = 0.0
    n_edges_discovered: int = 0
    n_edges_removed: int = 0
    mp_iterations_run: int = 0
    avg_gedig_local: float = 0.0
    # Pointwise reranking fields (Spec J)
    pointwise_rerank_applied: bool = False
    pointwise_rerank_n_scored: int = 0
    pointwise_rerank_n_calls: int = 0
    pointwise_rerank_ms: float = 0.0
    pointwise_rerank_avg_score: float = 0.0
    # Query decomposition fields (Spec K)
    query_decomp_applied: bool = False
    n_sub_queries: int = 0
    n_decomp_new_candidates: int = 0
    n_decomp_new_gold: int = 0
    query_decomp_ms: float = 0.0
    # Reasoning reranking fields (Spec L)
    reasoning_rerank_applied: bool = False
    reasoning_rerank_model: str = ""
    reasoning_rerank_n_scored: int = 0
    reasoning_rerank_n_calls: int = 0
    reasoning_rerank_ms: float = 0.0
    reasoning_rerank_avg_score: float = 0.0
    # RIA (Recursive Insight Architecture) fields (Spec M)
    ria_applied: bool = False
    ria_rounds: int = 0
    ria_beta0_history: list[int] = field(default_factory=list)
    ria_new_docs_per_round: list[int] = field(default_factory=list)
    ria_new_gold_per_round: list[int] = field(default_factory=list)
    ria_total_new_docs: int = 0
    ria_total_new_gold: int = 0
    ria_ms: float = 0.0

    # Token-level graph fields (Spec N)
    token_graph_applied: bool = False
    token_graph_avg_coverage: float = 0.0
    token_graph_avg_proximity: float = 0.0
    token_graph_avg_score: float = 0.0
    token_graph_n_docs: int = 0
    token_graph_ms: float = 0.0
    token_graph_spearman_bm25: float = 0.0
    token_graph_walk_score: bool = False
    token_graph_avg_beta1: float = 0.0
    token_graph_f_eval: bool = False
    token_graph_insight_mode: str = "none"
    token_graph_avg_n_insights: float = 0.0
    token_graph_avg_f_theta: float = 0.0

    # Entity graph F-eval fields (Spec O)
    entity_feval_applied: bool = False
    entity_feval_n_ag: int = 0
    entity_feval_n_dg: int = 0
    entity_feval_f_theta: float = 0.0
    entity_feval_avg_convergence: float = 0.0
    # Ranking DG/AG routing fields (Spec O.2)
    entity_feval_ranking_dg: float = 0.0
    entity_feval_adaptive_weight: float = 0.0
    # Multi-CoT Ensemble fields (Spec P)
    ensemble_applied: bool = False
    ensemble_n_cots: int = 0
    ensemble_cot_cache_hit: bool = False
    ensemble_ms: float = 0.0
    ensemble_n_ag_docs: int = 0
    ensemble_n_dg_docs: int = 0
    ensemble_avg_agreement: float = 0.0
    ensemble_score_variance_mean: float = 0.0


# ---------------------------------------------------------------------------
# CoT × Graph Pipeline
# ---------------------------------------------------------------------------

class BrightCoTPipeline:
    """Graph re-ranking with CoT dynamic augmentation.

    Parameters
    ----------
    model : str
        LLM model for CoT generation.
    initial_top_k : int
        Number of BM25 candidates.
    graph_top_k : int
        Docs used for entity graph construction.
    rerank_top_k : int
        Final output size.
    rerank_alpha : float
        BM25 weight (0=all graph, 1=all BM25).
    max_para_freq : int
        Discriminative entity filter.
    cot_weight : float
        Weight multiplier for CoT-connected nodes in graph scoring.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        initial_top_k: int = 100,
        graph_top_k: int = 50,
        rerank_top_k: int = 10,
        rerank_alpha: float = 0.1,
        max_para_freq: int = 5,
        cot_weight: float = 2.0,
        cot_retrieval_top_k: int = 50,
        cot_retrieval_max_concepts: int = 20,
        enable_cot_retrieval: bool = False,
        # Adaptive routing parameters
        enable_adaptive: bool = False,
        beta_low: int = 3,
        beta_high: int = 10,
        aggressive_top_k: int = 100,
        aggressive_max_concepts: int = 40,
        # Unified pipeline parameters
        dense_retriever=None,
        dense_domain: str = "",
        dense_top_k: int = 100,
        dense_cot_top_k: int = 50,
        dense_sim_threshold: float = 0.5,
        enable_llm_rerank: bool = False,
        llm_rerank_top_k: int = 20,
        # Graph mode (Spec F: D+E integration)
        graph_mode: str = "sentence",  # "sentence" | "episode"
        # geDIG routing parameters (Spec E)
        gedig_router=None,
        episode_index=None,
        episode_graph_builder=None,
        # geDIG scoring parameters (Spec H)
        scoring_mode: str = "classic",  # "classic" | "gedig"
        gedig_scoring_lambda: float = 1.0,
        gedig_scoring_sp_beta: float = 0.5,
        gedig_scoring_k_hop: int = 2,
        gedig_scoring_mp_iterations: int = 2,
        gedig_scoring_mp_alpha: float = 0.3,
        # Pointwise LLM reranking parameters (Spec J)
        enable_pointwise_rerank: bool = False,
        pointwise_rerank_top_k: int = 30,
        pointwise_batch_size: int = 5,
        pointwise_blend_weight: float = 0.4,  # pw weight in blend (0=ignore PW, 1=full replace)
        # Query decomposition parameters (Spec K)
        enable_query_decomp: bool = False,
        query_decomp_top_k: int = 50,  # BM25 top-k per sub-query
        query_decomp_max_sub: int = 5,  # max sub-questions
        # Reasoning reranking parameters (Spec L)
        enable_reasoning_rerank: bool = False,
        rerank_model: str = "",  # empty = same as main model
        reasoning_rerank_top_k: int = 20,
        reasoning_rerank_doc_chars: int = 4000,
        reasoning_rerank_blend_weight: float = 0.7,
        # RIA iterative expansion parameters (Spec M)
        enable_ria_loop: bool = False,
        ria_max_rounds: int = 3,
        ria_docs_per_round: int = 50,
        ria_feedback_top_k: int = 5,
        ria_beta0_target: int = 1,
        # Token-level graph parameters (Spec N)
        enable_token_graph: bool = False,
        token_graph_weight: float = 0.15,
        token_graph_max_tokens: int = 500,
        token_graph_walk_score: bool = False,
        token_graph_dg_penalty: float = 2.0,
        token_graph_f_eval: bool = False,
        token_graph_f_lambda: float = 1.0,
        token_graph_insight_mode: str = "none",
        # Entity graph F-eval parameters (Spec O)
        enable_entity_feval: bool = False,
        entity_feval_weight: float = 0.20,
        entity_feval_lambda: float = 1.0,
        # Multi-CoT Ensemble parameters (Spec P)
        n_cot_ensemble: int = 1,
        cot_cache_dir: str | None = None,
        cot_temperature: float = 0.7,
    ):
        self.llm = LLMAnswerer(model=model, temperature=0.0, max_tokens=300)
        self.initial_top_k = initial_top_k
        self.graph_top_k = graph_top_k
        self.rerank_top_k = rerank_top_k
        self.rerank_alpha = rerank_alpha
        self.max_para_freq = max_para_freq
        self.cot_weight = cot_weight
        self.cot_retrieval_top_k = cot_retrieval_top_k
        self.cot_retrieval_max_concepts = cot_retrieval_max_concepts
        self.enable_cot_retrieval = enable_cot_retrieval
        self.enable_adaptive = enable_adaptive
        self.beta_low = beta_low
        self.beta_high = beta_high
        self.aggressive_top_k = aggressive_top_k
        self.aggressive_max_concepts = aggressive_max_concepts
        # Unified pipeline
        self.dense_retriever = dense_retriever
        self.dense_domain = dense_domain
        self.dense_top_k = dense_top_k
        self.dense_cot_top_k = dense_cot_top_k
        self.dense_sim_threshold = dense_sim_threshold
        self.enable_llm_rerank = enable_llm_rerank
        self.llm_rerank_top_k = llm_rerank_top_k
        # Graph mode (Spec F)
        self.graph_mode = graph_mode
        # geDIG routing
        self.gedig_router = gedig_router
        self.episode_index = episode_index
        self.episode_graph_builder = episode_graph_builder
        # geDIG scoring (Spec H)
        self.scoring_mode = scoring_mode
        self.gedig_scoring_lambda = gedig_scoring_lambda
        self.gedig_scoring_sp_beta = gedig_scoring_sp_beta
        self.gedig_scoring_k_hop = gedig_scoring_k_hop
        self.gedig_scoring_mp_iterations = gedig_scoring_mp_iterations
        self.gedig_scoring_mp_alpha = gedig_scoring_mp_alpha
        # Pointwise reranking (Spec J)
        self.enable_pointwise_rerank = enable_pointwise_rerank
        self.pointwise_rerank_top_k = pointwise_rerank_top_k
        self.pointwise_batch_size = pointwise_batch_size
        self.pointwise_blend_weight = pointwise_blend_weight
        # Query decomposition (Spec K)
        self.enable_query_decomp = enable_query_decomp
        self.query_decomp_top_k = query_decomp_top_k
        self.query_decomp_max_sub = query_decomp_max_sub
        # Reasoning reranking (Spec L)
        self.enable_reasoning_rerank = enable_reasoning_rerank
        self.reasoning_rerank_top_k = reasoning_rerank_top_k
        self.reasoning_rerank_doc_chars = reasoning_rerank_doc_chars
        self.reasoning_rerank_blend_weight = reasoning_rerank_blend_weight
        self.rerank_model = rerank_model or model
        # RIA (Spec M)
        self.enable_ria_loop = enable_ria_loop
        self.ria_max_rounds = ria_max_rounds
        self.ria_docs_per_round = ria_docs_per_round
        self.ria_feedback_top_k = ria_feedback_top_k
        self.ria_beta0_target = ria_beta0_target
        # Token graph (Spec N)
        self.enable_token_graph = enable_token_graph
        self.token_graph_weight = token_graph_weight
        self.token_graph_max_tokens = token_graph_max_tokens
        self.token_graph_walk_score = token_graph_walk_score
        self.token_graph_dg_penalty = token_graph_dg_penalty
        self.token_graph_f_eval = token_graph_f_eval
        self.token_graph_f_lambda = token_graph_f_lambda
        self.token_graph_insight_mode = token_graph_insight_mode
        # Entity graph F-eval (Spec O)
        self.enable_entity_feval = enable_entity_feval
        self.entity_feval_weight = entity_feval_weight
        self.entity_feval_lambda = entity_feval_lambda
        # Multi-CoT Ensemble (Spec P)
        self.n_cot_ensemble = n_cot_ensemble
        self.cot_cache_dir = cot_cache_dir
        if n_cot_ensemble > 1:
            self.llm_ensemble = LLMAnswerer(
                model=model, temperature=cot_temperature, max_tokens=300
            )
        else:
            self.llm_ensemble = self.llm
        self._nlp = None  # lazy-load spaCy
        # Create separate LLM for reranking if model differs
        if self.rerank_model != model:
            self.rerank_llm = LLMAnswerer(
                model=self.rerank_model, temperature=0.0, max_tokens=500
            )
        else:
            self.rerank_llm = self.llm

    def _get_nlp(self):
        """Lazy-load spaCy model (only when --token-graph enabled)."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["ner"])
        return self._nlp

    def rerank(
        self,
        query: str,
        query_id: str,
        bm25_index: object,
        docs: list[dict],
        gold_ids: set[str] | None = None,
        excluded_ids: set[str] | None = None,
    ) -> BrightCoTResult:
        """Re-rank with CoT-augmented entity graph.

        When ``enable_cot_retrieval`` is True the phase order changes:
          BM25 → CoT → CoT Re-retrieval → merge → Graph → inject → score → combine
        Otherwise the original order is preserved for backward compatibility.
        """
        t0 = time.time()
        excluded = excluded_ids or set()

        # Tracking vars for CoT re-retrieval / adaptive diagnostics
        n_cot_retrieved = 0
        n_cot_new_gold = 0
        n_merged = 0
        ret_query = ""
        new_cands: list[tuple[int, float]] = []
        pre_beta_0 = 0
        routing_tier = 2  # default: standard (Tier 2)
        cot_skipped = False
        # Dense retrieval tracking
        n_dense_retrieved = 0
        n_dense_cot_retrieved = 0
        n_dense_new_gold = 0
        n_dense_graph_edges = 0
        llm_rerank_applied = False

        # ── Phase 0: Query Decomposition (Spec K) ─────────────────
        query_decomp_applied = False
        n_sub_queries = 0
        n_decomp_new_candidates = 0
        n_decomp_new_gold = 0
        query_decomp_ms = 0.0
        sub_queries: list[str] = []

        if self.enable_query_decomp:
            t_decomp = time.time()
            try:
                decomp_prompt = self._DECOMP_PROMPT.format(query=query[:500])
                decomp_resp = self.llm._llm_call_raw(decomp_prompt, max_tokens=300)
                sub_queries = [
                    line.strip().lstrip("0123456789.-) ")
                    for line in decomp_resp.strip().split("\n")
                    if line.strip() and len(line.strip()) > 10
                ]
                sub_queries = sub_queries[: self.query_decomp_max_sub]
                n_sub_queries = len(sub_queries)
                query_decomp_applied = bool(sub_queries)
            except Exception:
                pass
            query_decomp_ms = (time.time() - t_decomp) * 1000

        # ── Phase 1: BM25 retrieval ───────────────────────────────
        query_tokens = query.lower().split()
        bm25_scores = bm25_index.get_scores(query_tokens)

        scored = [
            (i, float(bm25_scores[i]))
            for i in range(len(docs))
            if docs[i]["id"] not in excluded
        ]
        scored.sort(key=lambda x: -x[1])
        top_candidates = scored[: self.initial_top_k]
        bm25_doc_ids = [docs[i]["id"] for i, _ in top_candidates]

        # ── Phase 1a: Sub-query retrieval (Spec K) ────────────────
        # Strategy: use max(original_bm25, best_sub_query_bm25) per candidate.
        # This gives sub-query candidates a fair BM25 score for Phase 6 ranking.
        decomp_best_bm25: dict[int, float] = {}  # doc_idx → best sub-query BM25
        if sub_queries:
            existing_ids = {docs[i]["id"] for i, _ in top_candidates}
            decomp_new_indices: set[int] = set()
            for sq in sub_queries:
                sq_tokens = sq.lower().split()
                if not sq_tokens:
                    continue
                sq_scores = bm25_index.get_scores(sq_tokens)
                sq_scored = [
                    (i, float(sq_scores[i]))
                    for i in range(len(docs))
                    if docs[i]["id"] not in excluded
                    and docs[i]["id"] not in existing_ids
                ]
                sq_scored.sort(key=lambda x: -x[1])
                sq_top = sq_scored[: self.query_decomp_top_k]
                for idx, sc in sq_top:
                    decomp_new_indices.add(idx)
                    existing_ids.add(docs[idx]["id"])
                    # Track best sub-query BM25 score per candidate
                    decomp_best_bm25[idx] = max(
                        decomp_best_bm25.get(idx, 0.0), sc
                    )
                # Also track best sub-query scores for EXISTING candidates
                for idx, _ in top_candidates:
                    sc = float(sq_scores[idx])
                    if sc > decomp_best_bm25.get(idx, 0.0):
                        decomp_best_bm25[idx] = sc

            # Use max(original_bm25, best_sub_query_bm25) for new candidates
            decomp_candidates = [
                (idx, max(float(bm25_scores[idx]), decomp_best_bm25.get(idx, 0.0)))
                for idx in decomp_new_indices
            ]
            decomp_candidates.sort(key=lambda x: -x[1])
            top_candidates += decomp_candidates

            # Also boost existing candidates if sub-query BM25 is higher
            top_candidates = [
                (idx, max(sc, decomp_best_bm25.get(idx, 0.0)))
                for idx, sc in top_candidates
            ]

            n_decomp_new_candidates = len(decomp_candidates)
            if gold_ids:
                # Count new gold from decomposition (not in original BM25 top-k)
                orig_ids = set(bm25_doc_ids)
                n_decomp_new_gold = sum(
                    1 for idx, _ in decomp_candidates
                    if docs[idx]["id"] in gold_ids and docs[idx]["id"] not in orig_ids
                )

        # ── Phase 1b: Dense retrieval (parallel pool expansion) ───
        id_to_idx = {docs[i]["id"]: i for i in range(len(docs))}
        dense_candidates: list[tuple[int, float]] = []  # tracked separately for graph slots
        if self.dense_retriever is not None:
            bm25_id_set = {docs[i]["id"] for i, _ in top_candidates}
            dense_results = self.dense_retriever.retrieve(
                query, self.dense_domain, top_k=self.dense_top_k,
                exclude_ids=excluded,
            )
            dense_candidates = [
                (id_to_idx[did], sim)
                for did, sim in dense_results
                if did not in bm25_id_set and did in id_to_idx
            ]
            # Append with bm25_score=0 to merged pool
            top_candidates = top_candidates + [(idx, 0.0) for idx, _ in dense_candidates]
            n_dense_retrieved = len(dense_candidates)

        # geDIG routing tracking
        gedig_value = 0.0
        gedig_delta_betti_0 = 0
        gedig_ig_value = 0.0
        gedig_ged_value = 0.0
        gedig_delta_sp_rel = 0.0
        gedig_computation_ms = 0.0
        n_doc_episodes = 0
        n_query_episodes = 0
        n_episode_cross_edges = 0

        # ── Phase 1.5: Routing decision ────────────────────────────
        # geDIG routing: use episode graph + geDIG value
        if self.gedig_router is not None and self.episode_index is not None:
            try:
                # Get document episodes for graph candidates
                graph_cand_ids = [
                    docs[i]["id"]
                    for i, _ in top_candidates[:self.graph_top_k]
                ]
                doc_episodes = self.episode_index.get_doc_episodes(
                    self.dense_domain, graph_cand_ids
                )
                # Get query episodes (pre-computed)
                query_episodes = self.episode_index.get_query_episodes(
                    self.dense_domain, query_id
                )
                if not query_episodes:
                    # Fallback: single episode from query text
                    from episode_graph import Episode
                    query_episodes = [
                        Episode(id=0, text=query[:500], type="question")
                    ]

                # Build episode graph
                doc_bm25 = {}
                for i, s in top_candidates[:self.graph_top_k]:
                    doc_bm25[docs[i]["id"]] = s

                eg_result = self.episode_graph_builder.build(
                    doc_episodes, query_episodes, query,
                    doc_bm25_scores=doc_bm25,
                )

                # geDIG routing decision
                routing_decision = self.gedig_router.compute_routing(eg_result)
                routing_tier = routing_decision.tier
                cot_skipped = (routing_tier == 1)

                # Store diagnostics
                gedig_value = routing_decision.gedig_value
                gedig_delta_betti_0 = routing_decision.delta_betti_0
                gedig_ig_value = routing_decision.ig_value
                gedig_ged_value = routing_decision.ged_value
                gedig_delta_sp_rel = routing_decision.delta_sp_rel
                gedig_computation_ms = routing_decision.computation_time_ms
                n_doc_episodes = eg_result.n_doc_episodes
                n_query_episodes = eg_result.n_query_episodes
                n_episode_cross_edges = eg_result.n_cross_doc_edges

            except Exception as e:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "geDIG routing failed: %s. Falling back to tier 2.", e
                )
                routing_tier = 2
                cot_skipped = False

        # Unified mode: force aggressive (Tier 3), skip adaptive routing
        # (only for unified mode which sets enable_cot_retrieval + dense together)
        elif self.dense_retriever is not None and self.gedig_router is None and self.enable_llm_rerank:
            routing_tier = 3
            cot_skipped = False
        elif self.enable_adaptive:
            pre_beta_0 = self._compute_pre_beta0(top_candidates, docs)
            if pre_beta_0 <= self.beta_low:
                routing_tier = 1   # skip CoT
            elif pre_beta_0 > self.beta_high:
                routing_tier = 3   # aggressive
            else:
                routing_tier = 2   # standard

        # ── Phase 2: CoT reasoning (conditional) ─────────────────
        cot_list: list[dict] = []
        cot_cache_hit = False

        if self.enable_adaptive and routing_tier == 1 and self.dense_retriever is None:
            # Tier 1: Skip CoT entirely (System 1 — graph-only)
            cot_text = ""
            all_cot_concepts: set[str] = set()
            cot_latency = 0.0
            cot_skipped = True
        else:
            # Tier 2/3 or non-adaptive: generate CoT
            t_cot = time.time()
            if self.n_cot_ensemble > 1:
                # ── Spec P: Multi-CoT Ensemble ──
                cot_list, cot_cache_hit = self._generate_or_load_cots(
                    query, query_id
                )
                # Union all concepts for broad re-retrieval
                all_cot_concepts: set[str] = set()
                for ci in cot_list:
                    all_cot_concepts |= ci["concepts"]
                # Use first CoT as representative text
                cot_text = cot_list[0]["text"]
            else:
                # N=1: original single-CoT path
                prompt = _COT_PROMPT.format(query=query[:500])
                try:
                    cot_text = self.llm._llm_call_raw(prompt, max_tokens=300)
                except Exception as e:
                    cot_text = f"[CoT error: {e}]"
                cot_entities = extract_entities(cot_text)
                cot_terms = _extract_lowercase_concepts(cot_text)
                all_cot_concepts = cot_entities | cot_terms
                cot_list = [{"text": cot_text, "concepts": all_cot_concepts,
                             "entities": cot_entities}]
                cot_cache_hit = False
            cot_latency = (time.time() - t_cot) * 1000

        # ── Phase 2.5: CoT Re-retrieval (adaptive intensity) ─────
        do_retrieval = (
            (self.enable_cot_retrieval or self.enable_adaptive)
            and not cot_skipped
        )

        if do_retrieval:
            existing_ids = {docs[i]["id"] for i, _ in top_candidates}
            # Tier 3 / Unified (aggressive) vs standard params
            if routing_tier == 3:
                eff_top_k = self.aggressive_top_k
                eff_max_concepts = self.aggressive_max_concepts
            else:
                eff_top_k = None   # use defaults
                eff_max_concepts = None
            new_cands, n_cot_new_gold, ret_query = self._cot_retrieval(
                all_cot_concepts, bm25_index, docs,
                existing_ids, excluded, gold_ids,
                top_k_override=eff_top_k,
                max_concepts_override=eff_max_concepts,
            )
            n_cot_retrieved = len(new_cands)
            # Merge: original top_candidates + new CoT-retrieved docs
            merged_candidates = top_candidates + new_cands

            # ── Phase 2.5b: Dense CoT retrieval ──────────────────
            if self.dense_retriever is not None and not cot_skipped:
                existing_all = {docs[i]["id"] for i, _ in merged_candidates}
                dense_cot = self.dense_retriever.retrieve(
                    cot_text, self.dense_domain,
                    top_k=self.dense_cot_top_k, exclude_ids=existing_all,
                )
                dense_cot_new = [
                    (id_to_idx[did], 0.0)
                    for did, _ in dense_cot
                    if did not in existing_all and did in id_to_idx
                ]
                merged_candidates += dense_cot_new
                n_dense_cot_retrieved = len(dense_cot_new)
                if gold_ids:
                    n_dense_new_gold = sum(
                        1 for idx, _ in dense_cot_new
                        if docs[idx]["id"] in gold_ids
                    )

            n_merged = len(merged_candidates)
            # Graph slot allocation: BM25 + CoT + QD (dense candidates stay in pool only)
            if routing_tier == 3:
                cot_graph_slots = min(len(new_cands), self.graph_top_k // 3)
            else:
                cot_graph_slots = min(len(new_cands), self.graph_top_k // 5)
            # QD graph slots (Spec K): include top sub-query candidates in graph
            qd_graph_slots = 0
            qd_for_graph: list[tuple[int, float]] = []
            if query_decomp_applied and n_decomp_new_candidates > 0:
                qd_graph_slots = min(n_decomp_new_candidates, self.graph_top_k // 10)
                # QD candidates are at the end of top_candidates
                qd_cands = top_candidates[self.initial_top_k:]
                qd_cands_sorted = sorted(qd_cands, key=lambda x: -x[1])
                qd_for_graph = qd_cands_sorted[:qd_graph_slots]
            # Use BM25 candidates (first initial_top_k) for graph, not dense
            bm25_only = top_candidates[:self.initial_top_k]
            bm25_graph_slots = self.graph_top_k - cot_graph_slots - qd_graph_slots
            graph_candidates = (
                bm25_only[:bm25_graph_slots]
                + new_cands[:cot_graph_slots]
                + qd_for_graph
            )
        else:
            merged_candidates = top_candidates
            n_merged = len(top_candidates)
            graph_candidates = top_candidates[: self.graph_top_k]

        # ── Phase 2.6: RIA Iterative Expansion (Spec M) ──────────
        ria_applied = False
        ria_rounds = 0
        ria_beta0_history: list[int] = []
        ria_new_docs_per_round: list[int] = []
        ria_new_gold_per_round: list[int] = []
        ria_ms = 0.0

        if self.enable_ria_loop and do_retrieval and not cot_skipped:
            import time as _time
            ria_start = _time.time()
            ria_applied = True

            # Collect all existing doc IDs in pool
            pool_ids = {docs[i]["id"] for i, _ in merged_candidates}

            prev_beta0 = float("inf")

            for ria_round in range(self.ria_max_rounds):
                # 2.6a: Compute mini-graph β₀ from current pool
                # Sort by score so top-k captures the best candidates including RIA additions
                sorted_for_beta0 = sorted(merged_candidates, key=lambda x: -x[1])
                cur_beta0 = self._compute_pre_beta0(sorted_for_beta0, docs)
                ria_beta0_history.append(cur_beta0)

                # 2.6b: Check stopping conditions
                if cur_beta0 <= self.ria_beta0_target:
                    break  # convergence
                if ria_round > 0 and cur_beta0 >= prev_beta0:
                    break  # β₀ not improving
                prev_beta0 = cur_beta0

                # 2.6c-d: LLM generates new search keywords from top docs
                feedback_docs = []
                sorted_pool = sorted(merged_candidates, key=lambda x: -x[1])
                for fidx, (doc_idx, _score) in enumerate(
                    sorted_pool[: self.ria_feedback_top_k]
                ):
                    doc = docs[doc_idx]
                    content = doc["content"][:2000]
                    feedback_docs.append(f"[Doc {fidx + 1}] {content}")

                new_keywords = self._ria_expand_query(
                    query, cot_text, feedback_docs, ria_round + 1
                )

                if not new_keywords:
                    break  # LLM produced no new keywords

                # 2.6e: BM25 re-retrieval with new keywords
                keyword_query = " ".join(new_keywords)
                keyword_tokens = keyword_query.lower().split()
                if not keyword_tokens:
                    break

                ria_scores = bm25_index.get_scores(keyword_tokens)
                new_scored = [
                    (i, float(ria_scores[i]))
                    for i in range(len(docs))
                    if docs[i]["id"] not in pool_ids
                    and docs[i]["id"] not in excluded
                ]
                new_scored.sort(key=lambda x: -x[1])
                round_new_cands = new_scored[: self.ria_docs_per_round]

                if not round_new_cands:
                    ria_new_docs_per_round.append(0)
                    ria_new_gold_per_round.append(0)
                    break  # no new docs found

                # 2.6f: Add to pool
                merged_candidates += round_new_cands
                for idx_r, _ in round_new_cands:
                    pool_ids.add(docs[idx_r]["id"])

                # Track gold hits
                round_gold = 0
                if gold_ids:
                    round_gold = sum(
                        1 for idx_r, _ in round_new_cands
                        if docs[idx_r]["id"] in gold_ids
                    )
                ria_new_docs_per_round.append(len(round_new_cands))
                ria_new_gold_per_round.append(round_gold)
                ria_rounds = ria_round + 1

            # Final β₀ after loop
            if ria_rounds > 0:
                sorted_final = sorted(merged_candidates, key=lambda x: -x[1])
                final_beta0 = self._compute_pre_beta0(sorted_final, docs)
                ria_beta0_history.append(final_beta0)

            ria_ms = (_time.time() - ria_start) * 1000

            # Recompute graph slot allocation with expanded pool
            ria_total_new = sum(ria_new_docs_per_round)
            if ria_total_new > 0:
                n_merged = len(merged_candidates)
                ria_graph_slots = min(ria_total_new, self.graph_top_k // 5)
                bm25_graph_slots = (
                    self.graph_top_k - cot_graph_slots - qd_graph_slots - ria_graph_slots
                )
                # RIA candidates are at the end of merged_candidates
                ria_cands = merged_candidates[n_merged - ria_total_new :]
                ria_cands_sorted = sorted(ria_cands, key=lambda x: -x[1])
                ria_for_graph = ria_cands_sorted[:ria_graph_slots]
                bm25_only = top_candidates[: self.initial_top_k]
                graph_candidates = (
                    bm25_only[:bm25_graph_slots]
                    + new_cands[:cot_graph_slots]
                    + qd_for_graph
                    + ria_for_graph
                )

        # ── Phase 3: Entity graph construction ────────────────────
        titles = []
        sentences_list = []
        doc_id_map = {}  # title -> doc_id

        for idx, (doc_idx, _) in enumerate(graph_candidates):
            doc = docs[doc_idx]
            title = f"doc_{idx}"
            doc_id_map[title] = doc["id"]

            content = doc["content"]

            # Spec F: use episode texts instead of sentence splitting
            if self.graph_mode == "episode" and self.episode_index is not None:
                doc_eps_list = self.episode_index.get_doc_episodes(
                    self.dense_domain, [doc["id"]]
                )
                ep_texts = [
                    ep.text for ep in doc_eps_list[0].episodes if ep.text
                ]
                if ep_texts:
                    sents = ep_texts[:30]
                else:
                    sents = _split_sentences(content, max_sentences=30)
                    if not sents:
                        sents = [content[:500]]
            # Spec G: hybrid = sentence split + episode texts
            elif self.graph_mode == "hybrid" and self.episode_index is not None:
                sents = _split_sentences(content, max_sentences=30)
                if not sents:
                    sents = [content[:500]]
                # Augment with episode texts
                doc_eps_list = self.episode_index.get_doc_episodes(
                    self.dense_domain, [doc["id"]]
                )
                ep_texts = [
                    ep.text for ep in doc_eps_list[0].episodes if ep.text
                ]
                # Cap total nodes per doc to avoid explosion
                max_total = 30
                if len(sents) + len(ep_texts) > max_total:
                    ep_texts = ep_texts[:max(max_total - len(sents), 3)]
                sents = sents + ep_texts
            else:
                sents = _split_sentences(content, max_sentences=30)
                if not sents:
                    sents = [content[:500]]

            titles.append(title)
            sentences_list.append(sents)

        # Prepare Tier D embeddings only for unified mode (LLM rerank)
        # For cot_retrieval + dense: Tier D edges hurt precision (see Spec I report)
        doc_embeddings = None
        if self.dense_retriever is not None and self.enable_llm_rerank:
            doc_id_list = [doc_id_map[t] for t in titles]
            doc_embeddings_raw = self.dense_retriever.get_doc_embeddings(
                self.dense_domain, doc_id_list
            )
            # Map titles (doc_0, doc_1, ...) → embedding
            doc_embeddings = {}
            for t in titles:
                did = doc_id_map[t]
                if did in doc_embeddings_raw:
                    doc_embeddings[t] = doc_embeddings_raw[did]

        graph = build_sentence_graph(
            titles, sentences_list, max_para_freq=self.max_para_freq,
            doc_embeddings=doc_embeddings,
            dense_sim_threshold=self.dense_sim_threshold,
        )
        n_dense_graph_edges = graph.graph.get("n_tier_d_edges", 0)

        # ── Phase 4: Inject CoT into graph (skip if Tier 1) ──────
        if not cot_skipped:
            n_injected, n_edges = self._inject_cot_nodes(
                graph, cot_text, all_cot_concepts, titles, sentences_list
            )
        else:
            n_injected, n_edges = 0, 0

        # Topology
        n_nodes = graph.number_of_nodes()
        n_edges_total = graph.number_of_edges()
        if n_nodes > 0:
            components = list(nx.connected_components(graph))
            beta_0 = len(components)
            beta_1 = n_edges_total - n_nodes + beta_0
        else:
            beta_0, beta_1 = 0, 0

        # ── Phase 4.5: Per-document token graph scoring (Spec N) ──
        token_graph_scores: dict[str, float] = {}
        token_graph_applied = False
        tg_avg_coverage = 0.0
        tg_avg_proximity = 0.0
        tg_avg_score = 0.0
        tg_n_docs = 0
        tg_ms = 0.0
        tg_spearman_bm25 = 0.0
        tg_avg_beta1 = 0.0
        tg_avg_n_insights = 0.0
        tg_avg_f_theta = 0.0

        if self.enable_token_graph:
            import time as _tg_time
            _tg_t0 = _tg_time.time()

            from token_graph import compute_token_scores_batch
            nlp = self._get_nlp()

            tg_texts: list[str] = []
            tg_ids: list[str] = []
            for idx, (doc_idx, _) in enumerate(graph_candidates):
                tg_texts.append(docs[doc_idx]["content"])
                title = f"doc_{idx}"
                tg_ids.append(doc_id_map.get(title, title))

            tg_scores_raw, tg_diags = compute_token_scores_batch(
                query, tg_texts, nlp,
                max_tokens=self.token_graph_max_tokens,
                use_walk_score=self.token_graph_walk_score,
                dg_penalty=self.token_graph_dg_penalty,
                use_f_eval=self.token_graph_f_eval,
                f_lambda=self.token_graph_f_lambda,
                insight_mode=self.token_graph_insight_mode,
            )

            # Min-max normalize to [0, 1]
            tg_max = max(tg_scores_raw) if tg_scores_raw else 1.0
            tg_min = min(tg_scores_raw) if tg_scores_raw else 0.0
            tg_range = tg_max - tg_min if tg_max > tg_min else 1.0
            for i, doc_id in enumerate(tg_ids):
                token_graph_scores[doc_id] = (tg_scores_raw[i] - tg_min) / tg_range

            token_graph_applied = True
            tg_n_docs = len(tg_ids)
            coverages = [d["coverage"] for d in tg_diags]
            proximities = [d["proximity_bonus"] for d in tg_diags]
            tg_avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
            tg_avg_proximity = sum(proximities) / len(proximities) if proximities else 0.0
            tg_avg_score = sum(tg_scores_raw) / len(tg_scores_raw) if tg_scores_raw else 0.0
            tg_avg_beta1 = sum(d.get("beta_1", 0) for d in tg_diags) / len(tg_diags) if tg_diags else 0.0
            tg_avg_n_insights = sum(d.get("n_insights", 0) for d in tg_diags) / len(tg_diags) if tg_diags else 0.0
            tg_avg_f_theta = sum(d.get("f_theta", 0) for d in tg_diags) / len(tg_diags) if tg_diags else 0.0

            # Spearman rho vs BM25 (redundancy check)
            if len(tg_ids) >= 5:
                try:
                    from scipy.stats import spearmanr
                    bm25_order = list(range(len(tg_ids)))
                    tg_order = [token_graph_scores[did] for did in tg_ids]
                    rho, _ = spearmanr(bm25_order, tg_order)
                    tg_spearman_bm25 = float(rho) if rho == rho else 0.0
                except ImportError:
                    pass  # scipy not available

            tg_ms = (_tg_time.time() - _tg_t0) * 1000

        # ── Phase 5: Graph scoring with CoT boost ────────────────
        n_edges_discovered = 0
        n_edges_removed = 0
        mp_iterations_run = 0
        avg_gedig_local = 0.0
        ensemble_applied = False
        ensemble_ms = 0.0
        ensemble_n_ag_docs = 0
        ensemble_n_dg_docs = 0
        ensemble_avg_agreement = 0.0
        ensemble_score_variance_mean = 0.0

        if self.n_cot_ensemble > 1 and len(cot_list) > 1 and not cot_skipped:
            # ── Spec P: Multi-CoT Ensemble scoring ──
            import numpy as _np
            t_ens = time.time()
            ensemble_applied = True

            per_cot_scores: list[dict[str, float]] = []

            for ci, cot_info in enumerate(cot_list):
                graph_i = graph.copy()
                self._inject_cot_nodes(
                    graph_i, cot_info["text"], cot_info["concepts"],
                    titles, sentences_list
                )

                if self.scoring_mode == "gedig":
                    scores_i, diag_i = self._compute_gedig_scores(
                        query, cot_info["text"], cot_info["concepts"],
                        graph_i, titles, sentences_list, doc_id_map
                    )
                elif self.scoring_mode == "gedig_refine":
                    scores_i, diag_i = self._compute_gedig_refine_scores(
                        query, cot_info["text"], cot_info["concepts"],
                        graph_i, titles, sentences_list, doc_id_map
                    )
                else:
                    scores_i = self._compute_graph_scores(
                        query, cot_info["text"], cot_info["concepts"],
                        graph_i, titles, sentences_list, doc_id_map
                    )
                    diag_i = {}

                per_cot_scores.append(scores_i)

                if ci == len(cot_list) - 1:
                    n_edges_discovered = diag_i.get("n_edges_discovered", 0)
                    n_edges_removed = diag_i.get("n_edges_removed", 0)
                    mp_iterations_run = diag_i.get("mp_iterations_run", 0)
                    avg_gedig_local = diag_i.get("avg_gedig_local", 0.0)

            # ── Ensemble aggregation ──
            all_doc_ids: set[str] = set()
            for s in per_cot_scores:
                all_doc_ids.update(s.keys())

            graph_scores: dict[str, float] = {}
            doc_agreements: dict[str, float] = {}

            variance_sum = 0.0
            for doc_id in all_doc_ids:
                arr = _np.array([s.get(doc_id, 0.0) for s in per_cot_scores])
                mean_s = float(_np.mean(arr))
                var_s = float(_np.var(arr))
                graph_scores[doc_id] = mean_s
                normalized_var = min(var_s / 0.25, 1.0)
                doc_agreements[doc_id] = 1.0 - normalized_var
                variance_sum += var_s

            # Re-normalize to [0, 1]
            if graph_scores:
                gs_max = max(graph_scores.values())
                gs_min = min(graph_scores.values())
                gs_range = gs_max - gs_min
                if gs_range > 1e-10:
                    graph_scores = {k: (v - gs_min) / gs_range
                                    for k, v in graph_scores.items()}

            # DG/AG classification
            for doc_id in all_doc_ids:
                agr = doc_agreements.get(doc_id, 0.0)
                if agr >= 0.8 and graph_scores.get(doc_id, 0) > 0.3:
                    ensemble_n_ag_docs += 1
                elif agr < 0.5:
                    ensemble_n_dg_docs += 1

            all_agr = list(doc_agreements.values())
            ensemble_avg_agreement = float(_np.mean(all_agr)) if all_agr else 0.0
            ensemble_score_variance_mean = variance_sum / len(all_doc_ids) if all_doc_ids else 0.0
            ensemble_ms = (time.time() - t_ens) * 1000

        elif self.scoring_mode == "gedig":
            graph_scores, gedig_diag = self._compute_gedig_scores(
                query, cot_text, all_cot_concepts, graph,
                titles, sentences_list, doc_id_map
            )
            n_edges_discovered = gedig_diag.get("n_edges_discovered", 0)
            n_edges_removed = gedig_diag.get("n_edges_removed", 0)
            mp_iterations_run = gedig_diag.get("mp_iterations_run", 0)
            avg_gedig_local = gedig_diag.get("avg_gedig_local", 0.0)
        elif self.scoring_mode == "gedig_refine":
            # Use geDIG to refine graph structure, then classic scoring
            graph_scores, gedig_diag = self._compute_gedig_refine_scores(
                query, cot_text, all_cot_concepts, graph,
                titles, sentences_list, doc_id_map
            )
            n_edges_discovered = gedig_diag.get("n_edges_discovered", 0)
            n_edges_removed = gedig_diag.get("n_edges_removed", 0)
            mp_iterations_run = gedig_diag.get("mp_iterations_run", 0)
            avg_gedig_local = gedig_diag.get("avg_gedig_local", 0.0)
        else:
            graph_scores = self._compute_graph_scores(
                query, cot_text, all_cot_concepts, graph,
                titles, sentences_list, doc_id_map
            )

        # ── Phase 5.25: Entity graph F-eval walk score (Spec O) ───
        entity_feval_scores = {}
        ef_diag: dict = {}
        if self.enable_entity_feval:
            from gedig_scoring import entity_graph_feval_scores
            entity_feval_scores, ef_diag = entity_graph_feval_scores(
                graph, query, cot_text, all_cot_concepts,
                titles, sentences_list, doc_id_map,
                f_lambda=self.entity_feval_lambda,
            )

        # ── Phase 5.5: Blend token graph + entity F-eval scores ──
        if self.enable_token_graph and token_graph_scores:
            w = self.token_graph_weight
            for doc_id in graph_scores:
                g = graph_scores[doc_id]
                t = token_graph_scores.get(doc_id, 0.0)
                graph_scores[doc_id] = (1.0 - w) * g + w * t

        # ── Ranking DG/AG routing for entity F-eval (Spec O.2) ──
        entity_feval_ranking_dg = 0.0
        entity_feval_adaptive_weight = 0.0
        if self.enable_entity_feval and entity_feval_scores:
            import math as _math

            # (a) Score dispersion DG: normalized entropy of top-10 graph_scores
            sorted_gs = sorted(graph_scores.values(), reverse=True)
            top_k_gs = sorted_gs[:10]
            sum_topk = sum(top_k_gs)
            if len(top_k_gs) >= 2 and sum_topk > 1e-12:
                probs = [v / sum_topk for v in top_k_gs]
                entropy = -sum(p * _math.log(p + 1e-10) for p in probs)
                max_entropy = _math.log(len(top_k_gs))
                score_dispersion_dg = entropy / max_entropy if max_entropy > 0 else 1.0
            else:
                score_dispersion_dg = 1.0  # all zero → maximum uncertainty

            # (b) Rank disagreement: 1 - Jaccard(base_top10, ef_top10)
            base_top10 = sorted(graph_scores, key=graph_scores.get, reverse=True)[:10]
            ef_top10 = sorted(entity_feval_scores, key=entity_feval_scores.get, reverse=True)[:10]
            overlap = len(set(base_top10) & set(ef_top10))
            rank_agreement = overlap / 10.0 if len(base_top10) >= 10 else (
                overlap / max(len(base_top10), 1)
            )
            rank_disagreement = 1.0 - rank_agreement

            # (c) Convergence signal: avg_convergence from entity F-eval diagnostics
            convergence_raw = ef_diag.get("avg_convergence", 0.0)
            convergence_signal = min(convergence_raw / 2.0, 1.0)  # normalize ~[0, 1]

            # 3-signal ranking DG (cf. main code 3-attention)
            entity_feval_ranking_dg = score_dispersion_dg * rank_disagreement * convergence_signal

            # Adaptive weight: base_weight × min(ranking_dg × max_factor, max_factor)
            max_factor = 2.5
            entity_feval_adaptive_weight = self.entity_feval_weight * min(
                entity_feval_ranking_dg * max_factor, max_factor
            )

            # Apply adaptive blend
            w = entity_feval_adaptive_weight
            if w > 1e-6:
                for doc_id in graph_scores:
                    g = graph_scores[doc_id]
                    ef = entity_feval_scores.get(doc_id, 0.0)
                    graph_scores[doc_id] = (1.0 - w) * g + w * ef

        # ── Phase 6: Combined ranking ────────────────────────────
        # Score all candidates in merged pool
        all_scored = merged_candidates
        bm25_max = max(s for _, s in all_scored) if all_scored else 1.0
        bm25_min = min(s for _, s in all_scored) if all_scored else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0

        combined: list[tuple[str, float]] = []
        for doc_idx, bm25_score in all_scored:
            doc_id = docs[doc_idx]["id"]
            bm25_norm = (bm25_score - bm25_min) / bm25_range
            g_score = graph_scores.get(doc_id, 0.0)
            final = self.rerank_alpha * bm25_norm + (1 - self.rerank_alpha) * g_score
            combined.append((doc_id, final))

        combined.sort(key=lambda x: -x[1])

        # ── Phase 7: LLM Rerank (optional) ───────────────────────
        if self.enable_llm_rerank and not cot_skipped:
            rerank_pool = combined[: self.llm_rerank_top_k]
            try:
                reranked = self._llm_rerank(
                    query, cot_text, rerank_pool, docs, id_to_idx
                )
                if reranked:
                    # Replace top portion with reranked results
                    reranked_set = {did for did, _ in reranked}
                    remaining = [
                        (did, s) for did, s in combined
                        if did not in reranked_set
                    ]
                    combined = reranked + remaining
                    llm_rerank_applied = True
            except Exception as e:
                # LLM rerank failure is non-fatal
                import logging
                logging.getLogger(__name__).warning("LLM rerank failed: %s", e)

        # ── Phase 7b: LLM Pointwise Rerank (Spec J) ──────────────
        pointwise_rerank_applied = False
        pointwise_n_scored = 0
        pointwise_n_calls = 0
        pointwise_ms = 0.0
        pointwise_avg_score = 0.0

        if self.enable_pointwise_rerank and not cot_skipped:
            t_pw = time.time()
            pw_pool = combined[: self.pointwise_rerank_top_k]
            try:
                pw_result = self._pointwise_rerank(
                    query, cot_text, pw_pool, docs, id_to_idx
                )
                if pw_result:
                    pw_scores, pw_n_calls_ = pw_result
                    pointwise_n_calls = pw_n_calls_
                    pointwise_n_scored = len(pw_scores)
                    if pw_scores:
                        pointwise_avg_score = sum(pw_scores.values()) / len(pw_scores)

                    # ---- Normalize PW scores within scored pool (min-max) ----
                    if len(pw_scores) >= 2:
                        pw_vals = list(pw_scores.values())
                        pw_min, pw_max = min(pw_vals), max(pw_vals)
                        pw_range = pw_max - pw_min
                        if pw_range > 1e-9:
                            pw_scores = {
                                did: (s - pw_min) / pw_range
                                for did, s in pw_scores.items()
                            }
                        # else: all same score, normalization has no effect

                    # ---- Blend PW with original combined scores ----
                    w = self.pointwise_blend_weight  # PW weight
                    scored_set = set(pw_scores.keys())
                    # Build lookup for original combined scores
                    orig_scores = {did: s for did, s in combined}
                    blended = []
                    for did, _ in pw_pool:
                        if did in scored_set:
                            orig = orig_scores.get(did, 0.0)
                            pw = pw_scores[did]
                            blended.append((did, w * pw + (1 - w) * orig))
                    # Sort scored pool by blended score
                    blended.sort(key=lambda x: -x[1])

                    # ---- Cap unscored docs at min of scored pool ----
                    min_scored = min(s for _, s in blended) if blended else 0.0
                    remaining = []
                    for did, s in combined:
                        if did not in scored_set:
                            remaining.append((did, min(s, min_scored)))
                    remaining.sort(key=lambda x: -x[1])

                    combined = blended + remaining
                    pointwise_rerank_applied = True
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Pointwise rerank failed: %s", e
                )
            pointwise_ms = (time.time() - t_pw) * 1000

        # ── Phase 7c: Reasoning LLM Rerank (Spec L) ──────────────
        reasoning_rerank_applied = False
        reasoning_n_scored = 0
        reasoning_n_calls = 0
        reasoning_ms = 0.0
        reasoning_avg_score = 0.0

        if self.enable_reasoning_rerank and not cot_skipped:
            t_rr = time.time()
            rr_pool = combined[: self.reasoning_rerank_top_k]
            try:
                rr_result = self._reasoning_rerank(
                    query, cot_text, rr_pool, docs, id_to_idx
                )
                if rr_result:
                    rr_scores, rr_n_calls_ = rr_result
                    reasoning_n_calls = rr_n_calls_
                    reasoning_n_scored = len(rr_scores)
                    if rr_scores:
                        reasoning_avg_score = sum(rr_scores.values()) / len(rr_scores)

                    # ---- Normalize scores within scored pool (min-max) ----
                    if len(rr_scores) >= 2:
                        rr_vals = list(rr_scores.values())
                        rr_min, rr_max = min(rr_vals), max(rr_vals)
                        rr_range = rr_max - rr_min
                        if rr_range > 1e-9:
                            rr_scores = {
                                did: (s - rr_min) / rr_range
                                for did, s in rr_scores.items()
                            }

                    # ---- Blend reasoning scores with combined scores ----
                    w = self.reasoning_rerank_blend_weight
                    scored_set = set(rr_scores.keys())
                    orig_scores = {did: s for did, s in combined}
                    blended = []
                    for did, _ in rr_pool:
                        if did in scored_set:
                            orig = orig_scores.get(did, 0.0)
                            rr = rr_scores[did]
                            blended.append((did, w * rr + (1 - w) * orig))
                    blended.sort(key=lambda x: -x[1])

                    # ---- Cap unscored docs at min of scored pool ----
                    min_scored = min(s for _, s in blended) if blended else 0.0
                    remaining = []
                    for did, s in combined:
                        if did not in scored_set:
                            remaining.append((did, min(s, min_scored)))
                    remaining.sort(key=lambda x: -x[1])

                    combined = blended + remaining
                    reasoning_rerank_applied = True
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Reasoning rerank failed: %s", e
                )
            reasoning_ms = (time.time() - t_rr) * 1000

        ranked_ids = [doc_id for doc_id, _ in combined[: self.rerank_top_k]]
        ranked_scores = [score for _, score in combined[: self.rerank_top_k]]

        return BrightCoTResult(
            query_id=query_id,
            ranked_doc_ids=ranked_ids,
            ranked_scores=ranked_scores,
            bm25_doc_ids=bm25_doc_ids[: self.rerank_top_k],
            beta_0=beta_0,
            beta_1=beta_1,
            n_graph_nodes=n_nodes,
            n_graph_edges=n_edges_total,
            n_docs_in_graph=len(graph_candidates),
            latency_ms=(time.time() - t0) * 1000,
            cot_text=cot_text,
            cot_entities=sorted(all_cot_concepts),
            n_cot_nodes_injected=n_injected,
            n_cot_edges_created=n_edges,
            cot_latency_ms=cot_latency,
            n_cot_retrieved=n_cot_retrieved,
            n_cot_new_gold=n_cot_new_gold,
            n_merged_candidates=n_merged,
            cot_retrieval_query=ret_query,
            pre_beta_0=pre_beta_0,
            routing_tier=routing_tier,
            cot_skipped=cot_skipped,
            n_dense_retrieved=n_dense_retrieved,
            n_dense_cot_retrieved=n_dense_cot_retrieved,
            n_dense_new_gold=n_dense_new_gold,
            n_dense_graph_edges=n_dense_graph_edges,
            llm_rerank_applied=llm_rerank_applied,
            graph_mode=self.graph_mode,
            gedig_value=gedig_value,
            gedig_delta_betti_0=gedig_delta_betti_0,
            gedig_ig_value=gedig_ig_value,
            gedig_ged_value=gedig_ged_value,
            gedig_delta_sp_rel=gedig_delta_sp_rel,
            gedig_computation_ms=gedig_computation_ms,
            n_doc_episodes=n_doc_episodes,
            n_query_episodes=n_query_episodes,
            n_episode_cross_edges=n_episode_cross_edges,
            # geDIG scoring (Spec H)
            scoring_mode=self.scoring_mode,
            gedig_scoring_lambda=self.gedig_scoring_lambda,
            gedig_scoring_sp_beta=self.gedig_scoring_sp_beta,
            n_edges_discovered=n_edges_discovered,
            n_edges_removed=n_edges_removed,
            mp_iterations_run=mp_iterations_run,
            avg_gedig_local=avg_gedig_local,
            # Pointwise reranking (Spec J)
            pointwise_rerank_applied=pointwise_rerank_applied,
            pointwise_rerank_n_scored=pointwise_n_scored,
            pointwise_rerank_n_calls=pointwise_n_calls,
            pointwise_rerank_ms=pointwise_ms,
            pointwise_rerank_avg_score=pointwise_avg_score,
            # Query decomposition (Spec K)
            query_decomp_applied=query_decomp_applied,
            n_sub_queries=n_sub_queries,
            n_decomp_new_candidates=n_decomp_new_candidates,
            n_decomp_new_gold=n_decomp_new_gold,
            query_decomp_ms=query_decomp_ms,
            # Reasoning reranking (Spec L)
            reasoning_rerank_applied=reasoning_rerank_applied,
            reasoning_rerank_model=self.rerank_model if reasoning_rerank_applied else "",
            reasoning_rerank_n_scored=reasoning_n_scored,
            reasoning_rerank_n_calls=reasoning_n_calls,
            reasoning_rerank_ms=reasoning_ms,
            reasoning_rerank_avg_score=reasoning_avg_score,
            # RIA iterative expansion (Spec M)
            ria_applied=ria_applied,
            ria_rounds=ria_rounds,
            ria_beta0_history=ria_beta0_history,
            ria_new_docs_per_round=ria_new_docs_per_round,
            ria_new_gold_per_round=ria_new_gold_per_round,
            ria_total_new_docs=sum(ria_new_docs_per_round),
            ria_total_new_gold=sum(ria_new_gold_per_round),
            ria_ms=ria_ms,
            # Token graph (Spec N)
            token_graph_applied=token_graph_applied,
            token_graph_avg_coverage=tg_avg_coverage,
            token_graph_avg_proximity=tg_avg_proximity,
            token_graph_avg_score=tg_avg_score,
            token_graph_n_docs=tg_n_docs,
            token_graph_ms=tg_ms,
            token_graph_spearman_bm25=tg_spearman_bm25,
            token_graph_walk_score=self.token_graph_walk_score if self.enable_token_graph else False,
            token_graph_avg_beta1=tg_avg_beta1,
            token_graph_f_eval=self.token_graph_f_eval if self.enable_token_graph else False,
            token_graph_insight_mode=self.token_graph_insight_mode if self.enable_token_graph else "none",
            token_graph_avg_n_insights=tg_avg_n_insights,
            token_graph_avg_f_theta=tg_avg_f_theta,
            # Entity graph F-eval (Spec O)
            entity_feval_applied=self.enable_entity_feval and bool(entity_feval_scores),
            entity_feval_n_ag=ef_diag.get("n_ag", 0),
            entity_feval_n_dg=ef_diag.get("n_dg", 0),
            entity_feval_f_theta=ef_diag.get("f_theta", 0.0),
            entity_feval_avg_convergence=ef_diag.get("avg_convergence", 0.0),
            # Ranking DG/AG routing (Spec O.2)
            entity_feval_ranking_dg=entity_feval_ranking_dg,
            entity_feval_adaptive_weight=entity_feval_adaptive_weight,
            # Multi-CoT Ensemble (Spec P)
            ensemble_applied=ensemble_applied,
            ensemble_n_cots=len(cot_list) if ensemble_applied else 0,
            ensemble_cot_cache_hit=cot_cache_hit if ensemble_applied else False,
            ensemble_ms=ensemble_ms,
            ensemble_n_ag_docs=ensemble_n_ag_docs,
            ensemble_n_dg_docs=ensemble_n_dg_docs,
            ensemble_avg_agreement=ensemble_avg_agreement,
            ensemble_score_variance_mean=ensemble_score_variance_mean,
        )

    def _compute_pre_beta0(
        self,
        top_candidates: list[tuple[int, float]],
        docs: list[dict],
    ) -> int:
        """Compute pre-β₀ from BM25 top-k for adaptive routing (no LLM call).

        Builds a lightweight entity graph from BM25 top candidates and
        returns the number of connected components (β₀).
        """
        graph_cands = top_candidates[: self.graph_top_k]
        titles: list[str] = []
        sentences_list: list[list[str]] = []
        for idx, (doc_idx, _) in enumerate(graph_cands):
            doc = docs[doc_idx]
            sents = _split_sentences(doc["content"], max_sentences=20)
            if not sents:
                sents = [doc["content"][:500]]
            titles.append(f"pre_{idx}")
            sentences_list.append(sents)
        pre_graph = build_sentence_graph(
            titles, sentences_list, max_para_freq=self.max_para_freq
        )
        if pre_graph.number_of_nodes() > 0:
            return len(list(nx.connected_components(pre_graph)))
        return 0

    def _cot_retrieval(
        self,
        all_cot_concepts: set[str],
        bm25_index: object,
        docs: list[dict],
        existing_doc_ids: set[str],
        excluded_ids: set[str],
        gold_ids: set[str] | None = None,
        top_k_override: int | None = None,
        max_concepts_override: int | None = None,
    ) -> tuple[list[tuple[int, float]], int, str]:
        """Use CoT concepts as a BM25 query to retrieve docs outside top-100.

        Returns
        -------
        new_candidates : list[tuple[int, float]]
            (doc_index, bm25_score) pairs for newly retrieved docs.
        n_new_gold : int
            Number of gold documents among the new candidates.
        retrieval_query : str
            The query string used for re-retrieval.
        """
        eff_top_k = top_k_override if top_k_override is not None else self.cot_retrieval_top_k
        eff_max_concepts = max_concepts_override if max_concepts_override is not None else self.cot_retrieval_max_concepts

        # Sort concepts by length descending (longer = more discriminative)
        sorted_concepts = sorted(all_cot_concepts, key=len, reverse=True)
        top_concepts = sorted_concepts[: eff_max_concepts]

        retrieval_query = " ".join(top_concepts)
        query_tokens = retrieval_query.lower().split()

        if not query_tokens:
            return [], 0, ""

        cot_scores = bm25_index.get_scores(query_tokens)

        # Filter out existing candidates and excluded docs
        new_scored = [
            (i, float(cot_scores[i]))
            for i in range(len(docs))
            if docs[i]["id"] not in existing_doc_ids
            and docs[i]["id"] not in excluded_ids
        ]
        new_scored.sort(key=lambda x: -x[1])
        new_candidates = new_scored[: eff_top_k]

        # Count gold hits among new candidates
        n_new_gold = 0
        if gold_ids:
            n_new_gold = sum(
                1 for idx, _ in new_candidates if docs[idx]["id"] in gold_ids
            )

        return new_candidates, n_new_gold, retrieval_query

    def _ria_expand_query(
        self,
        query: str,
        cot_text: str,
        feedback_docs: list[str],
        round_num: int,
    ) -> list[str]:
        """Generate new search keywords from top retrieved documents (Spec M).

        Uses LLM to identify information gaps and suggest new search terms
        based on the current retrieval results.

        Returns
        -------
        keywords : list[str]
            New search keywords/phrases (5-10 items).
        """
        docs_text = "\n\n".join(feedback_docs)
        prompt = (
            "You are a search expert analyzing retrieval results for a complex query.\n\n"
            f"Query: {query}\n\n"
            f"Previous reasoning: {cot_text[:1000]}\n\n"
            f"Top retrieved documents (round {round_num}):\n"
            f"{docs_text}\n\n"
            "Based on these documents, identify:\n"
            "1. What information gaps remain to fully answer the query?\n"
            "2. What new search terms, concepts, or entities should we look for?\n"
            "3. What related topics or domains might contain relevant documents?\n\n"
            "Output 5-10 new search keywords/phrases, one per line.\n"
            "Do NOT repeat the original query terms. Focus on NEW concepts found in the documents."
        )

        try:
            response = self.llm._llm_call_raw(prompt, max_tokens=300)
            # Parse: one keyword per line, filter empty
            lines = [
                line.strip().lstrip("0123456789.-) ")
                for line in response.strip().split("\n")
            ]
            keywords = [line for line in lines if line and len(line) > 2]
            return keywords[:10]
        except Exception:
            return []

    # ── Multi-CoT Ensemble (Spec P) ─────────────────────────────

    def _generate_or_load_cots(
        self,
        query: str,
        query_id: str,
    ) -> tuple[list[dict], bool]:
        """Generate N CoTs or load from cache.

        Returns (cot_list, cache_hit) where each item is
        {"text": str, "concepts": set[str], "entities": set[str]}.
        """
        import json as _json
        from pathlib import Path as _Path

        N = self.n_cot_ensemble
        cache_hit = False

        # ── Try cache ──
        if self.cot_cache_dir is not None:
            cache_dir = _Path(self.cot_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{query_id}.json"

            if cache_file.exists():
                with open(cache_file) as f:
                    cached = _json.load(f)
                if len(cached) >= N:
                    cot_list = []
                    for entry in cached[:N]:
                        ents = set(entry.get("entities", []))
                        terms = set(entry.get("terms", []))
                        cot_list.append({
                            "text": entry["text"],
                            "concepts": ents | terms,
                            "entities": ents,
                        })
                    return cot_list, True

        # ── Generate N CoTs ──
        prompt = _COT_PROMPT.format(query=query[:500])
        cot_list = []

        for i in range(N):
            try:
                text = self.llm_ensemble._llm_call_raw(prompt, max_tokens=300)
            except Exception as e:
                text = f"[CoT error {i}: {e}]"
            entities = extract_entities(text)
            terms = _extract_lowercase_concepts(text)
            cot_list.append({
                "text": text,
                "concepts": entities | terms,
                "entities": entities,
            })

        # ── Write cache ──
        if self.cot_cache_dir is not None:
            serializable = [
                {
                    "text": c["text"],
                    "entities": sorted(c["entities"]),
                    "terms": sorted(c["concepts"] - c["entities"]),
                }
                for c in cot_list
            ]
            cache_file = _Path(self.cot_cache_dir) / f"{query_id}.json"
            with open(cache_file, "w") as f:
                _json.dump(serializable, f, indent=2, ensure_ascii=False)

        return cot_list, False

    def _inject_cot_nodes(
        self,
        graph: nx.Graph,
        cot_text: str,
        cot_concepts: set[str],
        titles: list[str],
        sentences_list: list[list[str]],
    ) -> tuple[int, int]:
        """Inject CoT reasoning as virtual nodes into the entity graph.

        Each CoT sentence becomes a node (title="cot"). Edges are created
        to existing document nodes that share entities with the CoT.

        Returns (n_nodes_added, n_edges_added).
        """
        if not cot_concepts or graph.number_of_nodes() == 0:
            return 0, 0

        # Split CoT into sentences
        cot_sents = _split_sentences(cot_text, max_sentences=10)
        if not cot_sents:
            return 0, 0

        # Pre-compute entities per existing node
        node_entities: dict[int, set[str]] = {}
        for n in graph.nodes():
            text = graph.nodes[n].get("text", "")
            ents = extract_entities(text)
            # Also extract lowercase terms from node text
            terms = _extract_lowercase_concepts(text)
            node_entities[n] = ents | terms

        n_added = 0
        n_edges = 0
        base_id = max(graph.nodes()) + 1 if graph.nodes() else 0

        for si, sent in enumerate(cot_sents):
            node_id = base_id + si
            graph.add_node(
                node_id,
                para_idx=-1,
                sent_idx=si,
                title="cot",
                text=sent,
            )
            n_added += 1

            # Extract entities from this CoT sentence
            sent_ents = extract_entities(sent)
            sent_terms = _extract_lowercase_concepts(sent)
            sent_concepts = sent_ents | sent_terms

            # Connect to existing doc nodes with shared entities
            for existing_node, existing_ents in node_entities.items():
                overlap = sent_concepts & existing_ents
                if overlap:
                    # Edge weight proportional to overlap
                    overlap_ratio = len(overlap) / max(
                        len(sent_concepts), len(existing_ents), 1
                    )
                    weight = 0.5 + 0.5 * overlap_ratio  # range [0.5, 1.0]
                    cost = 0.15  # Between Tier 1 (0.05-0.10) and Tier 2 (0.20-0.50)

                    if not graph.has_edge(node_id, existing_node):
                        graph.add_edge(
                            node_id,
                            existing_node,
                            edge_type="cot_bridge",
                            cost=cost,
                            weight=weight,
                            strength=weight,
                        )
                        n_edges += 1

        return n_added, n_edges

    def _compute_graph_scores(
        self,
        query: str,
        cot_text: str,
        cot_concepts: set[str],
        graph: nx.Graph,
        titles: list[str],
        sentences_list: list[list[str]],
        doc_id_map: dict[str, str],
    ) -> dict[str, float]:
        """Compute graph scores with CoT entity boost.

        Documents connected to CoT nodes get a scoring bonus.
        """
        if graph.number_of_nodes() == 0:
            return {}

        # PageRank with CoT nodes
        try:
            pagerank = nx.pagerank(graph, weight="weight")
        except Exception:
            pagerank = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes()}

        # Query + CoT entities (combined)
        query_entities = extract_entities(query)
        query_terms = _extract_lowercase_concepts(query)
        all_query_concepts = query_entities | query_terms | cot_concepts

        query_tokens = set(query.lower().split())
        cot_tokens = set(cot_text.lower().split())
        all_tokens = query_tokens | cot_tokens

        # Identify CoT nodes
        cot_node_set = {
            n for n in graph.nodes()
            if graph.nodes[n].get("title") == "cot"
        }

        doc_scores: dict[str, float] = {}

        for title_idx, title in enumerate(titles):
            doc_id = doc_id_map.get(title, title)

            doc_nodes = [
                n for n in graph.nodes()
                if graph.nodes[n].get("title") == title
            ]

            if not doc_nodes:
                doc_scores[doc_id] = 0.0
                continue

            # 1. PageRank (now boosted by CoT connections)
            pr_sum = sum(pagerank.get(n, 0.0) for n in doc_nodes)
            pr_avg = pr_sum / len(doc_nodes)

            # 2. Entity overlap with query + CoT
            doc_entities = set()
            doc_text_tokens = set()
            for sent in sentences_list[title_idx]:
                doc_entities.update(extract_entities(sent))
                doc_entities.update(_extract_lowercase_concepts(sent))
                doc_text_tokens.update(sent.lower().split())

            entity_overlap = len(all_query_concepts & doc_entities)
            token_overlap = len(all_tokens & doc_text_tokens)

            entity_score = entity_overlap / max(len(all_query_concepts), 1)
            token_score = min(token_overlap / max(len(all_tokens), 1), 1.0)

            # 3. Degree centrality
            degree_sum = sum(graph.degree(n) for n in doc_nodes)
            degree_avg = degree_sum / len(doc_nodes)
            degree_norm = min(degree_avg / 10.0, 1.0)

            # 4. ★ CoT bridge bonus: docs connected to CoT nodes
            cot_connection_count = 0
            for dn in doc_nodes:
                for neighbor in graph.neighbors(dn):
                    if neighbor in cot_node_set:
                        cot_connection_count += 1

            cot_bonus = min(cot_connection_count / max(len(cot_node_set), 1), 1.0)

            # Combined score (5 components)
            score = (
                0.25 * pr_avg * graph.number_of_nodes()
                + 0.25 * entity_score
                + 0.15 * token_score
                + 0.10 * degree_norm
                + 0.25 * cot_bonus * self.cot_weight  # CoT boost
            )

            doc_scores[doc_id] = score

        # Normalize to [0, 1]
        if doc_scores:
            max_gs = max(doc_scores.values())
            if max_gs > 0:
                doc_scores = {k: v / max_gs for k, v in doc_scores.items()}

        return doc_scores

    def _compute_gedig_scores(
        self,
        query: str,
        cot_text: str,
        cot_concepts: set[str],
        graph: nx.Graph,
        titles: list[str],
        sentences_list: list[list[str]],
        doc_id_map: dict[str, str],
    ) -> tuple[dict[str, float], dict]:
        """geDIG-based document scoring (Spec H).

        Flow:
          1. Compute TF-IDF node features
          2. Inject query node
          3. Message passing (propagate query relevance)
          4. Edge reevaluation (discover/remove edges)
          5. Per-document local geDIG scoring

        Returns:
            (doc_scores, diagnostics) where doc_scores is {doc_id: score}
            normalized to [0, 1].
        """
        import numpy as np
        import logging
        from gedig_scoring import MessagePassingNX, EdgeReevaluatorNX, GeDIGDocScorer
        from entity_graph import compute_node_tfidf_features

        _log = logging.getLogger(__name__)

        # Edge case: tiny graph
        if graph.number_of_nodes() < 3:
            _log.debug("geDIG scoring: graph too small (%d nodes), "
                       "falling back to classic.", graph.number_of_nodes())
            scores = self._compute_graph_scores(
                query, cot_text, cot_concepts, graph,
                titles, sentences_list, doc_id_map,
            )
            return scores, {"fallback": True}

        # 1. TF-IDF features
        features_before, vectorizer = compute_node_tfidf_features(
            graph, max_features=500
        )

        # 2. Query vector in same TF-IDF space
        query_combined = query + " " + cot_text
        query_vec = vectorizer.transform([query_combined])[0].astype(np.float32)

        # Save graph state before modifications
        graph_before = graph.copy()
        features_before_copy = features_before.copy()

        # 3. Inject query node
        scorer = GeDIGDocScorer(
            lambda_weight=self.gedig_scoring_lambda,
            sp_beta=self.gedig_scoring_sp_beta,
            k_hop=self.gedig_scoring_k_hop,
        )
        graph_with_query, features_with_query, query_node_id = (
            scorer.inject_query_node(
                graph, features_before, query_vec, cot_concepts
            )
        )

        # 4. Message passing
        mp = MessagePassingNX(
            alpha=self.gedig_scoring_mp_alpha,
            iterations=self.gedig_scoring_mp_iterations,
        )
        features_after = mp.forward(
            graph_with_query, features_with_query, query_vec
        )

        # 5. Edge reevaluation (adaptive thresholds for TF-IDF features)
        # TF-IDF cosine sims are much lower than dense embeddings (mean~0.05
        # vs ~0.5), so we compute percentile-based thresholds from the data.
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        pairwise = _cos_sim(features_after)
        upper_tri = pairwise[np.triu_indices_from(pairwise, k=1)]
        if len(upper_tri) > 0:
            sim_thresh = float(np.percentile(upper_tri, 50))   # retain top 50%
            new_thresh = float(np.percentile(upper_tri, 95))   # discover top 5%
            # Ensure minimum thresholds
            sim_thresh = max(sim_thresh, 0.01)
            new_thresh = max(new_thresh, sim_thresh + 0.01)
        else:
            sim_thresh, new_thresh = 0.05, 0.15

        er = EdgeReevaluatorNX(
            similarity_threshold=sim_thresh,
            new_edge_threshold=new_thresh,
            max_new_edges_per_node=5,
        )
        graph_after, n_discovered, n_removed = er.reevaluate(
            graph_with_query, features_after, query_vec
        )

        _log.debug(
            "geDIG scoring: MP iters=%d, edges discovered=%d removed=%d "
            "(sim_thresh=%.4f new_thresh=%.4f)",
            self.gedig_scoring_mp_iterations, n_discovered, n_removed,
            sim_thresh, new_thresh,
        )

        # 6. Per-document scoring (geDIG: structural + message passing + SP)
        doc_scores, doc_diagnostics = scorer.score_documents(
            graph_before=graph_before,
            graph_after=graph_after,
            node_features_before=features_before_copy,
            node_features_after=features_after,
            query_vector=query_vec,
            titles=titles,
            doc_id_map=doc_id_map,
        )

        # 7. CoT bridge bonus — docs connected to CoT nodes get a boost.
        #    This is the most important signal from classic scoring (effective
        #    weight 0.50 of the 1.25 total).  geDIG's message passing propagates
        #    query relevance but doesn't explicitly track CoT connections.
        cot_node_set = {
            n for n in graph_after.nodes()
            if graph_after.nodes[n].get("title") == "cot"
        }
        cot_bridge_scores: dict[str, float] = {}
        for title in titles:
            doc_id = doc_id_map.get(title, title)
            doc_nodes = [
                n for n in graph_after.nodes()
                if graph_after.nodes[n].get("title") == title
            ]
            if not doc_nodes or not cot_node_set:
                cot_bridge_scores[doc_id] = 0.0
                continue
            cot_count = sum(
                1 for dn in doc_nodes
                for nb in graph_after.neighbors(dn)
                if nb in cot_node_set
            )
            cot_bridge_scores[doc_id] = min(
                cot_count / max(len(cot_node_set), 1), 1.0
            )

        # Blend geDIG structural scores with CoT bridge signal
        # w_struct: weight for geDIG structural score (mp + gedig + sp)
        # w_cot: weight for CoT bridge bonus
        w_struct = 0.55
        w_cot = 0.45

        blended: dict[str, float] = {}
        for doc_id in doc_scores:
            blended[doc_id] = (
                w_struct * doc_scores[doc_id]
                + w_cot * cot_bridge_scores.get(doc_id, 0.0) * self.cot_weight
            )

        # Normalize blended scores to [0, 1]
        if blended:
            max_b = max(blended.values())
            min_b = min(blended.values())
            rng_b = max_b - min_b
            if rng_b > 1e-10:
                blended = {k: (v - min_b) / rng_b for k, v in blended.items()}
            elif max_b > 0:
                blended = {k: v / max_b for k, v in blended.items()}

        # Compute average local geDIG for diagnostics
        local_gedigs = [
            d.get("local_gedig", 0.0) for d in doc_diagnostics.values()
        ]
        avg_local = float(np.mean(local_gedigs)) if local_gedigs else 0.0

        diagnostics = {
            "n_edges_discovered": n_discovered,
            "n_edges_removed": n_removed,
            "mp_iterations_run": self.gedig_scoring_mp_iterations,
            "avg_gedig_local": round(avg_local, 4),
        }

        return blended, diagnostics

    def _compute_gedig_refine_scores(
        self,
        query: str,
        cot_text: str,
        cot_concepts: set[str],
        graph: nx.Graph,
        titles: list[str],
        sentences_list: list[list[str]],
        doc_id_map: dict[str, str],
    ) -> tuple[dict[str, float], dict]:
        """geDIG graph refinement + classic scoring (Spec H gedig_refine).

        Uses geDIG message passing and edge reevaluation to REFINE the graph
        structure, then applies the classic 5-component scoring formula to the
        refined graph.  This is the "best of both worlds" approach:
        - geDIG organizes the graph (discovers new edges, removes weak ones)
        - Classic scoring (PageRank, entity, token, degree, CoT bridge) provides
          well-calibrated document scores on the improved graph.
        """
        import numpy as np
        import logging
        from gedig_scoring import MessagePassingNX, EdgeReevaluatorNX, GeDIGDocScorer
        from entity_graph import compute_node_tfidf_features

        _log = logging.getLogger(__name__)

        if graph.number_of_nodes() < 3:
            scores = self._compute_graph_scores(
                query, cot_text, cot_concepts, graph,
                titles, sentences_list, doc_id_map,
            )
            return scores, {"fallback": True}

        # 1. TF-IDF features
        features, vectorizer = compute_node_tfidf_features(graph, max_features=500)

        # 2. Query vector
        query_combined = query + " " + cot_text
        query_vec = vectorizer.transform([query_combined])[0].astype(np.float32)

        # 3. Inject query node (needed for message passing)
        scorer = GeDIGDocScorer(
            lambda_weight=self.gedig_scoring_lambda,
            sp_beta=self.gedig_scoring_sp_beta,
            k_hop=self.gedig_scoring_k_hop,
        )
        graph_q, features_q, _ = scorer.inject_query_node(
            graph, features, query_vec, cot_concepts
        )

        # 4. Message passing
        mp = MessagePassingNX(
            alpha=self.gedig_scoring_mp_alpha,
            iterations=self.gedig_scoring_mp_iterations,
        )
        features_after = mp.forward(graph_q, features_q, query_vec)

        # 5. Adaptive edge reevaluation
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        pairwise = _cos_sim(features_after)
        upper_tri = pairwise[np.triu_indices_from(pairwise, k=1)]
        if len(upper_tri) > 0:
            sim_thresh = max(float(np.percentile(upper_tri, 50)), 0.01)
            new_thresh = max(float(np.percentile(upper_tri, 95)), sim_thresh + 0.01)
        else:
            sim_thresh, new_thresh = 0.05, 0.15

        er = EdgeReevaluatorNX(
            similarity_threshold=sim_thresh,
            new_edge_threshold=new_thresh,
            max_new_edges_per_node=5,
        )
        refined_graph, n_discovered, n_removed = er.reevaluate(
            graph_q, features_after, query_vec
        )

        _log.debug(
            "geDIG refine: edges discovered=%d removed=%d",
            n_discovered, n_removed,
        )

        # 6. Apply classic 5-component scoring to the REFINED graph
        graph_scores = self._compute_graph_scores(
            query, cot_text, cot_concepts, refined_graph,
            titles, sentences_list, doc_id_map,
        )

        diagnostics = {
            "n_edges_discovered": n_discovered,
            "n_edges_removed": n_removed,
            "mp_iterations_run": self.gedig_scoring_mp_iterations,
            "avg_gedig_local": 0.0,
        }

        return graph_scores, diagnostics


    # ── LLM Pointwise Reranking (Spec J) ────────────────────────

    # ── Query Decomposition Prompt (Spec K) ─────────────────────
    _DECOMP_PROMPT = """You are a search expert. Break this complex query into specific sub-questions that would each help find relevant documents.

Query: {query}

Generate 3-5 focused sub-questions. Each should:
- Target a specific aspect, mechanism, or sub-topic
- Use precise technical terms that would appear in relevant documents
- Be answerable independently

Output ONLY the sub-questions, one per line (no numbering, no explanation):"""

    _POINTWISE_PROMPT = """You are a relevance expert. Rate how well each document helps answer the query.

IMPORTANT: This is a reasoning-intensive query. Documents may be indirectly relevant — they might provide background knowledge, supporting evidence, or address a sub-question needed to answer the main query. Consider INDIRECT relevance highly.

Query: {query}

Key reasoning needed:
{cot_text}

Scoring guide (use the FULL 0-10 range, aim for spread):
- 8-10: Directly answers the query or provides critical evidence
- 5-7: Provides useful background, related mechanisms, or partial answers
- 2-4: Loosely related topic, minimal useful information
- 0-1: Completely irrelevant

Documents:
{doc_list}

Output ONLY scores (one per line):
D1: <score>
D2: <score>
...
"""

    def _pointwise_rerank(
        self,
        query: str,
        cot_text: str,
        candidates: list[tuple[str, float]],
        docs: list[dict],
        id_to_idx: dict[str, int],
    ) -> tuple[dict[str, float], int] | None:
        """LLM pointwise relevance scoring (Spec J).

        Scores each candidate document individually using LLM reasoning.
        Returns (doc_id -> normalized_score, n_api_calls) or None on failure.
        """
        if not candidates:
            return None

        batch_size = self.pointwise_batch_size
        all_scores: dict[str, float] = {}
        n_calls = 0

        # Process in batches
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start: batch_start + batch_size]

            # Build document list with content snippets
            doc_parts = []
            for i, (doc_id, _score) in enumerate(batch):
                idx = id_to_idx.get(doc_id)
                if idx is not None:
                    content = docs[idx]["content"][:1500]
                else:
                    content = "[content unavailable]"
                doc_parts.append(f"D{i+1}: {content}")
            doc_list = "\n\n".join(doc_parts)

            prompt = self._POINTWISE_PROMPT.format(
                query=query[:500],
                cot_text=cot_text[:500],
                doc_list=doc_list,
            )

            try:
                response = self.llm._llm_call_raw(prompt, max_tokens=150)
                n_calls += 1

                # Parse "D1: 7\nD2: 3\n..." → scores
                for line in response.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Match patterns like "D1: 7", "D1:7", "D1 - 7", "D1: 7/10"
                    m = re.match(r"D(\d+)\s*[:\-]\s*(\d+(?:\.\d+)?)", line)
                    if m:
                        d_idx = int(m.group(1)) - 1  # 1-indexed → 0-indexed
                        score = float(m.group(2))
                        score = min(max(score, 0.0), 10.0)  # clamp 0-10
                        if 0 <= d_idx < len(batch):
                            doc_id = batch[d_idx][0]
                            all_scores[doc_id] = score / 10.0  # normalize to [0, 1]
            except Exception:
                # Individual batch failure is non-fatal; continue with next
                continue

        if not all_scores:
            return None

        return all_scores, n_calls

    # ── LLM Listwise Reranking ──────────────────────────────────

    _RERANK_PROMPT = """Given a query and reasoning, rank the following documents by relevance.
Return ONLY a comma-separated list of document numbers (most relevant first).
Do not include any other text.

Query: {query}
Reasoning: {cot_text}

Documents:
{doc_list}

Ranking:"""

    def _llm_rerank(
        self,
        query: str,
        cot_text: str,
        candidates: list[tuple[str, float]],
        docs: list[dict],
        id_to_idx: dict[str, int],
    ) -> list[tuple[str, float]] | None:
        """LLM listwise reranking of top candidates.

        Returns reranked list of (doc_id, score) or None on failure.
        """
        if not candidates:
            return None

        # Build document list for prompt
        doc_list_parts = []
        for i, (doc_id, score) in enumerate(candidates):
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                content = docs[idx]["content"][:200]
            else:
                content = "[content unavailable]"
            doc_list_parts.append(f"[{i+1}] {content}")
        doc_list = "\n".join(doc_list_parts)

        prompt = self._RERANK_PROMPT.format(
            query=query[:300],
            cot_text=cot_text[:300],
            doc_list=doc_list,
        )
        response = self.llm._llm_call_raw(prompt, max_tokens=100)

        # Parse "1, 5, 3, ..." → reordered list
        try:
            # Extract numbers from response
            numbers = re.findall(r"\d+", response)
            seen = set()
            reranked = []
            for n_str in numbers:
                n = int(n_str) - 1  # 1-indexed → 0-indexed
                if 0 <= n < len(candidates) and n not in seen:
                    seen.add(n)
                    # Assign descending scores for reranked items
                    rerank_score = 1.0 - len(reranked) * (1.0 / len(candidates))
                    reranked.append((candidates[n][0], rerank_score))

            if not reranked:
                return None

            # Append any candidates not mentioned by LLM (keep original order)
            for i, (doc_id, score) in enumerate(candidates):
                if i not in seen:
                    rerank_score = 1.0 - len(reranked) * (1.0 / len(candidates))
                    reranked.append((doc_id, rerank_score))

            return reranked
        except Exception:
            return None

    # ── Reasoning LLM Reranking (Spec L) ───────────────────────

    _REASONING_RERANK_PROMPT = """You are an expert relevance assessor for reasoning-intensive information retrieval.

Your task: Determine how relevant this document is for answering the query below.

IMPORTANT: This query requires REASONING — the document may not directly answer the query but could provide essential background knowledge, mechanisms, evidence, or sub-answers needed for the full answer. Consider both DIRECT and INDIRECT relevance.

Query: {query}

Reasoning context (chain-of-thought about what information is needed):
{cot_text}

Document:
{doc_content}

Instructions:
1. Think step-by-step about how this document relates to the query.
2. Consider: Does it explain a mechanism? Provide evidence? Answer a sub-question? Give background needed to reason about the answer?
3. Assign a relevance score from 0 to 10.

Scoring guide:
- 9-10: Directly answers the query or provides critical evidence for the answer
- 7-8: Explains key mechanisms or provides strong supporting evidence
- 5-6: Provides useful background or addresses a related sub-question
- 3-4: Tangentially related, some useful context
- 1-2: Loosely related topic, minimal useful information
- 0: Completely irrelevant

Output your reasoning (1-3 sentences) followed by your score on the last line in this exact format:
SCORE: <number>"""

    def _reasoning_rerank(
        self,
        query: str,
        cot_text: str,
        candidates: list[tuple[str, float]],
        docs: list[dict],
        id_to_idx: dict[str, int],
    ) -> tuple[dict[str, float], int] | None:
        """LLM reasoning-trace relevance scoring (Spec L).

        Scores each candidate document individually using a stronger LLM
        with chain-of-thought reasoning before score assignment.
        One document per LLM call for maximum reasoning quality.

        Returns (doc_id -> normalized_score, n_api_calls) or None on failure.
        """
        if not candidates:
            return None

        all_scores: dict[str, float] = {}
        n_calls = 0
        doc_chars = self.reasoning_rerank_doc_chars

        for doc_id, _score in candidates:
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                content = docs[idx]["content"][:doc_chars]
            else:
                content = "[content unavailable]"

            prompt = self._REASONING_RERANK_PROMPT.format(
                query=query[:1000],
                cot_text=cot_text[:1000],
                doc_content=content,
            )

            try:
                response = self.rerank_llm._llm_call_raw(prompt, max_tokens=300)
                n_calls += 1

                # Parse "SCORE: 7" from the response
                score = self._parse_reasoning_score(response)
                if score is not None:
                    all_scores[doc_id] = score / 10.0  # normalize to [0, 1]
            except Exception:
                # Individual doc failure is non-fatal; continue
                continue

        if not all_scores:
            return None

        return all_scores, n_calls

    @staticmethod
    def _parse_reasoning_score(response: str) -> float | None:
        """Extract score from reasoning rerank response.

        Looks for "SCORE: <number>" pattern, falling back to last number
        in the response if the pattern isn't found.
        """
        # Primary: look for "SCORE: 7" or "SCORE: 7.5"
        m = re.search(r"SCORE\s*:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if m:
            score = float(m.group(1))
            return min(max(score, 0.0), 10.0)

        # Fallback: last number on its own line
        lines = response.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            m = re.match(r"^(\d+(?:\.\d+)?)\s*$", line)
            if m:
                score = float(m.group(1))
                if 0 <= score <= 10:
                    return score

        return None


# ---------------------------------------------------------------------------
# Concept extraction (lowercase terms from CoT)
# ---------------------------------------------------------------------------

# Scientific/technical terms often aren't capitalized in CoT output
_CONCEPT_RE = re.compile(
    r"\b([a-z][a-z]{2,}(?:\s+[a-z]{2,}){0,2})\b"
)

_STOP_CONCEPTS: set[str] = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "were", "been", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "can", "not", "but", "also", "such",
    "than", "then", "when", "where", "which", "what", "how", "why",
    "some", "any", "all", "each", "every", "both", "either", "neither",
    "more", "most", "less", "very", "just", "only", "even", "still",
    "about", "into", "over", "under", "after", "before", "between",
    "through", "during", "because", "since", "while", "although",
    "however", "therefore", "moreover", "furthermore", "additionally",
    "involves", "related", "relevant", "specific", "particular",
    "general", "important", "significant", "various", "different",
    "other", "another", "first", "second", "third", "step", "think",
    "query", "question", "answer", "document", "search", "find",
    "need", "help", "know", "understand", "provide", "include",
    "consider", "look", "make", "take", "give", "get", "use",
    "like", "way", "thing", "part", "point", "case", "well",
    "core", "asking", "concepts", "terms", "areas", "knowledge",
    "domain", "intermediate", "reasoning", "steps", "connect",
    "potential", "answers", "technical", "theories", "named",
    "entities", "involved",
}


def _extract_lowercase_concepts(text: str) -> set[str]:
    """Extract potential concept terms (2-3 word phrases) from text.

    Captures terms that aren't capitalized (common in CoT output)
    like 'phototaxis', 'proximate causation', 'natural selection'.
    """
    if not text:
        return set()

    # Single and multi-word terms
    words = text.lower().split()
    concepts = set()

    # Bigrams and trigrams
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            # Clean punctuation
            phrase = re.sub(r"[^a-z\s]", "", phrase).strip()
            if (
                phrase
                and len(phrase) > 4
                and phrase not in _STOP_CONCEPTS
                and all(w not in _STOP_CONCEPTS for w in phrase.split())
            ):
                concepts.add(phrase)

    return concepts
