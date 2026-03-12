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

        # ── Phase 1b: Dense retrieval (parallel pool expansion) ───
        id_to_idx = {docs[i]["id"]: i for i in range(len(docs))}
        if self.dense_retriever is not None:
            bm25_id_set = {docs[i]["id"] for i, _ in top_candidates}
            dense_results = self.dense_retriever.retrieve(
                query, self.dense_domain, top_k=self.dense_top_k,
                exclude_ids=excluded,
            )
            dense_new = [
                (id_to_idx[did], 0.0)
                for did, _ in dense_results
                if did not in bm25_id_set and did in id_to_idx
            ]
            top_candidates = top_candidates + dense_new
            n_dense_retrieved = len(dense_new)

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
        elif self.dense_retriever is not None and self.gedig_router is None:
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
        if self.enable_adaptive and routing_tier == 1 and self.dense_retriever is None:
            # Tier 1: Skip CoT entirely (System 1 — graph-only)
            cot_text = ""
            all_cot_concepts: set[str] = set()
            cot_latency = 0.0
            cot_skipped = True
        else:
            # Tier 2/3 or non-adaptive: generate CoT
            t_cot = time.time()
            prompt = _COT_PROMPT.format(query=query[:500])
            try:
                cot_text = self.llm._llm_call_raw(prompt, max_tokens=300)
            except Exception as e:
                cot_text = f"[CoT error: {e}]"
            cot_latency = (time.time() - t_cot) * 1000
            cot_entities = extract_entities(cot_text)
            cot_terms = _extract_lowercase_concepts(cot_text)
            all_cot_concepts = cot_entities | cot_terms

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
            # Aggressive/Unified: more graph slots (1/3 vs 1/5)
            if routing_tier == 3:
                cot_graph_slots = min(len(new_cands), self.graph_top_k // 3)
            else:
                cot_graph_slots = min(len(new_cands), self.graph_top_k // 5)
            graph_candidates = (
                top_candidates[: self.graph_top_k - cot_graph_slots]
                + new_cands[:cot_graph_slots]
            )
        else:
            merged_candidates = top_candidates
            n_merged = len(top_candidates)
            graph_candidates = top_candidates[: self.graph_top_k]

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
            else:
                sents = _split_sentences(content, max_sentences=30)
                if not sents:
                    sents = [content[:500]]

            titles.append(title)
            sentences_list.append(sents)

        # Prepare Tier D embeddings if dense retriever is available
        doc_embeddings = None
        if self.dense_retriever is not None:
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

        # ── Phase 5: Graph scoring with CoT boost ────────────────
        graph_scores = self._compute_graph_scores(
            query, cot_text, all_cot_concepts, graph,
            titles, sentences_list, doc_id_map
        )

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
