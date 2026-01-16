"""geDIG adapter for HotpotQA benchmark."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

# Add InsightSpike source to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.algorithms.gating import decide_gates

if TYPE_CHECKING:
    from .data_loader import HotpotQAExample

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class GeDIGResult:
    """Result from geDIG processing."""

    answer: str
    retrieved_facts: list[tuple[str, int]]
    latency_ms: float
    gedig_score: float
    initial_ag_fired: bool
    initial_dg_fired: bool
    ag_fired: bool
    dg_fired: bool
    graph_edges: int
    metadata: dict


class GeDIGHotpotQAAdapter:
    """geDIG-based RAG system for HotpotQA.

    Uses geDIG gauge to decide:
    - AG (Attention Gate): Should we explore more? (g0 > theta_ag)
    - DG (Decision Gate): Is the answer ready? (b(t) <= theta_dg)
    """

    def __init__(
        self,
        lambda_weight: float = 1.0,
        gamma: float = 1.0,
        theta_ag: float = 0.4,
        theta_dg: float = 0.0,
        max_hops: int = 2,
        top_k: int = 5,
        llm_model: str = "gpt-4o-mini",
        max_expansions: int = 1,
        expansion_seeds: int = 2,
        tfidf_dim: int = 64,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 256,
    ):
        self.lambda_weight = lambda_weight
        self.gamma = gamma
        self.theta_ag = theta_ag
        self.theta_dg = theta_dg
        self.max_hops = max_hops
        self.top_k = top_k
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.max_expansions = max(0, int(max_expansions))
        self.expansion_seeds = max(0, int(expansion_seeds))
        self.tfidf_dim = max(0, int(tfidf_dim))

        # Initialize geDIG core
        self.gedig_core = GeDIGCore(
            lambda_weight=lambda_weight,
            node_cost=1.0,
            edge_cost=1.0,
            enable_multihop=True,
            max_hops=max_hops,
            spike_detection_mode="and",
            tau_s=0.15,
            tau_i=0.25,
        )

        # BM25 retrieval (closed-world, per example)
        self._bm25_cls = None
        self.client = None

        # Knowledge graph
        self.graph = nx.Graph()

    def setup(self, examples: list[HotpotQAExample]) -> None:
        """Validate dependencies."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank-bm25")

        self._bm25_cls = BM25Okapi

    def _get_llm_client(self):
        """Lazy-load OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Please install openai: pip install openai")

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.client = OpenAI(api_key=api_key)

        return self.client

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _tokenize_for_features(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _stable_hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little", signed=False)

    def _build_tfidf_stats(
        self, corpus: list[tuple[str, int, str]]
    ) -> tuple[dict[str, float], int]:
        if self.tfidf_dim <= 0:
            return {}, len(corpus)
        df = Counter()
        for title, _, text in corpus:
            tokens = self._tokenize_for_features(title) + self._tokenize_for_features(text)
            for tok in set(tokens):
                df[tok] += 1
        doc_count = max(len(corpus), 1)
        idf = {
            tok: math.log((doc_count + 1) / (freq + 1)) + 1.0
            for tok, freq in df.items()
        }
        return idf, doc_count

    def _tfidf_vector(
        self, tokens: list[str], idf_map: dict[str, float], doc_count: int
    ) -> list[float]:
        if self.tfidf_dim <= 0:
            return []
        if not tokens:
            return [0.0] * self.tfidf_dim
        total = len(tokens)
        counts = Counter(tokens)
        default_idf = math.log((doc_count + 1) / 2.0) + 1.0 if doc_count > 0 else 1.0
        vec = [0.0] * self.tfidf_dim
        for tok, cnt in counts.items():
            tf = cnt / total
            idf_val = idf_map.get(tok, default_idf)
            idx = self._stable_hash(tok) % self.tfidf_dim
            vec[idx] += tf * idf_val
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _length_norm(self, tokens: list[str], scale: int = 40) -> float:
        if not tokens:
            return 0.0
        return min(len(tokens) / float(scale), 1.0)

    def _make_question_vector(
        self, question_tokens: list[str], tfidf_vector: list[float] | None = None
    ) -> list[float]:
        base = [
            0.0,  # score_norm
            1.0,  # overlap proxy
            self._length_norm(question_tokens),
            1.0,  # title overlap proxy
            1.0,  # is_question
        ]
        if tfidf_vector:
            return base + tfidf_vector
        return base

    def _make_fact_vector(
        self,
        question_tokens: set[str],
        title: str,
        text: str,
        score: float,
        max_score: float,
        tfidf_vector: list[float] | None = None,
    ) -> list[float]:
        text_tokens = self._tokenize_for_features(text)
        title_tokens = self._tokenize_for_features(title)
        overlap = (
            len(set(text_tokens) & question_tokens) / max(len(question_tokens), 1)
            if question_tokens
            else 0.0
        )
        title_overlap = 1.0 if question_tokens and (set(title_tokens) & question_tokens) else 0.0
        score_norm = score / max_score if max_score > 0 else 0.0
        base = [
            score_norm,
            overlap,
            self._length_norm(text_tokens),
            title_overlap,
            0.0,
        ]
        if tfidf_vector:
            return base + tfidf_vector
        return base

    def _build_base_graph(
        self,
        question: str,
        question_tokens: list[str],
        question_tfidf: list[float] | None = None,
    ) -> nx.Graph:
        g = nx.Graph()
        g.add_node(
            "Q",
            type="question",
            text=question[:100],
            feature=self._make_question_vector(question_tokens, question_tfidf),
        )
        return g

    def _build_bm25_for_example(
        self, example: HotpotQAExample
    ) -> tuple[list[tuple[str, int, str]], "BM25Okapi"]:
        if self._bm25_cls is None:
            raise RuntimeError("BM25 class not initialized. Call setup() first.")

        corpus = example.get_all_sentences()
        tokenized_corpus = [self._tokenize(text) for _, _, text in corpus]
        return corpus, self._bm25_cls(tokenized_corpus)

    def _retrieve(
        self,
        query: str,
        corpus: list[tuple[str, int, str]],
        bm25: "BM25Okapi",
        top_k: int,
    ) -> list[tuple[str, int, str, float]]:
        """Retrieve top-k sentences using BM25 (closed-world per example)."""
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [
            (corpus[i][0], corpus[i][1], corpus[i][2], float(scores[i]))
            for i in top_indices
        ]

    def _merge_retrieved(
        self,
        base: list[tuple[str, int, str, float]],
        extra: list[tuple[str, int, str, float]],
    ) -> list[tuple[str, int, str, float]]:
        merged: dict[tuple[str, int], tuple[str, float]] = {}
        for title, sent_id, text, score in base + extra:
            key = (title, sent_id)
            if key not in merged or score > merged[key][1]:
                merged[key] = (text, score)
        merged_list = [
            (title, sent_id, text, score)
            for (title, sent_id), (text, score) in merged.items()
        ]
        merged_list.sort(key=lambda item: (-item[3], item[0], item[1]))
        return merged_list

    def _build_graph_from_facts(
        self,
        question: str,
        question_tokens: list[str],
        facts: list[tuple[str, int, str, float]],
        idf_map: dict[str, float] | None = None,
        doc_count: int | None = None,
        question_tfidf: list[float] | None = None,
    ) -> nx.Graph:
        """Build a knowledge graph from retrieved facts.

        Graph structure:
        - Central node: question
        - Fact nodes: each retrieved fact
        - Edges: question-fact (by relevance), fact-fact (by title proximity)
        """
        if idf_map is None or doc_count is None:
            corpus = [(title, sent_id, text) for title, sent_id, text, _ in facts]
            idf_map, doc_count = self._build_tfidf_stats(corpus)
        if question_tfidf is None:
            question_tfidf = self._tfidf_vector(question_tokens, idf_map, doc_count)

        g = self._build_base_graph(question, question_tokens, question_tfidf)
        question_token_set = set(question_tokens)
        max_score = max((f[3] for f in facts), default=1.0)

        # Add fact nodes and edges to question
        for i, (title, sent_id, text, score) in enumerate(facts):
            fact_tokens = self._tokenize_for_features(title) + self._tokenize_for_features(text)
            tfidf_vector = self._tfidf_vector(fact_tokens, idf_map, doc_count)
            node_id = f"F{i}"
            g.add_node(
                node_id,
                type="fact",
                title=title,
                sent_id=sent_id,
                text=text[:100],
                feature=self._make_fact_vector(
                    question_token_set, title, text, score, max_score, tfidf_vector
                ),
            )
            # Edge weight based on BM25 score (normalized)
            weight = score / max_score if max_score > 0 else 0.5
            g.add_edge("Q", node_id, weight=weight)

        # Add fact-to-fact edges (same title = related)
        for i, (title_i, _, _, _) in enumerate(facts):
            for j, (title_j, _, _, _) in enumerate(facts):
                if i < j and title_i == title_j:
                    g.add_edge(f"F{i}", f"F{j}", weight=0.8)

        return g

    def _calculate_gedig(
        self,
        g_prev: nx.Graph,
        g_now: nx.Graph,
        focal_nodes: set[str] | None = None,
        query_vector: list[float] | None = None,
    ) -> tuple[float, float, bool, bool]:
        """Calculate geDIG score and gate decisions.

        Returns:
            (gedig_score, gmin, ag_fired, dg_fired)
        """
        try:
            candidate_count = max(g_now.number_of_nodes() - 1, 1)
            result = self.gedig_core.calculate(
                g_prev=g_prev,
                g_now=g_now,
                focal_nodes=focal_nodes,
                l1_candidates=candidate_count,
                query_vector=query_vector,
            )

            hop_results = result.hop_results or {}
            if 0 in hop_results:
                g0 = hop_results[0].gedig
            else:
                g0 = result.gedig_value
            gmin = min(hr.gedig for hr in hop_results.values()) if hop_results else g0

            gates = decide_gates(g0, gmin, self.theta_ag, self.theta_dg)

            return g0, gmin, gates.ag, gates.dg

        except Exception as e:
            # Fallback: return neutral values
            return 0.0, 0.0, False, True

    def _build_expansion_queries(
        self, question: str, retrieved: list[tuple[str, int, str, float]]
    ) -> list[str]:
        if self.expansion_seeds <= 0:
            return [question]
        queries = []
        for title, _, text, _ in retrieved[: self.expansion_seeds]:
            snippet = text[:80]
            queries.append(f"{question} {title} {snippet}")
        return queries or [question]

    def score_example(self, example: HotpotQAExample) -> tuple[float, float]:
        """Compute g0/gmin for initial retrieval (no expansion)."""
        corpus, bm25 = self._build_bm25_for_example(example)
        idf_map, doc_count = self._build_tfidf_stats(corpus)
        retrieved = self._retrieve(example.question, corpus, bm25, self.top_k)
        question_tokens = self._tokenize_for_features(example.question)
        question_tfidf = self._tfidf_vector(question_tokens, idf_map, doc_count)
        query_vector = self._make_question_vector(question_tokens, question_tfidf)
        g_prev = self._build_base_graph(example.question, question_tokens, question_tfidf)
        g_now = self._build_graph_from_facts(
            example.question,
            question_tokens,
            retrieved,
            idf_map=idf_map,
            doc_count=doc_count,
            question_tfidf=question_tfidf,
        )
        g0, gmin, _, _ = self._calculate_gedig(
            g_prev, g_now, query_vector=query_vector
        )
        return g0, gmin

    def _generate_answer(self, question: str, context: list[str]) -> str:
        """Generate answer using LLM."""
        client = self._get_llm_client()

        context_str = "\n".join(f"- {sent}" for sent in context)
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_str}

Question: {question}

Answer (be concise, just give the answer):"""

        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )

        return response.choices[0].message.content.strip()

    def process(self, example: HotpotQAExample) -> GeDIGResult:
        """Process a single question using geDIG-guided retrieval.

        Algorithm:
        1. Initial retrieval
        2. Build graph
        3. Calculate geDIG (compare to empty graph)
        4. If AG fires: expand search (optional refinement)
        5. If DG fires: generate answer
        6. Evaluate geDIG for the integration decision
        """
        start_time = time.time()
        question_tokens = self._tokenize_for_features(example.question)
        corpus, bm25 = self._build_bm25_for_example(example)
        idf_map, doc_count = self._build_tfidf_stats(corpus)
        question_tfidf = self._tfidf_vector(question_tokens, idf_map, doc_count)
        query_vector = self._make_question_vector(question_tokens, question_tfidf)

        # Step 1: Initial retrieval (closed-world per example)
        retrieved = self._retrieve(example.question, corpus, bm25, self.top_k)

        # Step 2: Build initial graph (empty -> retrieved)
        g_prev = self._build_base_graph(example.question, question_tokens, question_tfidf)
        g_now = self._build_graph_from_facts(
            example.question,
            question_tokens,
            retrieved,
            idf_map=idf_map,
            doc_count=doc_count,
            question_tfidf=question_tfidf,
        )

        # Step 3: Calculate geDIG
        gedig_score, gmin, ag_fired, dg_fired = self._calculate_gedig(
            g_prev, g_now, query_vector=query_vector
        )
        initial_g0, initial_gmin = gedig_score, gmin
        initial_ag, initial_dg = ag_fired, dg_fired

        # Step 4: If AG fires and DG not yet, expand retrieval iteratively
        expansions = 0
        while ag_fired and not dg_fired and expansions < self.max_expansions and retrieved:
            expansions += 1
            retrieval_k = min(len(corpus), self.top_k * (expansions + 1))
            expanded: list[tuple[str, int, str, float]] = []
            for query in self._build_expansion_queries(example.question, retrieved):
                expanded.extend(self._retrieve(query, corpus, bm25, retrieval_k))
            before_count = len(retrieved)
            retrieved = self._merge_retrieved(retrieved, expanded)
            if len(retrieved) == before_count:
                break

            g_expanded = self._build_graph_from_facts(
                example.question,
                question_tokens,
                retrieved[: self.top_k * (expansions + 1)],
                idf_map=idf_map,
                doc_count=doc_count,
                question_tfidf=question_tfidf,
            )
            gedig_score, gmin, ag_fired, dg_fired = self._calculate_gedig(
                g_now, g_expanded, query_vector=query_vector
            )
            g_now = g_expanded

        # Step 5: Generate answer
        context_limit = min(len(retrieved), self.top_k * (expansions + 1))
        context_texts = [text for _, _, text, _ in retrieved[:context_limit]]
        answer = self._generate_answer(example.question, context_texts)

        # Extract retrieved facts for evaluation
        retrieved_facts = [
            (title, sent_id) for title, sent_id, _, _ in retrieved[:context_limit]
        ]

        latency_ms = (time.time() - start_time) * 1000

        return GeDIGResult(
            answer=answer,
            retrieved_facts=retrieved_facts,
            latency_ms=latency_ms,
            gedig_score=gedig_score,
            initial_ag_fired=initial_ag,
            initial_dg_fired=initial_dg,
            ag_fired=ag_fired,
            dg_fired=dg_fired,
            graph_edges=g_now.number_of_edges(),
            metadata={
                "model": self.llm_model,
                "top_k": self.top_k,
                "lambda": self.lambda_weight,
                "g0": gedig_score,
                "gmin": gmin,
                "initial_g0": initial_g0,
                "initial_gmin": initial_gmin,
                "initial_ag": initial_ag,
                "initial_dg": initial_dg,
                "expansions": expansions,
                "expansion_seeds": self.expansion_seeds,
            },
        )

    def reset(self) -> None:
        """Reset internal state."""
        self.graph = nx.Graph()
