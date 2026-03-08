"""geDIG v2/v3 adapter for HotpotQA — extended gauge with Betti numbers.

This adapter wraps the main-codebase ``GeDIGCore`` and adds the
extended F (gauge) formula:

    F = ΔEPC_norm − λ·(ΔH_norm + γ₁·Δβ₁ − γ₀·Δβ₀)

The ``structural_mode`` parameter selects which topological terms are
active, giving 5 experimental conditions:

| Condition | structural_mode | γ₀   | γ₁  |
|-----------|-----------------|------|-----|
| A         | sp              | 0    | 0   |
| B         | betti           | 0    | 1.0 |
| C         | betti_full      | 1.0  | 0   |
| D         | betti_full      | 1.0  | 1.0 |
| E         | betti_full      | *    | *   | (tuned)

v3 adds **Hybrid mode** (``hybrid_mode``):
    System 1 (DG fires) → single-call answer (fast, cheap)
    System 2 (DG not fires) → CoT reasoning with retrieval (accurate)
    Inspired by Dual Process Theory (Kahneman, 2011).
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import types as _types

import networkx as nx

# Ensure InsightSpike src is importable
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Stub torch_geometric if not installed (only graph_edit_distance uses it,
# and we don't need GED in this experiment)
if "torch_geometric" not in sys.modules:
    _tg = _types.ModuleType("torch_geometric")
    _tg_data = _types.ModuleType("torch_geometric.data")
    _tg_data.Data = type("Data", (), {})  # type: ignore[attr-defined]
    sys.modules["torch_geometric"] = _tg
    sys.modules["torch_geometric.data"] = _tg_data

from insightspike.algorithms.gedig_core import GeDIGCore  # noqa: E402
from insightspike.algorithms.gating import decide_gates    # noqa: E402

from .answerer import LLMAnswerer      # noqa: E402
from .graph_builder import (           # noqa: E402
    GraphBuildConfig,
    KnowledgeGraphBuilder,
)
from .retriever import BM25Retriever, RetrievedFact  # noqa: E402

if TYPE_CHECKING:
    from .data_loader import HotpotQAExample

_TOKEN_RE = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------------- #
# Result dataclass
# ---------------------------------------------------------------------- #


@dataclass
class GeDIGv2Result:
    """Result from geDIG v2 processing — includes Betti diagnostics."""

    answer: str
    retrieved_facts: list[tuple[str, int]]
    latency_ms: float

    # geDIG gauge values
    gedig_score: float          # raw geDIG value from GeDIGCore
    extended_f: float           # F with Betti correction

    # Gate decisions
    initial_ag_fired: bool
    initial_dg_fired: bool
    ag_fired: bool
    dg_fired: bool

    # Betti diagnostics
    betti_0_before: int = 0
    betti_0_after: int = 0
    delta_betti_0: int = 0
    betti_1_before: int = 0
    betti_1_after: int = 0
    delta_betti_1: int = 0

    # Graph stats
    graph_nodes: int = 0
    graph_edges: int = 0
    expansions: int = 0

    # Hybrid (v3) diagnostics
    system_used: str = "system1"   # "system1" or "system2"
    cot_steps: int = 0

    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Adapter
# ---------------------------------------------------------------------- #


class GeDIGv2Adapter:
    """geDIG v2 adapter for HotpotQA with extended Betti gauge.

    Parameters
    ----------
    structural_mode : str
        One of ``"sp"`` (Condition A — shortpath only, no Betti),
        ``"betti"`` (Condition B — β₁ only), or ``"betti_full"``
        (Conditions C/D/E — both β₀ and β₁, weights set by γ₀/γ₁).
    gamma_0 : float
        Weight for −Δβ₀ term (island merging bonus).
    gamma_1 : float
        Weight for +Δβ₁ term (cycle penalty).
    lambda_weight : float
        Overall trade-off weight λ in the gauge.
    theta_ag, theta_dg : float
        Thresholds for Attention Gate and Decision Gate.
    """

    def __init__(
        self,
        structural_mode: str = "betti_full",
        gamma_0: float = 1.0,
        gamma_1: float = 1.0,
        lambda_weight: float = 1.0,
        theta_ag: float = 0.4,
        theta_dg: float = 0.0,
        max_hops: int = 2,
        top_k: int = 5,
        max_expansions: int = 1,
        expansion_seeds: int = 2,
        tfidf_dim: int = 64,
        q_link_top_k: int = 3,
        entity_overlap_threshold: float = 0.3,
        # LLM params
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 256,
        llm_retry_max: int = 3,
        # Hybrid (v3) params
        hybrid_mode: bool = False,
        max_cot_steps: int = 2,
    ):
        self.structural_mode = structural_mode
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.lambda_weight = lambda_weight
        self.theta_ag = theta_ag
        self.theta_dg = theta_dg
        self.top_k = top_k
        self.max_expansions = max(0, int(max_expansions))
        self.expansion_seeds = max(0, int(expansion_seeds))
        self.hybrid_mode = hybrid_mode
        self.max_cot_steps = max_cot_steps

        # GeDIGCore (main codebase)
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

        # Retriever
        self.retriever = BM25Retriever()

        # Graph builder (v2: β₀-sensitive)
        self.graph_builder = KnowledgeGraphBuilder(
            GraphBuildConfig(
                q_link_top_k=q_link_top_k,
                tfidf_dim=tfidf_dim,
                entity_overlap_threshold=entity_overlap_threshold,
            )
        )

        # LLM answerer
        self.answerer = LLMAnswerer(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            retry_max=llm_retry_max,
        )

    # ------------------------------------------------------------------ #
    # Extended F formula
    # ------------------------------------------------------------------ #

    def _compute_extended_f(self, base_gedig: float, result) -> float:
        """Compute extended gauge F from raw geDIG value + Betti numbers.

        ``result`` is the ``GeDIGResult`` from ``GeDIGCore.calculate()``.

        Condition A (sp):         F = base_gedig  (no Betti correction)
        Condition B (betti):      F = base_gedig − λ·γ₁·Δβ₁
        Condition C/D/E (full):   F = base_gedig − λ·(γ₁·Δβ₁ − γ₀·Δβ₀)
        """
        if self.structural_mode == "sp":
            return base_gedig

        betti_correction = self.lambda_weight * (
            self.gamma_1 * result.delta_betti_1
            - self.gamma_0 * result.delta_betti_0
        )
        return base_gedig - betti_correction

    # ------------------------------------------------------------------ #
    # Gate helpers
    # ------------------------------------------------------------------ #

    def _calculate_gedig(
        self,
        g_prev: nx.Graph,
        g_now: nx.Graph,
        query_vector: list[float] | None = None,
    ) -> tuple[float, float, bool, bool, object]:
        """Compute geDIG gauge and gate decisions.

        Returns
        -------
        (extended_f, gmin, ag_fired, dg_fired, core_result)
        """
        try:
            candidate_count = max(g_now.number_of_nodes() - 1, 1)
            core_result = self.gedig_core.calculate(
                g_prev=g_prev,
                g_now=g_now,
                l1_candidates=candidate_count,
                query_vector=query_vector,
            )

            hop_results = core_result.hop_results or {}
            g0 = hop_results[0].gedig if 0 in hop_results else core_result.gedig_value
            gmin = min(hr.gedig for hr in hop_results.values()) if hop_results else g0

            extended_f = self._compute_extended_f(g0, core_result)

            gates = decide_gates(extended_f, gmin, self.theta_ag, self.theta_dg)
            return extended_f, gmin, gates.ag, gates.dg, core_result

        except Exception:
            return 0.0, 0.0, False, True, None

    # ------------------------------------------------------------------ #
    # Expansion query builder
    # ------------------------------------------------------------------ #

    def _build_expansion_queries(
        self,
        question: str,
        retrieved: list[RetrievedFact],
        question_type: str | None = None,
    ) -> list[str]:
        """Build expansion queries from seed facts (same logic as v1)."""
        max_queries = max(4, self.expansion_seeds * 4 + 4)
        seen: set[str] = set()
        queries: list[str] = []

        def add(q: str) -> None:
            if len(queries) >= max_queries:
                return
            cleaned = " ".join(q.split())
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            queries.append(cleaned)

        add(question)

        # Entity-based expansion for comparison questions
        if (question_type or "").lower() == "comparison" or " or " in question.lower():
            for match in re.findall(r'"([^"]+)"', question):
                add(match.strip())

        for fact in retrieved[: self.expansion_seeds]:
            title = fact.title.strip()
            snippet = fact.text.strip()[:80]
            if title:
                add(title)
                add(f"{question} {title}")
            if snippet and title:
                add(f"{title} {snippet}")

        return queries or [question]

    # ------------------------------------------------------------------ #
    # System 2: CoT fallback (v3 Hybrid)
    # ------------------------------------------------------------------ #

    _COT_PROMPT = """\
Given the following context paragraphs and question, continue the chain-of-thought reasoning.
Generate ONLY the next single reasoning sentence. Do NOT repeat previous reasoning.
When you have enough information to answer, write: "So the answer is: <answer>."

Context:
{context}

Question: {question}

Reasoning so far:
{cot_so_far}

Next reasoning sentence:"""

    _ANSWER_RE = re.compile(
        r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)", re.IGNORECASE
    )

    # Dedicated extraction prompt for System 2 final answer (v3.1)
    _EXTRACT_PROMPT = """\
Based on the context and reasoning below, answer the question.
Give ONLY the answer — the shortest possible entity name or phrase.
Do NOT explain. Do NOT add articles (a/an/the). Do NOT add periods.

Context:
{context}

Reasoning: {reasoning}

Question: {question}

Answer (shortest form, e.g., "Paris" not "The city of Paris"):"""

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """Post-process answer to remove common LLM decorations."""
        if not answer:
            return answer
        # Strip surrounding whitespace and quotes
        answer = answer.strip().strip('"').strip("'").strip()
        # Remove trailing period / sentence-enders
        answer = answer.rstrip(".!,;:")
        # Remove leading articles for short answers (< 5 words)
        words = answer.split()
        if len(words) <= 5 and words and words[0].lower() in ("a", "an", "the"):
            answer = " ".join(words[1:])
        return answer.strip()

    def _cot_fallback(
        self,
        question: str,
        retrieved: list[RetrievedFact],
        corpus: list,
        bm25: object,
    ) -> tuple[str, int]:
        """System 2: CoT reasoning with iterative BM25 retrieval.

        Returns (answer, cot_steps).
        """
        # Working set of collected sentences
        collected: dict[tuple[str, int], str] = {}
        for f in retrieved:
            collected[(f.title, f.sent_id)] = f.text

        cot_sentences: list[str] = []
        answer = None

        for step in range(self.max_cot_steps):
            # Build context
            context_str = "\n".join(
                f"[{title}] {text}" for (title, _), text in collected.items()
            )
            cot_so_far = " ".join(cot_sentences) if cot_sentences else "(none yet)"

            prompt = self._COT_PROMPT.format(
                context=context_str,
                question=question,
                cot_so_far=cot_so_far,
            )
            next_sentence = self.answerer._llm_call_raw(prompt)
            cot_sentences.append(next_sentence)

            # Check if answer found
            match = self._ANSWER_RE.search(next_sentence)
            if match:
                answer = self._clean_answer(match.group(1))
                break

            # Use the CoT sentence as a new BM25 query
            if len(collected) < 15:
                new_facts = self.retriever.retrieve(
                    next_sentence, corpus, bm25, self.top_k
                )
                for f in new_facts:
                    key = (f.title, f.sent_id)
                    if key not in collected and len(collected) < 15:
                        collected[key] = f.text

        # If no answer extracted from CoT, use dedicated extraction prompt
        if answer is None:
            context_str = "\n".join(
                f"- {text}" for text in collected.values()
            )
            reasoning_str = " ".join(cot_sentences) if cot_sentences else ""
            extract_prompt = self._EXTRACT_PROMPT.format(
                context=context_str,
                reasoning=reasoning_str,
                question=question,
            )
            answer = self.answerer._llm_call_raw(extract_prompt, max_tokens=50)

        # Clean answer: strip trailing periods, quotes, leading articles
        answer = self._clean_answer(answer)

        return answer, len(cot_sentences)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def process(self, example: "HotpotQAExample") -> GeDIGv2Result:
        """Process a single HotpotQA question using geDIG v2.

        Algorithm:
        1. BM25 retrieval
        2. Build β₀-sensitive graph
        3. Compute extended gauge F
        4. If AG fires: expand search
        5. Generate answer
        """
        start_time = time.time()

        # --- Setup ---
        corpus, bm25 = self.retriever.build_index(example)
        idf_map, doc_count = self.graph_builder.build_tfidf_stats(
            [RetrievedFact(t, s, x, 0.0) for t, s, x in corpus]
        )
        query_vector = self.graph_builder.get_query_vector(
            example.question, idf_map, doc_count
        )

        # --- Step 1: Initial retrieval ---
        retrieved = self.retriever.retrieve(
            example.question, corpus, bm25, self.top_k
        )

        # --- Step 2: Build initial graphs ---
        g_prev = self.graph_builder.build_empty_graph(
            example.question, idf_map, doc_count
        )
        g_now = self.graph_builder.build_graph(
            example.question, retrieved, idf_map, doc_count
        )

        # --- Step 3: Calculate extended gauge ---
        ext_f, gmin, ag_fired, dg_fired, core_result = self._calculate_gedig(
            g_prev, g_now, query_vector=query_vector
        )
        initial_ag, initial_dg = ag_fired, dg_fired

        # --- Step 4: Expansion loop ---
        expansions = 0
        while ag_fired and not dg_fired and expansions < self.max_expansions and retrieved:
            expansions += 1
            retrieval_k = min(len(corpus), self.top_k * (expansions + 1))
            expansion_queries = self._build_expansion_queries(
                example.question, retrieved, example.question_type
            )
            expanded = self.retriever.retrieve_multi_query(
                expansion_queries, corpus, bm25, retrieval_k
            )

            # Merge
            merged_map: dict[tuple[str, int], RetrievedFact] = {}
            for f in list(retrieved) + expanded:
                key = (f.title, f.sent_id)
                if key not in merged_map or f.score > merged_map[key].score:
                    merged_map[key] = f
            new_retrieved = sorted(merged_map.values(), key=lambda f: -f.score)

            if len(new_retrieved) == len(retrieved):
                break  # no new facts found

            retrieved = new_retrieved

            # Rebuild graph and re-evaluate
            g_expanded = self.graph_builder.build_graph(
                example.question,
                retrieved[: self.top_k * (expansions + 1)],
                idf_map,
                doc_count,
            )
            ext_f, gmin, ag_fired, dg_fired, core_result = self._calculate_gedig(
                g_now, g_expanded, query_vector=query_vector
            )
            g_now = g_expanded

        # --- Step 5: Generate answer (System 1 / System 2 switch) ---
        context_limit = min(len(retrieved), self.top_k * (expansions + 1))
        context_texts = [f.text for f in retrieved[:context_limit]]

        system_used = "system1"
        cot_steps = 0

        if self.hybrid_mode and not (dg_fired and not ag_fired):
            # System 2: DG did NOT fire alone → needs deeper reasoning
            system_used = "system2"
            answer, cot_steps = self._cot_fallback(
                example.question, retrieved[:context_limit], corpus, bm25
            )
        else:
            # System 1: DG fired (high confidence) → direct answer
            answer = self.answerer.generate(example.question, context_texts)

        # --- Collect Betti diagnostics ---
        b0_before = core_result.betti_0_before if core_result else 0
        b0_after = core_result.betti_0_after if core_result else 0
        db0 = core_result.delta_betti_0 if core_result else 0
        b1_before = core_result.betti_1_before if core_result else 0
        b1_after = core_result.betti_1_after if core_result else 0
        db1 = core_result.delta_betti_1 if core_result else 0

        retrieved_facts = [
            (f.title, f.sent_id) for f in retrieved[:context_limit]
        ]

        latency_ms = (time.time() - start_time) * 1000

        return GeDIGv2Result(
            answer=answer,
            retrieved_facts=retrieved_facts,
            latency_ms=latency_ms,
            gedig_score=core_result.gedig_value if core_result else 0.0,
            extended_f=ext_f,
            initial_ag_fired=initial_ag,
            initial_dg_fired=initial_dg,
            ag_fired=ag_fired,
            dg_fired=dg_fired,
            betti_0_before=b0_before,
            betti_0_after=b0_after,
            delta_betti_0=db0,
            betti_1_before=b1_before,
            betti_1_after=b1_after,
            delta_betti_1=db1,
            graph_nodes=g_now.number_of_nodes(),
            graph_edges=g_now.number_of_edges(),
            expansions=expansions,
            system_used=system_used,
            cot_steps=cot_steps,
            metadata={
                "structural_mode": self.structural_mode,
                "gamma_0": self.gamma_0,
                "gamma_1": self.gamma_1,
                "lambda": self.lambda_weight,
                "extended_f": ext_f,
                "gedig_raw": core_result.gedig_value if core_result else 0.0,
                "question_type": example.question_type,
                "hybrid_mode": self.hybrid_mode,
                "system_used": system_used,
                "cot_steps": cot_steps,
            },
        )

    def reset(self) -> None:
        """Reset internal state between examples."""
        pass
