"""v12: Open-World Topology-Guided Retrieval Pipeline for FRAMES.

β₀-driven iterative retrieval over Wikipedia:
  1. Initial retrieval: Question → entity extraction → Wikipedia search
  2. Graph construction: Retrieved articles → sentence-level entity graph
  3. Iterative bridging: β₀ > 1 → Component Gap Query → Wikipedia search
  4. Context construction: Subgraph-first ordering (v11 P2d insight)
  5. Answer generation: LLM with topology-optimized context

Key innovations:
  - β₀ as retrieval trigger (not just routing signal)
  - F-value convergence as stopping condition
  - Wikipedia API as open-world corpus
  - Adapts adapter.py v8 Component Gap Query for open-world setting

Usage::

    from wiki_retriever import WikipediaRetriever
    from open_world_pipeline import OpenWorldPipeline

    wiki = WikipediaRetriever()
    pipeline = OpenWorldPipeline(wiki_retriever=wiki)
    result = pipeline.run("Who was the 15th president?",
                          gold_titles=["James Buchanan"])
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import networkx as nx

from answerer import LLMAnswerer
from corpus_graph import CorpusGraphBuilder, SubgraphResult, RoutingDecision
from entity_graph import extract_entities
from wiki_retriever import WikipediaRetriever, WikiArticle, SearchResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalState:
    """State of the iterative retrieval process."""

    articles: list[WikiArticle] = field(default_factory=list)
    graph: nx.Graph | None = None
    beta_0: int = 0
    beta_1: int = 0
    f_value: float = 0.0
    iteration: int = 0
    search_queries: list[str] = field(default_factory=list)
    bridge_queries: list[str] = field(default_factory=list)
    convergence_reason: str = ""       # "beta0_1", "delta_f", "max_iter", "no_new"
    gold_recall: float | None = None
    gold_precision: float | None = None

    @property
    def n_articles(self) -> int:
        return len(self.articles)

    @property
    def titles(self) -> list[str]:
        return [a.title for a in self.articles]


@dataclass
class PipelineResult:
    """Complete result of the open-world pipeline."""

    answer: str
    retrieval_state: RetrievalState
    system_used: str                    # "system1" / "system2"
    n_llm_calls: int                    # bridge queries + answer
    latency_ms: float
    context_tokens_est: int
    subgraph_n_paras: int = 0
    subgraph_gold_precision: float = 0.0
    subgraph_gold_recall: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ANSWER_PROMPT = """\
Read ALL of the following paragraphs carefully, then answer the question.
The most relevant paragraphs are listed first.

{paragraphs}

Question: {question}

Think step by step. Focus especially on the first few paragraphs which are \
most likely to contain key information. Trace the reasoning chain, then give \
your final answer.

IMPORTANT: Your final answer must be a short phrase (a few words). \
Write it on the last line after "Answer: "."""

_BRIDGE_PROMPT = """\
Two groups of retrieved Wikipedia articles are disconnected — they share no \
common entities or concepts.

Group A ({n_a} articles about "{topic_a}"):
{facts_a}

Group B ({n_b} articles about "{topic_b}"):
{facts_b}

Original question: {question}

What Wikipedia article or concept would bridge Group A and Group B \
to help answer this question?
Write a short Wikipedia search query (3-10 words) to find the \
bridging article."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OpenWorldPipeline:
    """β₀-driven iterative retrieval pipeline for FRAMES.

    Parameters
    ----------
    wiki_retriever : WikipediaRetriever
        Wikipedia API retriever instance.
    initial_top_k : int
        Number of articles to retrieve in initial search.
    bridge_top_k : int
        Number of articles to retrieve per bridge query.
    max_iterations : int
        Maximum β₀ > 1 bridge iterations.
    k_hop : int
        Subgraph BFS radius (from v11 P2d: 3).
    max_subgraph_paras : int
        Maximum paragraphs in subgraph (from v11 P2d: 15).
    max_para_freq : int
        Discriminative entity filter.
    theta_f : float
        F-value routing threshold (999.0 = always System 2, v11 best).
    delta_f_epsilon : float
        F-value convergence threshold for stopping.
    model : str
        LLM model for bridge queries and answer generation.
    """

    def __init__(
        self,
        wiki_retriever: WikipediaRetriever,
        initial_top_k: int = 5,
        bridge_top_k: int = 3,
        max_iterations: int = 3,
        k_hop: int = 3,
        max_subgraph_paras: int = 15,
        max_para_freq: int = 3,
        theta_f: float = 999.0,
        delta_f_epsilon: float = 0.05,
        model: str = "gpt-4o",
    ):
        self.wiki_retriever = wiki_retriever
        self.initial_top_k = initial_top_k
        self.bridge_top_k = bridge_top_k
        self.max_iterations = max_iterations
        self.k_hop = k_hop
        self.max_subgraph_paras = max_subgraph_paras
        self.max_para_freq = max_para_freq
        self.theta_f = theta_f
        self.delta_f_epsilon = delta_f_epsilon

        self.answerer = LLMAnswerer(model=model, max_tokens=512)
        self.graph_builder = CorpusGraphBuilder(
            k_hop=k_hop,
            max_subgraph_paras=max_subgraph_paras,
            max_para_freq=max_para_freq,
        )

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def run(
        self,
        question: str,
        gold_titles: list[str] | None = None,
    ) -> PipelineResult:
        """Execute the full open-world retrieval pipeline.

        Parameters
        ----------
        question : str
            The question to answer.
        gold_titles : list[str] | None
            Gold-standard article titles (for evaluation only).

        Returns
        -------
        PipelineResult
            Answer and diagnostic information.
        """
        t0 = time.time()
        n_llm_calls = 0
        state = RetrievalState()

        try:
            # Phase 1: Initial retrieval
            state = self._initial_retrieval(question, state)

            if not state.articles:
                return PipelineResult(
                    answer="Unknown",
                    retrieval_state=state,
                    system_used="none",
                    n_llm_calls=0,
                    latency_ms=(time.time() - t0) * 1000,
                    context_tokens_est=0,
                    error="No articles retrieved",
                )

            # Phase 2: Build graph and analyze topology
            state = self._build_and_analyze(question, state, gold_titles)

            # Phase 3: Iterative bridge retrieval (if β₀ > 1)
            prev_f = state.f_value
            while (
                state.beta_0 > 1
                and state.iteration < self.max_iterations
            ):
                new_articles = self._bridge_retrieval(question, state)
                n_llm_calls += 1  # bridge query LLM call

                if not new_articles:
                    state.convergence_reason = "no_new"
                    break

                # Add new articles and rebuild graph
                state.articles.extend(new_articles)
                state.iteration += 1
                state = self._build_and_analyze(question, state, gold_titles)

                # Check F-value convergence
                delta_f = abs(state.f_value - prev_f)
                if delta_f < self.delta_f_epsilon:
                    state.convergence_reason = "delta_f"
                    break

                prev_f = state.f_value

            if not state.convergence_reason:
                if state.beta_0 <= 1:
                    state.convergence_reason = "beta0_1"
                else:
                    state.convergence_reason = "max_iter"

            # Phase 4: Construct context and generate answer
            titles = [a.title for a in state.articles]
            sentences_list = [a.sentences for a in state.articles]
            routing = self.graph_builder.route(
                question, titles, sentences_list,
                theta_f=self.theta_f, gold_titles=gold_titles,
            )

            prompt = self._build_prompt(
                question, titles, sentences_list, routing
            )
            answer = self._generate_answer(prompt)
            n_llm_calls += 1

            # Compute gold metrics on retrieved articles
            if gold_titles:
                retrieved_set = {t.lower() for t in titles}
                gold_set = {t.lower() for t in gold_titles}
                intersection = retrieved_set & gold_set
                state.gold_recall = (
                    len(intersection) / len(gold_set) if gold_set else 0.0
                )
                state.gold_precision = (
                    len(intersection) / len(retrieved_set)
                    if retrieved_set else 0.0
                )

            return PipelineResult(
                answer=answer,
                retrieval_state=state,
                system_used=routing.system,
                n_llm_calls=n_llm_calls,
                latency_ms=(time.time() - t0) * 1000,
                context_tokens_est=routing.context_tokens_est,
                subgraph_n_paras=len(routing.subgraph.paragraph_indices),
                subgraph_gold_precision=routing.subgraph.gold_precision,
                subgraph_gold_recall=routing.subgraph.gold_recall,
            )

        except Exception as e:
            return PipelineResult(
                answer="Unknown",
                retrieval_state=state,
                system_used="error",
                n_llm_calls=n_llm_calls,
                latency_ms=(time.time() - t0) * 1000,
                context_tokens_est=0,
                error=str(e),
            )

    # ------------------------------------------------------------------ #
    # Phase 1: Initial Retrieval
    # ------------------------------------------------------------------ #

    def _initial_retrieval(
        self, question: str, state: RetrievalState
    ) -> RetrievalState:
        """Retrieve initial articles from Wikipedia.

        Strategy:
          1. Extract entities from question
          2. Search with full question
          3. Search with entity-based queries
          4. Deduplicate and merge
        """
        queries: list[str] = []

        # Full question as search query
        queries.append(question[:200])

        # Entity-based queries
        entities = extract_entities(question)
        if entities:
            # Take top entities by length (longer = more specific)
            sorted_ents = sorted(entities, key=len, reverse=True)
            # Combine top 2-3 entities as a query
            if len(sorted_ents) >= 2:
                queries.append(" ".join(sorted_ents[:3]))
            # Also search individual long entities
            for ent in sorted_ents[:2]:
                if len(ent.split()) >= 2:  # multi-word entities
                    queries.append(ent)

        # Deduplicate queries
        seen: set[str] = set()
        unique_queries: list[str] = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen and q_lower:
                seen.add(q_lower)
                unique_queries.append(q)

        # Search Wikipedia
        result = self.wiki_retriever.multi_query_search(
            unique_queries,
            top_k_per_query=max(2, self.initial_top_k // len(unique_queries) + 1),
            max_total=self.initial_top_k,
        )

        state.articles = result.articles
        state.search_queries = result.queries_used
        return state

    # ------------------------------------------------------------------ #
    # Phase 2: Graph Construction and Topology Analysis
    # ------------------------------------------------------------------ #

    def _build_and_analyze(
        self,
        question: str,
        state: RetrievalState,
        gold_titles: list[str] | None = None,
    ) -> RetrievalState:
        """Build entity graph and compute topology metrics."""
        titles = [a.title for a in state.articles]
        sentences_list = [a.sentences for a in state.articles]

        # Build graph
        graph = self.graph_builder.build(titles, sentences_list)
        state.graph = graph

        # Extract subgraph for topology analysis
        subgraph_result = self.graph_builder.extract_subgraph(
            question, titles, sentences_list, gold_titles=gold_titles
        )

        # Compute F-value
        f_value, _ = self.graph_builder.compute_f_value(subgraph_result)

        state.beta_0 = subgraph_result.beta_0
        state.beta_1 = subgraph_result.beta_1
        state.f_value = f_value

        return state

    # ------------------------------------------------------------------ #
    # Phase 3: Bridge Retrieval (β₀ > 1)
    # ------------------------------------------------------------------ #

    def _bridge_retrieval(
        self,
        question: str,
        state: RetrievalState,
    ) -> list[WikiArticle]:
        """Generate bridge query and retrieve bridging articles.

        Adapted from adapter.py v8 _component_gap_retrieval().
        Key difference: retrieves from Wikipedia API instead of closed corpus.
        """
        if state.graph is None:
            return []

        # Get connected components
        components = list(nx.connected_components(state.graph))
        if len(components) <= 1:
            return []

        # Sort by size, get two largest
        components.sort(key=len, reverse=True)
        comp_a = components[0]
        comp_b = components[1]

        # Extract representative text from each component
        topic_a, facts_a = self._component_summary(state.graph, comp_a)
        topic_b, facts_b = self._component_summary(state.graph, comp_b)

        # Generate bridge query via LLM
        bridge_prompt = _BRIDGE_PROMPT.format(
            n_a=len(comp_a),
            topic_a=topic_a,
            facts_a=facts_a,
            n_b=len(comp_b),
            topic_b=topic_b,
            facts_b=facts_b,
            question=question,
        )

        try:
            bridge_query = self.answerer._llm_call_raw(
                bridge_prompt, max_tokens=50
            )
            # Clean: first line only, strip quotes
            bridge_query = (
                bridge_query.strip().split("\n")[0].strip('"').strip("'")
            )
        except Exception:
            bridge_query = f"{topic_a} {topic_b}"

        state.bridge_queries.append(bridge_query)

        # Search Wikipedia with bridge query
        existing_titles = {a.title.lower() for a in state.articles}
        result = self.wiki_retriever.search_and_fetch(
            bridge_query,
            top_k=self.bridge_top_k,
            exclude_titles=existing_titles,
        )

        # Also try a direct title combination query
        title_query = f"{topic_a} {topic_b}"
        if title_query.lower() != bridge_query.lower():
            title_result = self.wiki_retriever.search_and_fetch(
                title_query,
                top_k=2,
                exclude_titles=existing_titles | {
                    a.title.lower() for a in result.articles
                },
            )
            result.articles.extend(title_result.articles)

        return result.articles

    @staticmethod
    def _component_summary(
        graph: nx.Graph, comp_nodes: set
    ) -> tuple[str, str]:
        """Extract representative text from a connected component.

        Returns (main_title, facts_snippet).
        """
        titles: set[str] = set()
        texts: list[str] = []

        for node in sorted(comp_nodes):
            data = graph.nodes[node]
            title = data.get("para_title", "")
            text = data.get("text", "")
            if title:
                titles.add(title)
            if text:
                texts.append(text[:100])

        main_title = next(iter(titles)) if titles else "unknown"
        snippet = " | ".join(texts[:3])
        return main_title, snippet

    # ------------------------------------------------------------------ #
    # Phase 4: Context and Answer
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        question: str,
        titles: list[str],
        sentences_list: list[list[str]],
        routing: RoutingDecision,
    ) -> str:
        """Build LLM prompt with topology-optimized context ordering."""
        # Get paragraph ordering from routing
        para_indices = routing.context_paragraphs

        # Build paragraphs text
        parts: list[str] = []
        for idx in para_indices:
            if 0 <= idx < len(titles):
                title = titles[idx]
                sents = sentences_list[idx]
                text = " ".join(sents[:30])  # cap per-paragraph length
                parts.append(f"[{title}]\n{text}")

        paragraphs_text = "\n\n".join(parts)

        return _ANSWER_PROMPT.format(
            paragraphs=paragraphs_text,
            question=question,
        )

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer via LLM."""
        if LLMAnswerer.is_mock_enabled():
            return "Mock answer"

        raw = self.answerer._llm_call_raw(prompt, max_tokens=512)

        # Extract answer after "Answer:" marker
        if "Answer:" in raw:
            return raw.split("Answer:")[-1].strip()
        # Fallback: last line
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        return lines[-1] if lines else raw.strip()


# ---------------------------------------------------------------------------
# Gold-only mode (baseline)
# ---------------------------------------------------------------------------

class GoldOnlyPipeline:
    """Baseline: answer using gold articles directly (no retrieval).

    Uses the same LLM prompt as OpenWorldPipeline for fair comparison.
    """

    def __init__(
        self,
        k_hop: int = 3,
        max_subgraph_paras: int = 15,
        max_para_freq: int = 3,
        theta_f: float = 999.0,
        model: str = "gpt-4o",
    ):
        self.k_hop = k_hop
        self.max_subgraph_paras = max_subgraph_paras
        self.max_para_freq = max_para_freq
        self.theta_f = theta_f
        self.answerer = LLMAnswerer(model=model, max_tokens=512)
        self.graph_builder = CorpusGraphBuilder(
            k_hop=k_hop,
            max_subgraph_paras=max_subgraph_paras,
            max_para_freq=max_para_freq,
        )

    def run(
        self,
        question: str,
        titles: list[str],
        sentences_list: list[list[str]],
        gold_titles: list[str] | None = None,
    ) -> PipelineResult:
        """Answer using pre-provided context (no retrieval)."""
        t0 = time.time()
        state = RetrievalState()
        state.articles = [
            WikiArticle(title=t, sentences=s)
            for t, s in zip(titles, sentences_list)
        ]
        state.convergence_reason = "gold_only"

        try:
            # Build graph
            graph = self.graph_builder.build(titles, sentences_list)
            state.graph = graph

            # Route
            routing = self.graph_builder.route(
                question, titles, sentences_list,
                theta_f=self.theta_f, gold_titles=gold_titles,
            )

            state.beta_0 = routing.subgraph.beta_0
            state.beta_1 = routing.subgraph.beta_1
            state.f_value = routing.f_value

            # Build prompt
            para_indices = routing.context_paragraphs
            parts: list[str] = []
            for idx in para_indices:
                if 0 <= idx < len(titles):
                    title = titles[idx]
                    sents = sentences_list[idx]
                    text = " ".join(sents[:30])
                    parts.append(f"[{title}]\n{text}")
            paragraphs_text = "\n\n".join(parts)
            prompt = _ANSWER_PROMPT.format(
                paragraphs=paragraphs_text, question=question
            )

            # Generate answer
            if LLMAnswerer.is_mock_enabled():
                answer = "Mock answer"
            else:
                raw = self.answerer._llm_call_raw(prompt, max_tokens=512)
                if "Answer:" in raw:
                    answer = raw.split("Answer:")[-1].strip()
                else:
                    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
                    answer = lines[-1] if lines else raw.strip()

            if gold_titles:
                retrieved_set = {t.lower() for t in titles}
                gold_set = {t.lower() for t in gold_titles}
                intersection = retrieved_set & gold_set
                state.gold_recall = (
                    len(intersection) / len(gold_set) if gold_set else 0.0
                )
                state.gold_precision = (
                    len(intersection) / len(retrieved_set)
                    if retrieved_set else 0.0
                )

            return PipelineResult(
                answer=answer,
                retrieval_state=state,
                system_used=routing.system,
                n_llm_calls=1,
                latency_ms=(time.time() - t0) * 1000,
                context_tokens_est=routing.context_tokens_est,
                subgraph_n_paras=len(routing.subgraph.paragraph_indices),
                subgraph_gold_precision=routing.subgraph.gold_precision,
                subgraph_gold_recall=routing.subgraph.gold_recall,
            )

        except Exception as e:
            return PipelineResult(
                answer="Unknown",
                retrieval_state=state,
                system_used="error",
                n_llm_calls=0,
                latency_ms=(time.time() - t0) * 1000,
                context_tokens_est=0,
                error=str(e),
            )
