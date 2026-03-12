"""v11: Pre-computed Corpus-level Entity Graph with F-value Routing.

Two-phase architecture:
  Offline (once per example, query-independent):
    1. Build sentence-level entity graph from ALL paragraphs
    2. Pre-compute node-level features: beta_0, beta_1, centrality
    3. Build para-to-nodes mapping

  Online (per query):
    1. Match query entities to graph nodes
    2. Extract k-hop subgraph from query-matched nodes
    3. Map subgraph nodes back to paragraph indices
    4. Compute F-value (subgraph topology) for routing
    5. Route: System 1 (subgraph only) / System 2 (full context, subgraph first)

Key difference from v10:
  - v10: per-query graph built from Q-relevant paragraphs
  - v11: corpus-level graph built BEFORE seeing the question
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from entity_graph import (
    extract_entities,
    build_sentence_graph,
)


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class SubgraphResult:
    """Result of subgraph extraction for a single query."""

    paragraph_indices: list[int]       # Indices of paragraphs in the subgraph
    paragraph_titles: list[str]        # Titles for readability
    n_subgraph_nodes: int              # Sentence-level nodes in subgraph
    n_subgraph_edges: int              # Edges in subgraph
    query_matched_nodes: int           # Number of seed nodes matched from query

    # Topology of the subgraph
    beta_0: int = 0                    # Connected components
    beta_1: int = 0                    # Cycle rank (E - V + C)

    # Subgraph quality vs gold (if gold is known)
    gold_precision: float = 0.0        # |subgraph ∩ gold| / |subgraph|
    gold_recall: float = 0.0           # |subgraph ∩ gold| / |gold|


@dataclass
class RoutingDecision:
    """F-value routing decision for a single query."""

    system: str                        # "system1" or "system2"
    f_value: float                     # Topology-based F gauge value
    subgraph: SubgraphResult           # Extracted subgraph info
    context_paragraphs: list[int]      # Paragraph indices to feed to LLM
    context_tokens_est: int            # Estimated token count

    # Diagnostic
    beta_0_subgraph: int = 0
    beta_1_subgraph: int = 0


# ------------------------------------------------------------------ #
# Corpus Graph Builder
# ------------------------------------------------------------------ #

class CorpusGraphBuilder:
    """Build a corpus-level entity graph and extract query-specific subgraphs.

    Parameters
    ----------
    max_para_freq : int
        Discriminative entity filtering threshold for edge construction.
        Entities appearing in more than this many paragraphs are ignored.
    k_hop : int
        Number of BFS hops for subgraph extraction from query-matched nodes.
    max_subgraph_paras : int
        Maximum number of paragraphs returned by subgraph extraction.
    """

    def __init__(
        self,
        max_para_freq: int = 3,
        k_hop: int = 2,
        max_subgraph_paras: int = 10,
    ):
        self.max_para_freq = max_para_freq
        self.k_hop = k_hop
        self.max_subgraph_paras = max_subgraph_paras

        self._graph: nx.Graph | None = None
        self._para_to_nodes: dict[int, list[int]] = {}
        self._node_to_para: dict[int, int] = {}
        self._titles: list[str] = []
        self._node_entities: dict[int, set[str]] = {}

    # ------------------------------------------------------------------ #
    # Offline: Build corpus graph
    # ------------------------------------------------------------------ #

    def build(
        self,
        titles: list[str],
        sentences_list: list[list[str]],
    ) -> nx.Graph:
        """Build corpus-level sentence graph from all paragraphs.

        Reuses entity_graph.build_sentence_graph() for three-tier edge
        construction (context, entity, similarity).

        Additionally pre-computes:
          - Node degree centrality
          - Per-node entity sets (for query matching)
          - Para-to-node and node-to-para mappings

        Parameters
        ----------
        titles : list[str]
            Paragraph titles.
        sentences_list : list[list[str]]
            Sentences per paragraph.

        Returns
        -------
        nx.Graph
            Sentence-level graph with three-tier edges.
        """
        self._titles = titles
        self._graph = build_sentence_graph(
            titles, sentences_list, self.max_para_freq,
        )

        # Build mappings
        self._para_to_nodes = defaultdict(list)
        self._node_to_para = {}
        for node in self._graph.nodes:
            p_idx = self._graph.nodes[node]["para_idx"]
            self._para_to_nodes[p_idx].append(node)
            self._node_to_para[node] = p_idx

        # Pre-compute entity sets per node
        self._node_entities = {}
        for node in self._graph.nodes:
            text = self._graph.nodes[node].get("text", "")
            title = self._graph.nodes[node].get("title", "")
            self._node_entities[node] = extract_entities(f"{title} {text}")

        # Pre-compute centrality
        if self._graph.number_of_nodes() > 0:
            centrality = nx.degree_centrality(self._graph)
            for node, cent in centrality.items():
                self._graph.nodes[node]["centrality"] = cent
        else:
            pass  # Empty graph edge case

        # Store global topology
        n_components = nx.number_connected_components(self._graph)
        self._graph.graph["corpus_beta_0"] = n_components
        self._graph.graph["corpus_beta_1"] = (
            self._graph.number_of_edges()
            - self._graph.number_of_nodes()
            + n_components
        )

        return self._graph

    # ------------------------------------------------------------------ #
    # Online: Query-to-node matching
    # ------------------------------------------------------------------ #

    def _match_query_to_nodes(
        self,
        question: str,
        titles: list[str],
    ) -> list[int]:
        """Find graph nodes relevant to the query.

        Three-tier matching (adapted from entity_graph._find_q_relevant_paragraphs):

        Tier 1 (Strong): Node's paragraph title appears in the question
        Tier 2 (Medium): Node shares 2+ entities with the question
        Tier 3 (Weak):   2+ title words appear in question (fallback)

        Returns list of sentence-level node indices, sorted by match strength.
        """
        if self._graph is None:
            return []

        q_lower = question.lower()
        q_entities = extract_entities(question)
        q_words = set(q_lower.split())

        # Score each node
        node_scores: list[tuple[int, float]] = []

        for node in self._graph.nodes:
            p_idx = self._node_to_para[node]
            title = titles[p_idx]
            title_lower = title.lower()

            score = 0.0

            # Tier 1: Title in question (strong signal)
            if title_lower in q_lower and len(title_lower) > 3:
                score += 3.0

            # Tier 2: Entity overlap with question
            node_ents = self._node_entities.get(node, set())
            entity_overlap = len(q_entities & node_ents)
            if entity_overlap >= 2:
                score += 2.0 + 0.1 * entity_overlap

            # Tier 3: Title words in question (weak)
            title_words = set(title_lower.split())
            title_word_overlap = len(title_words & q_words)
            if title_word_overlap >= 2:
                score += 1.0 + 0.05 * title_word_overlap

            if score > 0:
                node_scores.append((node, score))

        # Sort by score descending
        node_scores.sort(key=lambda x: -x[1])

        # Return matched nodes
        matched = [n for n, _ in node_scores]

        # Fallback: if no matches, take top-3 paragraphs by entity overlap
        if not matched:
            para_scores: dict[int, int] = defaultdict(int)
            for node in self._graph.nodes:
                node_ents = self._node_entities.get(node, set())
                overlap = len(q_entities & node_ents)
                p_idx = self._node_to_para[node]
                para_scores[p_idx] = max(para_scores[p_idx], overlap)

            top_paras = sorted(para_scores, key=lambda p: -para_scores[p])[:3]
            for p_idx in top_paras:
                matched.extend(self._para_to_nodes.get(p_idx, []))

        return matched

    # ------------------------------------------------------------------ #
    # Online: Subgraph extraction
    # ------------------------------------------------------------------ #

    def extract_subgraph(
        self,
        question: str,
        titles: list[str],
        sentences_list: list[list[str]],
        gold_titles: list[str] | None = None,
    ) -> SubgraphResult:
        """Extract k-hop subgraph from query-matched nodes.

        Algorithm:
          1. Match query to seed nodes
          2. BFS k-hop expansion from seeds (unweighted)
          3. Map sentence nodes to unique paragraph indices
          4. Cap at max_subgraph_paras (by centrality_sum per paragraph)
          5. Compute subgraph topology (beta_0, beta_1)
          6. Compute gold precision/recall if gold_titles provided

        Parameters
        ----------
        question : str
            The query.
        titles : list[str]
            Paragraph titles.
        sentences_list : list[list[str]]
            Sentences per paragraph.
        gold_titles : list[str] | None
            Gold supporting paragraph titles for diagnostic evaluation.

        Returns
        -------
        SubgraphResult
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return SubgraphResult(
                paragraph_indices=[], paragraph_titles=[],
                n_subgraph_nodes=0, n_subgraph_edges=0,
                query_matched_nodes=0,
            )

        # Step 1: Find seed nodes
        seeds = self._match_query_to_nodes(question, titles)
        n_seeds = len(seeds)

        if not seeds:
            # No matches — return empty subgraph
            return SubgraphResult(
                paragraph_indices=[], paragraph_titles=[],
                n_subgraph_nodes=0, n_subgraph_edges=0,
                query_matched_nodes=0,
            )

        # Step 2: BFS k-hop expansion
        subgraph_nodes: set[int] = set()
        frontier: set[int] = set(seeds)

        for _hop in range(self.k_hop + 1):  # 0-hop = seeds only
            subgraph_nodes.update(frontier)
            next_frontier: set[int] = set()
            for node in frontier:
                for neighbor in self._graph.neighbors(node):
                    if neighbor not in subgraph_nodes:
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        # Step 3: Map nodes to paragraph indices
        para_centrality: dict[int, float] = defaultdict(float)
        para_nodes: dict[int, list[int]] = defaultdict(list)

        for node in subgraph_nodes:
            p_idx = self._node_to_para[node]
            cent = self._graph.nodes[node].get("centrality", 0.0)
            para_centrality[p_idx] += cent
            para_nodes[p_idx].append(node)

        # Step 4: Cap at max_subgraph_paras by centrality sum
        para_indices = sorted(
            para_centrality.keys(),
            key=lambda p: -para_centrality[p],
        )

        if len(para_indices) > self.max_subgraph_paras:
            para_indices = para_indices[:self.max_subgraph_paras]
            # Recompute subgraph_nodes to only include selected paragraphs
            selected_paras = set(para_indices)
            subgraph_nodes = {
                n for n in subgraph_nodes
                if self._node_to_para[n] in selected_paras
            }

        # Step 5: Compute subgraph topology
        sub_g = self._graph.subgraph(subgraph_nodes)
        n_sub_nodes = sub_g.number_of_nodes()
        n_sub_edges = sub_g.number_of_edges()
        beta_0 = nx.number_connected_components(sub_g) if n_sub_nodes > 0 else 0
        beta_1 = max(0, n_sub_edges - n_sub_nodes + beta_0)

        # Step 6: Compute gold precision/recall
        para_titles = [titles[i] for i in para_indices]
        gold_precision = 0.0
        gold_recall = 0.0

        if gold_titles and para_titles:
            gold_set = set(gold_titles)
            sub_set = set(para_titles)
            intersection = gold_set & sub_set
            gold_precision = len(intersection) / len(sub_set) if sub_set else 0.0
            gold_recall = len(intersection) / len(gold_set) if gold_set else 0.0

        return SubgraphResult(
            paragraph_indices=para_indices,
            paragraph_titles=para_titles,
            n_subgraph_nodes=n_sub_nodes,
            n_subgraph_edges=n_sub_edges,
            query_matched_nodes=n_seeds,
            beta_0=beta_0,
            beta_1=beta_1,
            gold_precision=round(gold_precision, 4),
            gold_recall=round(gold_recall, 4),
        )

    # ------------------------------------------------------------------ #
    # Online: F-value computation
    # ------------------------------------------------------------------ #

    def compute_f_value(
        self,
        subgraph_result: SubgraphResult,
    ) -> tuple[float, dict[str, float]]:
        """Compute F-value for routing from subgraph topology.

        Simplified gauge based on subgraph structure:

          F = (1 - beta_0_norm) + beta_1_norm

        Where:
          beta_0_norm = (beta_0 - 1) / max(n_paras - 1, 1)
            0 = fully connected, 1 = all disconnected
          beta_1_norm = beta_1 / max(n_edges, 1)
            0 = tree, higher = more cycles/redundant paths

        Intuition:
          - High F: well-connected subgraph with redundant paths
            → LLM can reason from subgraph alone (System 1)
          - Low F: disconnected subgraph with tree-like structure
            → Need full context for missing links (System 2)

        Returns
        -------
        (f_value, diagnostics_dict)
        """
        n_paras = len(subgraph_result.paragraph_indices)
        beta_0 = subgraph_result.beta_0
        beta_1 = subgraph_result.beta_1
        n_edges = subgraph_result.n_subgraph_edges

        if n_paras == 0:
            return 0.0, {"beta_0_norm": 1.0, "beta_1_norm": 0.0}

        # Normalize beta_0: 0 = fully connected, 1 = all isolated
        beta_0_norm = (beta_0 - 1) / max(n_paras - 1, 1)

        # Normalize beta_1: cycles relative to edges
        beta_1_norm = beta_1 / max(n_edges, 1)

        # F = connectivity + redundancy
        f_value = (1.0 - beta_0_norm) + beta_1_norm

        diagnostics = {
            "beta_0_norm": round(beta_0_norm, 4),
            "beta_1_norm": round(beta_1_norm, 4),
            "n_paras": n_paras,
            "beta_0": beta_0,
            "beta_1": beta_1,
            "n_edges": n_edges,
        }

        return round(f_value, 4), diagnostics

    # ------------------------------------------------------------------ #
    # Online: Full routing pipeline
    # ------------------------------------------------------------------ #

    def route(
        self,
        question: str,
        titles: list[str],
        sentences_list: list[list[str]],
        theta_f: float = 0.0,
        gold_titles: list[str] | None = None,
    ) -> RoutingDecision:
        """Full routing pipeline: extract → compute F → decide system.

        Parameters
        ----------
        question : str
            The query.
        titles : list[str]
            All paragraph titles for this example.
        sentences_list : list[list[str]]
            All paragraph sentences.
        theta_f : float
            F-value threshold.
            F >= theta_f → System 1 (answer from subgraph paragraphs only)
            F <  theta_f → System 2 (full context with subgraph first)
        gold_titles : list[str] | None
            Gold supporting paragraph titles for diagnostic evaluation.

        Returns
        -------
        RoutingDecision
        """
        # Step 1: Extract subgraph
        subgraph = self.extract_subgraph(
            question, titles, sentences_list, gold_titles,
        )

        # Step 2: Compute F-value
        f_value, _diagnostics = self.compute_f_value(subgraph)

        # Step 3: Routing decision
        if f_value >= theta_f and subgraph.paragraph_indices:
            system = "system1"
            # System 1: use only subgraph paragraphs
            context_paras = list(subgraph.paragraph_indices)
        else:
            system = "system2"
            # System 2: subgraph paragraphs first, then remaining
            chain_set = set(subgraph.paragraph_indices)
            remaining = [i for i in range(len(titles)) if i not in chain_set]
            context_paras = list(subgraph.paragraph_indices) + remaining

        # Estimate tokens
        tokens_est = 0
        for i in context_paras:
            if i < len(sentences_list):
                tokens_est += sum(len(s.split()) for s in sentences_list[i])

        return RoutingDecision(
            system=system,
            f_value=f_value,
            subgraph=subgraph,
            context_paragraphs=context_paras,
            context_tokens_est=tokens_est,
            beta_0_subgraph=subgraph.beta_0,
            beta_1_subgraph=subgraph.beta_1,
        )
