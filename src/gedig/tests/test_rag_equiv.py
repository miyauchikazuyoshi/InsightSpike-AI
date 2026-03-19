"""RAG adapter equivalence tests (R1-R3).

Verifies that the unified adapter produces numerically equivalent
results to the legacy compute_qkv_attention implementation.

R4-R6 (E2E tests) require full data and are run separately.
"""

import math
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

# Add RAG experiment to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments" / "hotpotqa_v2" / "src"))


def _build_small_graph():
    """Build a small test graph with S and T nodes."""
    g = nx.DiGraph()

    # 2 sentence nodes
    g.add_node("s_0_0", node_type="S", para_idx=0, sent_idx=0,
               title="doc_0", text="The enzyme catalyzes reactions")
    g.add_node("s_1_0", node_type="S", para_idx=1, sent_idx=0,
               title="doc_1", text="Mitochondria contain many enzymes")

    # 2 token nodes
    g.add_node("t_0_0", node_type="T", lemma="enzyme", pos="NOUN",
               sent_idx=0, doc_idx=0, dep_role="head", is_root=False, idf_norm=0.5)
    g.add_node("t_1_0", node_type="T", lemma="enzyme", pos="NOUN",
               sent_idx=0, doc_idx=1, dep_role="child", is_root=False, idf_norm=0.5)

    # Edges
    g.add_edge("s_0_0", "s_1_0", edge_type="entity_overlap", cost=0.3)
    g.add_edge("s_1_0", "s_0_0", edge_type="entity_overlap", cost=0.3)
    g.add_edge("s_0_0", "t_0_0", edge_type="contains", cost=0.05)
    g.add_edge("t_0_0", "s_0_0", edge_type="contains", cost=0.05)
    g.add_edge("s_1_0", "t_1_0", edge_type="contains", cost=0.05)
    g.add_edge("t_1_0", "s_1_0", edge_type="contains", cost=0.05)
    g.add_edge("t_0_0", "t_1_0", edge_type="same_lemma_x", cost=0.4)
    g.add_edge("t_1_0", "t_0_0", edge_type="same_lemma_x", cost=0.4)

    return g


def _add_qk_vectors(graph, query_lemmas):
    """Add Q/K vectors to nodes (simulating compute_qkv_attention's node setup)."""
    for nid, data in graph.nodes(data=True):
        ntype = data.get("node_type", "S")
        if ntype == "T":
            lemma = data.get("lemma", "")
            direct = 1.0 if lemma in query_lemmas else 0.0
            q_vec = np.array([1.0 * direct, 0.6 * 0.0, 0.8 * 0.0])
            pos_w = 1.0 if data.get("pos") == "NOUN" else 0.6
            idf = data.get("idf_norm", 0.5)
            dep_c = 0.7 if data.get("dep_role") == "head" else 0.3
            k_vec = np.array([1.0 * pos_w, 0.8 * idf, 0.5 * dep_c])
        else:
            text = data.get("text", "").lower()
            match = sum(1 for ql in query_lemmas if ql in text)
            q_vec = np.array([1.0 * match / max(len(query_lemmas), 1), 0.6 * 0.0, 0.8 * 0.0])
            k_vec = np.array([1.0 * 0.5, 0.8 * 0.5, 0.5 * 1.0])

        data["q_vec"] = q_vec
        data["k_vec"] = k_vec
        data["q_score"] = float(np.linalg.norm(q_vec))


# ─── R1: Per-edge F-value Equivalence ────────────────────────────

class TestR1_EdgeFValueEquivalence:
    """R1: |f_old - f_new| < 1e-6 on identical graph."""

    def test_manual_f_matches_adapter(self):
        """Compare manual f = cost - λ·dot(Q,K)/√d to adapter."""
        from gedig.adapters.rag import RAGFEval

        q = [0.8, 0.3, 0.0]
        k = [1.0, 0.4, 0.15]
        cost = 0.3
        f_lambda = 1.0
        d_k = 3.0

        # Manual computation (legacy formula)
        dot = sum(qi * ki for qi, ki in zip(q, k))
        alpha = dot / math.sqrt(d_k)
        f_manual = cost - f_lambda * alpha

        # Adapter computation
        adapter = RAGFEval(f_lambda=f_lambda, d_k=d_k)
        f_adapter = adapter.compute_edge_f(cost, q, k)

        assert abs(f_manual - f_adapter) < 1e-10

    def test_all_edges_on_graph(self):
        """Compare edge F-values between legacy and unified on a graph."""
        graph = _build_small_graph()
        query_lemmas = {"enzyme", "reaction"}
        _add_qk_vectors(graph, query_lemmas)

        from gedig.adapters.rag import RAGFEval
        adapter = RAGFEval(f_lambda=1.0, d_k=3.0)

        d_k = 3.0
        for u, v, edata in graph.edges(data=True):
            q = graph.nodes[u]["q_vec"]
            k = graph.nodes[v]["k_vec"]
            cost = edata.get("cost", 0.5)

            # Legacy
            alpha = float(np.dot(q, k) / math.sqrt(d_k))
            f_legacy = cost - 1.0 * alpha

            # Unified
            f_unified = adapter.compute_edge_f(cost, q.tolist(), k.tolist())

            assert abs(f_legacy - f_unified) < 1e-10, \
                f"Edge ({u},{v}): legacy={f_legacy:.8f} unified={f_unified:.8f}"


# ─── R2: AG/DG Classification Match ─────────────────────────────

class TestR2_AGDGClassificationMatch:
    """R2: AG/DG classification must match 100%."""

    def test_classification_matches(self):
        graph = _build_small_graph()
        query_lemmas = {"enzyme", "reaction"}
        _add_qk_vectors(graph, query_lemmas)

        from gedig.adapters.rag import RAGFEval
        adapter = RAGFEval(f_lambda=1.0, d_k=3.0, percentile=0.3)

        d_k = 3.0
        f_values = {}
        for u, v, edata in graph.edges(data=True):
            q = graph.nodes[u]["q_vec"]
            k = graph.nodes[v]["k_vec"]
            cost = edata.get("cost", 0.5)
            alpha = float(np.dot(q, k) / math.sqrt(d_k))
            f_val = cost - 1.0 * alpha
            f_values[(u, v)] = f_val

        # Legacy classification
        sorted_f = sorted(f_values.values())
        theta_legacy = sorted_f[int(len(sorted_f) * 0.3)]
        legacy_ag = {e for e, fv in f_values.items() if fv < theta_legacy}
        legacy_dg = {e for e, fv in f_values.items() if fv >= theta_legacy}

        # Unified classification
        result = adapter.classify_edges(f_values)
        unified_ag = set(result.ag_edges)
        unified_dg = set(result.dg_edges)

        assert legacy_ag == unified_ag, f"AG mismatch: {legacy_ag ^ unified_ag}"
        assert legacy_dg == unified_dg, f"DG mismatch: {legacy_dg ^ unified_dg}"


# ─── R3: Propagation Equivalence ────────────────────────────────

class TestR3_PropagationEquivalence:
    """R3: |rel_old - rel_new| < 1e-6 on all nodes."""

    def test_propagation_matches(self):
        """Compare legacy graph_attention_propagation vs unified adapter."""
        graph = _build_small_graph()
        query_lemmas = {"enzyme"}
        _add_qk_vectors(graph, query_lemmas)

        # Set up flow weights on edges
        for u, v, edata in graph.edges(data=True):
            q = graph.nodes[u]["q_vec"]
            k = graph.nodes[v]["k_vec"]
            alpha = float(np.dot(q, k) / math.sqrt(3.0))
            cost = edata.get("cost", 0.5)
            edata["flow"] = max(alpha * cost, 0.0)

        import copy

        # Legacy propagation
        g_legacy = copy.deepcopy(graph)
        for nid, data in g_legacy.nodes(data=True):
            data["relevance"] = data.get("q_score", 0.0)

        for _ in range(2):
            new_rel = {}
            for nid in g_legacy.nodes:
                preds = list(g_legacy.predecessors(nid))
                if not preds:
                    new_rel[nid] = g_legacy.nodes[nid]["relevance"]
                    continue
                total_flow = sum(g_legacy[p][nid].get("flow", 0.0) for p in preds)
                if total_flow > 1e-8:
                    agg = sum(g_legacy[p][nid].get("flow", 0.0) * g_legacy.nodes[p]["relevance"]
                              for p in preds) / total_flow
                else:
                    agg = 0.0
                new_rel[nid] = 0.7 * g_legacy.nodes[nid]["relevance"] + 0.3 * agg
            for nid, rel in new_rel.items():
                g_legacy.nodes[nid]["relevance"] = rel

        # Unified propagation
        g_unified = copy.deepcopy(graph)
        from gedig.adapters.rag import RAGFEval
        rag = RAGFEval()
        init_rel = {nid: data.get("q_score", 0.0) for nid, data in g_unified.nodes(data=True)}
        propagated = rag.propagate(g_unified, init_rel, n_iterations=2, alpha=0.3)

        # Compare
        for nid in graph.nodes:
            rel_legacy = g_legacy.nodes[nid]["relevance"]
            rel_unified = propagated[nid]
            assert abs(rel_legacy - rel_unified) < 1e-6, \
                f"Node {nid}: legacy={rel_legacy:.8f} unified={rel_unified:.8f}"
