"""Spec Z: Analytical Heterogeneous Graph Transformer (AGHT).

Unified Sentence-Token graph with QKV-based edge evaluation.
Combines entity_graph.py (sentence-level) and token_graph.py (token-level)
into a single heterogeneous graph with cross-level edges.

Node types:
  S (Sentence): para_idx, sent_idx, text, title, entities
  T (Token):    lemma, pos, sent_idx, doc_idx

Edge types (8):
  1. context        S<->S  (same doc, adjacent)     cost 0.05-0.10
  2. entity_overlap S<->S  (cross doc, shared ent)   cost 0.20-0.50
  3. similarity     S<->S  (cross doc, TF-IDF cos)   cost 0.50-0.80
  4. dep            T->T   (same doc, dependency)     cost 0.10-0.20
  5. same_lemma     T<->T  (same doc, lexical)        cost 0.30-0.70
  6. contains       S->T   (sentence contains token)  cost 0.05
  7. same_lemma_x   T<->T  (cross doc, lexical)       cost 0.30-0.70
  8. cot_link       S<->S  (CoT concept virtual)      cost 0.10-0.30

QKV Edge Evaluation:
  Q(u) = query-dependent features of source node
  K(v) = intrinsic features of target node
  alpha = dot(Q, K) / sqrt(d_k)
  f = cost(e) - lambda * alpha   (AG if f < theta, DG otherwise)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

# Content POS tags for token nodes
_CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

# POS importance weights for K features
_POS_WEIGHT = {
    "NOUN": 1.0, "PROPN": 1.0, "VERB": 0.8, "ADJ": 0.6,
}

# Edge base costs
_COST_CONTEXT_ADJ = 0.05
_COST_CONTEXT_NEAR = 0.08
_COST_CONTEXT_FAR = 0.10
_COST_CONTAINS = 0.05
_COST_DEP = 0.15
_CTX_MAX_SENT_DIST = 6


@dataclass
class AGHTConfig:
    """Tunable parameters for AGHT (10 adjustable + fixed)."""
    # Q weights
    w_q1: float = 1.0   # direct match / query entity fraction
    w_q2: float = 0.6   # neighborhood density
    w_q3: float = 0.8   # CoT relevance

    # K weights
    w_k1: float = 1.0   # POS importance / entity density
    w_k2: float = 0.8   # IDF / TF-IDF norm
    w_k3: float = 0.5   # dep centrality / position score

    # V weights
    w_v1: float = 1.0   # base cost (fixed)
    w_v2: float = 0.3   # bridge bonus
    w_v3: float = 0.2   # level crossing

    # F-eval
    f_lambda: float = 1.0

    # Message passing
    mp_iterations: int = 2
    mp_alpha: float = 0.3

    # Graph construction
    max_tokens_per_doc: int = 200
    same_lemma_x_max_freq: int = 10
    sim_threshold: float = 0.30
    max_para_freq: int = 5

    # Deprecated compatibility field. AGHT has delegated to RAGFEval
    # unconditionally since the 2026-02 refactor.
    use_unified_feval: bool = False


@dataclass
class AGHTResult:
    """Diagnostics from AGHT scoring."""
    n_s_nodes: int = 0
    n_t_nodes: int = 0
    n_edges: int = 0
    n_edges_by_type: dict = field(default_factory=dict)
    n_ag: int = 0
    n_dg: int = 0
    f_theta: float = 0.0
    beta_1: int = 0
    build_ms: float = 0.0
    eval_ms: float = 0.0
    mp_ms: float = 0.0
    score_ms: float = 0.0


# ===================================================================
# Phase 3': Build Unified Heterogeneous Graph
# ===================================================================

def build_unified_graph(
    titles: list[str],
    sentences_list: list[list[str]],
    nlp,
    max_para_freq: int = 5,
    max_tokens_per_doc: int = 200,
    same_lemma_x_max_freq: int = 10,
    sim_threshold: float = 0.30,
) -> nx.DiGraph:
    """Build unified sentence-token heterogeneous graph.

    Returns a DiGraph with both S and T nodes and all 8 edge types.
    """
    g = nx.DiGraph()
    n_paras = len(titles)

    # ── Sentence nodes ──────────────────────────────────────
    s_nodes = []  # (node_id, para_idx, sent_idx, text, title)
    para_to_s_nodes: dict[int, list[str]] = defaultdict(list)
    s_flat = 0

    for p_idx in range(n_paras):
        title = titles[p_idx]
        sents = sentences_list[p_idx]
        for s_idx, sent in enumerate(sents):
            nid = f"s_{p_idx}_{s_idx}"
            g.add_node(nid,
                       node_type="S",
                       para_idx=p_idx,
                       sent_idx=s_idx,
                       title=title,
                       text=sent[:200])
            s_nodes.append((nid, p_idx, s_idx, sent, title))
            para_to_s_nodes[p_idx].append(nid)
            s_flat += 1

    # ── Extract entities per sentence (spaCy) ──
    sent_entities: dict[str, set[str]] = {}
    for nid, p_idx, s_idx, text, title in s_nodes:
        doc = nlp(text[:500])
        ents = set()
        for ent in doc.ents:
            if len(ent.text) > 2:
                ents.add(ent.text.lower())
        for chunk in doc.noun_chunks:
            if len(chunk.root.lemma_) > 2 and chunk.root.pos_ in _CONTENT_POS:
                ents.add(chunk.root.lemma_.lower())
        sent_entities[nid] = ents

    # Discriminative entity filtering
    entity_para_freq: dict[str, set[int]] = defaultdict(set)
    for nid, p_idx, _, _, _ in s_nodes:
        for e in sent_entities[nid]:
            entity_para_freq[e].add(p_idx)

    disc_entities: dict[str, set[str]] = {
        nid: {e for e in ents if len(entity_para_freq[e]) <= max_para_freq}
        for nid, ents in sent_entities.items()
    }

    # ── Token nodes (content words only) ──────────────────
    lemma_doc_freq: dict[str, set[int]] = defaultdict(set)  # for IDF
    t_nodes_by_doc: dict[int, list[str]] = defaultdict(list)
    s_to_t_nodes: dict[str, list[str]] = defaultdict(list)  # for contains edges

    for p_idx in range(n_paras):
        full_text = " ".join(sentences_list[p_idx])
        doc = nlp(full_text[:3000])
        t_count = 0
        for token in doc:
            if token.pos_ not in _CONTENT_POS:
                continue
            if len(token.lemma_) <= 2:
                continue
            if t_count >= max_tokens_per_doc:
                break

            tnid = f"t_{p_idx}_{token.i}"
            lemma = token.lemma_.lower()

            # Determine which sentence node this token belongs to
            # Use character offset to find sentence index
            tok_sent_idx = 0
            char_count = 0
            for si, s_text in enumerate(sentences_list[p_idx]):
                char_count += len(s_text) + 1
                if token.idx < char_count:
                    tok_sent_idx = si
                    break

            g.add_node(tnid,
                       node_type="T",
                       lemma=lemma,
                       pos=token.pos_,
                       sent_idx=tok_sent_idx,
                       doc_idx=p_idx,
                       dep_role="head" if token.head.i == token.i else "child",
                       is_root=(token.dep_ == "ROOT"))

            # Track for IDF
            lemma_doc_freq[lemma].add(p_idx)
            t_nodes_by_doc[p_idx].append(tnid)

            # Map to sentence node for contains edges
            s_nid = f"s_{p_idx}_{tok_sent_idx}"
            if g.has_node(s_nid):
                s_to_t_nodes[s_nid].append(tnid)

            # Dep edge (T→T)
            if token.head.i != token.i:
                head_tnid = f"t_{p_idx}_{token.head.i}"
                if g.has_node(head_tnid):
                    g.add_edge(head_tnid, tnid,
                               edge_type="dep", cost=_COST_DEP)
                    g.add_edge(tnid, head_tnid,
                               edge_type="dep", cost=_COST_DEP)
            t_count += 1

    # Compute IDF for K features
    for nid, data in g.nodes(data=True):
        if data.get("node_type") == "T":
            lemma = data["lemma"]
            df = len(lemma_doc_freq.get(lemma, set()))
            idf = math.log(max(n_paras, 1) / max(df, 1)) if df > 0 else 0
            max_idf = math.log(max(n_paras, 1))
            data["idf_norm"] = min(idf / max(max_idf, 1e-6), 1.0)

    # ── Edge Type 1: Context (S<->S, same doc) ──────────
    n_by_type = defaultdict(int)

    for p_idx, s_nids in para_to_s_nodes.items():
        for i in range(len(s_nids)):
            for j in range(i + 1, len(s_nids)):
                si = g.nodes[s_nids[i]]["sent_idx"]
                sj = g.nodes[s_nids[j]]["sent_idx"]
                dist = abs(si - sj)
                if dist > _CTX_MAX_SENT_DIST:
                    continue
                if dist <= 1:
                    cost = _COST_CONTEXT_ADJ
                elif dist <= 3:
                    cost = _COST_CONTEXT_NEAR
                else:
                    cost = _COST_CONTEXT_FAR
                g.add_edge(s_nids[i], s_nids[j],
                           edge_type="context", cost=cost)
                g.add_edge(s_nids[j], s_nids[i],
                           edge_type="context", cost=cost)
                n_by_type["context"] += 1

    # ── Edge Type 2: Entity overlap (S<->S, cross doc) ──
    s_node_ids = [n[0] for n in s_nodes]
    for i in range(len(s_node_ids)):
        ni = s_node_ids[i]
        pi = g.nodes[ni]["para_idx"]
        if not disc_entities.get(ni):
            continue
        for j in range(i + 1, len(s_node_ids)):
            nj = s_node_ids[j]
            pj = g.nodes[nj]["para_idx"]
            if pi == pj:
                continue
            if not disc_entities.get(nj):
                continue
            shared = disc_entities[ni] & disc_entities[nj]
            if not shared:
                continue
            min_c = min(len(disc_entities[ni]), len(disc_entities[nj]))
            ratio = len(shared) / max(min_c, 1)
            cost = 0.20 + 0.30 * (1.0 - ratio)
            if not g.has_edge(ni, nj) or g[ni][nj].get("cost", 1.0) > cost:
                g.add_edge(ni, nj, edge_type="entity_overlap", cost=cost)
                g.add_edge(nj, ni, edge_type="entity_overlap", cost=cost)
                n_by_type["entity_overlap"] += 1

    # ── Edge Type 3: Similarity (S<->S, cross doc) ──
    # Simple TF-IDF based (reuse entity sets as bag-of-words proxy)
    for i in range(len(s_node_ids)):
        ni = s_node_ids[i]
        pi = g.nodes[ni]["para_idx"]
        ei = sent_entities.get(ni, set())
        if not ei:
            continue
        for j in range(i + 1, len(s_node_ids)):
            nj = s_node_ids[j]
            pj = g.nodes[nj]["para_idx"]
            if pi == pj:
                continue
            ej = sent_entities.get(nj, set())
            if not ej:
                continue
            # Jaccard similarity as proxy
            inter = len(ei & ej)
            union = len(ei | ej)
            if union == 0:
                continue
            sim = inter / union
            if sim < sim_threshold:
                continue
            cost = 0.80 - 0.30 * sim
            if not g.has_edge(ni, nj) or g[ni][nj].get("cost", 1.0) > cost:
                g.add_edge(ni, nj, edge_type="similarity", cost=cost)
                g.add_edge(nj, ni, edge_type="similarity", cost=cost)
                n_by_type["similarity"] += 1

    # ── Edge Type 5: same_lemma (T<->T, same doc) ──
    for p_idx, t_nids in t_nodes_by_doc.items():
        lemma_groups: dict[str, list[str]] = defaultdict(list)
        for tnid in t_nids:
            lemma_groups[g.nodes[tnid]["lemma"]].append(tnid)

        for lemma, nodes_list in lemma_groups.items():
            if len(nodes_list) < 2 or len(nodes_list) > 20:
                continue
            for i in range(len(nodes_list)):
                for j in range(i + 1, min(i + 4, len(nodes_list))):
                    ni, nj = nodes_list[i], nodes_list[j]
                    si = g.nodes[ni]["sent_idx"]
                    sj = g.nodes[nj]["sent_idx"]
                    dist = abs(si - sj)
                    if dist > 5:
                        continue
                    cost = 0.30 + 0.08 * min(dist, 5)
                    if not g.has_edge(ni, nj):
                        g.add_edge(ni, nj, edge_type="same_lemma", cost=cost)
                        g.add_edge(nj, ni, edge_type="same_lemma", cost=cost)
                        n_by_type["same_lemma"] += 1

    # ── Edge Type 6: contains (S→T) ──────────────────
    for s_nid, t_nids in s_to_t_nodes.items():
        for tnid in t_nids[:20]:  # cap per sentence
            g.add_edge(s_nid, tnid, edge_type="contains", cost=_COST_CONTAINS)
            g.add_edge(tnid, s_nid, edge_type="contains", cost=_COST_CONTAINS)
            n_by_type["contains"] += 1

    # ── Edge Type 7: same_lemma_x (T<->T, cross doc) ──
    # Build lemma → [(doc_idx, tnid)] index
    lemma_to_tnodes: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for nid, data in g.nodes(data=True):
        if data.get("node_type") == "T":
            lemma_to_tnodes[data["lemma"]].append((data["doc_idx"], nid))

    for lemma, entries in lemma_to_tnodes.items():
        # Skip very common or very rare lemmas
        doc_set = {d for d, _ in entries}
        if len(doc_set) < 2 or len(doc_set) > same_lemma_x_max_freq:
            continue
        # Connect one representative per doc pair
        by_doc: dict[int, str] = {}
        for doc_idx, tnid in entries:
            if doc_idx not in by_doc:
                by_doc[doc_idx] = tnid
        doc_ids = sorted(by_doc.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                ni = by_doc[doc_ids[i]]
                nj = by_doc[doc_ids[j]]
                cost = 0.40  # moderate cross-doc cost
                if not g.has_edge(ni, nj):
                    g.add_edge(ni, nj, edge_type="same_lemma_x", cost=cost)
                    g.add_edge(nj, ni, edge_type="same_lemma_x", cost=cost)
                    n_by_type["same_lemma_x"] += 1

    g.graph["n_edges_by_type"] = dict(n_by_type)
    return g


# ===================================================================
# Phase 4': QKV Edge Evaluation
# ===================================================================

def compute_qkv_attention(
    graph: nx.DiGraph,
    query: str,
    nlp,
    cot_concepts: set[str] | None = None,
    config: AGHTConfig | None = None,
) -> AGHTResult:
    """Compute QKV attention scores and AG/DG classification on all edges.

    Modifies graph in-place: adds q_score, k_score, alpha, f_value,
    f_class attributes to nodes and edges.
    """
    if config is None:
        config = AGHTConfig()
    if cot_concepts is None:
        cot_concepts = set()

    # ── Query lemmas ──
    q_doc = nlp(query[:500])
    query_lemmas = {
        t.lemma_.lower() for t in q_doc
        if t.pos_ in _CONTENT_POS and len(t.lemma_) > 2
    }
    query_entities = set()
    for ent in q_doc.ents:
        if len(ent.text) > 2:
            query_entities.add(ent.text.lower())
    for chunk in q_doc.noun_chunks:
        if len(chunk.root.lemma_) > 2:
            query_entities.add(chunk.root.lemma_.lower())

    # ── Compute Q and K for all nodes ──
    for nid, data in graph.nodes(data=True):
        ntype = data.get("node_type", "S")

        if ntype == "T":
            # Q for Token: direct match + neighborhood density
            lemma = data.get("lemma", "")
            direct = 1.0 if lemma in query_lemmas else 0.0

            # 1-hop neighbor query density
            nbrs = list(graph.predecessors(nid)) + list(graph.successors(nid))
            if nbrs:
                nbr_match = sum(
                    1 for n in nbrs
                    if graph.nodes[n].get("lemma", "") in query_lemmas
                )
                nbr_density = nbr_match / len(nbrs)
            else:
                nbr_density = 0.0

            q_vec = np.array([
                config.w_q1 * direct,
                config.w_q2 * nbr_density,
                0.0,  # no CoT for tokens
            ])

            # K for Token: POS importance, IDF, dep centrality
            pos_w = _POS_WEIGHT.get(data.get("pos", ""), 0.3)
            idf_n = data.get("idf_norm", 0.5)
            dep_cent = 1.0 if data.get("is_root", False) else (
                0.7 if data.get("dep_role", "") == "head" else 0.3
            )

            k_vec = np.array([
                config.w_k1 * pos_w,
                config.w_k2 * idf_n,
                config.w_k3 * dep_cent,
            ])

        else:  # Sentence node
            # Q for Sentence: lemma coverage + token-derived density + CoT overlap
            text = data.get("text", "")
            text_lower = text.lower()

            # (1) Lemma coverage: fraction of query lemmas found in sentence
            # This is the primary signal — uses lemma matching, not just NER
            sent_doc = nlp(text[:300])
            sent_lemmas = {
                t.lemma_.lower() for t in sent_doc
                if t.pos_ in _CONTENT_POS and len(t.lemma_) > 2
            }
            lemma_overlap = len(query_lemmas & sent_lemmas)
            lemma_coverage = lemma_overlap / max(len(query_lemmas), 1)

            # Also check entity matches (additive, not replacement)
            ent_match = sum(1 for e in query_entities if e in text_lower)
            ent_frac = ent_match / max(len(query_entities), 1)

            # Combined: lemma coverage (primary) + entity boost
            q1_combined = min(lemma_coverage + 0.3 * ent_frac, 1.0)

            # (2) Token-derived density: fraction of child T nodes matching query
            # Uses contains edges to propagate token-level info to sentence
            t_children = [
                n for n in graph.successors(nid)
                if graph.nodes[n].get("node_type") == "T"
            ]
            if t_children:
                t_match = sum(
                    1 for n in t_children
                    if graph.nodes[n].get("lemma", "") in query_lemmas
                )
                t_density = t_match / len(t_children)
            else:
                # Fallback: neighbor S density
                nbrs = list(graph.predecessors(nid)) + list(graph.successors(nid))
                s_nbrs = [n for n in nbrs if graph.nodes[n].get("node_type") == "S"]
                if s_nbrs:
                    nbr_match = sum(
                        1 for n in s_nbrs
                        if any(e in graph.nodes[n].get("text", "").lower()
                               for e in query_entities)
                    )
                    t_density = nbr_match / len(s_nbrs)
                else:
                    t_density = 0.0

            # (3) CoT concept overlap
            cot_match = sum(1 for c in cot_concepts if c.lower() in text_lower)
            cot_overlap = cot_match / max(len(cot_concepts), 1)

            q_vec = np.array([
                config.w_q1 * q1_combined,
                config.w_q2 * t_density,
                config.w_q3 * cot_overlap,
            ])

            # K for Sentence: entity density, information richness, position
            n_words = max(len(text.split()), 1)
            # Information richness: unique content lemmas / total words
            info_richness = min(len(sent_lemmas) / max(n_words, 1) * 3, 1.0)
            # Position score (earlier sentences = higher)
            sent_idx = data.get("sent_idx", 0)
            position = 1.0 / (1.0 + 0.2 * sent_idx)

            k_vec = np.array([
                config.w_k1 * min(len(sent_lemmas) / 10.0, 1.0),  # content density
                config.w_k2 * info_richness,
                config.w_k3 * position,
            ])

        data["q_vec"] = q_vec
        data["k_vec"] = k_vec
        data["q_score"] = float(np.linalg.norm(q_vec))

    # ── Compute attention and F-eval for all edges (unified core) ──
    d_k = 3.0
    f_values = {}

    from gedig.adapters.rag import RAGFEval
    rag_feval = RAGFEval(f_lambda=config.f_lambda, d_k=d_k)

    for u, v, edata in graph.edges(data=True):
        q = graph.nodes[u]["q_vec"]
        k = graph.nodes[v]["k_vec"]
        base_cost = edata.get("cost", 0.5)

        # F-eval via unified core
        f_val = rag_feval.compute_edge_f(base_cost, q.tolist(), k.tolist())
        alpha = float(np.dot(q, k) / math.sqrt(d_k))

        # V features
        etype = edata.get("edge_type", "")
        is_cross_level = etype in ("contains", "same_lemma_x")
        is_bridge = etype in ("entity_overlap", "same_lemma_x")
        v_vec = np.array([
            config.w_v1 * base_cost,
            config.w_v2 * (1.0 if is_bridge else 0.0),
            config.w_v3 * (1.0 if is_cross_level else 0.0),
        ])

        f_values[(u, v)] = f_val
        edata["alpha"] = alpha
        edata["v_vec"] = v_vec
        edata["f_value"] = f_val

    # AG/DG classification via unified core
    ag_dg_result = rag_feval.classify_edges(f_values)
    theta = ag_dg_result.threshold
    n_ag, n_dg = ag_dg_result.n_ag, ag_dg_result.n_dg

    ag_set = set(ag_dg_result.ag_edges)
    for (u, v), fv in f_values.items():
        edata = graph[u][v]
        if (u, v) in ag_set:
            edata["f_class"] = "AG"
            edata["flow_cost"] = 1.0
        else:
            edata["f_class"] = "DG"
            edata["flow_cost"] = 1.0 + (fv - theta)
        alpha = edata["alpha"]
        v_norm = float(np.linalg.norm(edata["v_vec"]))
        edata["flow"] = max(alpha * v_norm, 0.0)

    # ── Betti number ──
    ug = graph.to_undirected()
    n_comp = nx.number_connected_components(ug)
    beta_1 = ug.number_of_edges() - ug.number_of_nodes() + n_comp

    return AGHTResult(
        n_s_nodes=sum(1 for _, d in graph.nodes(data=True) if d.get("node_type") == "S"),
        n_t_nodes=sum(1 for _, d in graph.nodes(data=True) if d.get("node_type") == "T"),
        n_edges=graph.number_of_edges(),
        n_edges_by_type=graph.graph.get("n_edges_by_type", {}),
        n_ag=n_ag,
        n_dg=n_dg,
        f_theta=theta,
        beta_1=beta_1,
    )


# ===================================================================
# Phase 5': Graph Attention Message Passing + Document Scoring
# ===================================================================

def graph_attention_propagation(
    graph: nx.DiGraph,
    n_iterations: int = 2,
    mp_alpha: float = 0.3,
    use_unified: bool = True,  # Deprecated no-op; kept for call compatibility
) -> None:
    """Attention-weighted message passing on unified graph.

    Delegates to gedig.core.message_passing.AttentionPropagator.
    ``use_unified`` is ignored because the active AGHT path is fully migrated.
    Updates node['relevance'] in-place.
    """
    # Initialize relevance from Q-score
    init_rel = {}
    for nid, data in graph.nodes(data=True):
        val = data.get("q_score", 0.0)
        data["relevance"] = val
        init_rel[nid] = val

    from gedig.adapters.rag import RAGFEval
    rag = RAGFEval()
    propagated = rag.propagate(graph, init_rel, n_iterations, mp_alpha)
    for nid, rel in propagated.items():
        graph.nodes[nid]["relevance"] = rel


def score_documents(
    graph: nx.DiGraph,
    titles: list[str],
    doc_id_map: dict[str, str],
    bm25_scores: dict[str, float] | None = None,
    bm25_weight: float = 0.3,
) -> dict[str, float]:
    """Aggregate node relevance to per-document scores.

    Uses max-mean hybrid: 0.6 * max(S_relevance) + 0.4 * mean(S_relevance)
    plus token coverage bonus.
    Optionally blends in BM25 scores.
    """
    doc_s_scores: dict[str, list[float]] = defaultdict(list)
    doc_t_scores: dict[str, list[float]] = defaultdict(list)

    for nid, data in graph.nodes(data=True):
        ntype = data.get("node_type", "S")
        if ntype == "S":
            p_idx = data.get("para_idx")
            if p_idx is not None and p_idx < len(titles):
                doc_key = doc_id_map.get(titles[p_idx], "")
                if doc_key:
                    doc_s_scores[doc_key].append(data.get("relevance", 0.0))
        elif ntype == "T":
            doc_idx = data.get("doc_idx")
            if doc_idx is not None and doc_idx < len(titles):
                doc_key = doc_id_map.get(titles[doc_idx], "")
                if doc_key:
                    doc_t_scores[doc_key].append(data.get("relevance", 0.0))

    # Combine with max-mean hybrid
    all_docs = set(doc_s_scores.keys()) | set(doc_t_scores.keys())
    scores = {}
    for doc_id in all_docs:
        if doc_id in doc_s_scores:
            s_vals = doc_s_scores[doc_id]
            s_max = float(max(s_vals))
            s_mean = float(np.mean(s_vals))
            s_score = 0.6 * s_max + 0.4 * s_mean
        else:
            s_score = 0.0

        if doc_id in doc_t_scores:
            t_vals = doc_t_scores[doc_id]
            t_mean = float(np.mean(t_vals))
        else:
            t_mean = 0.0

        graph_score = 0.7 * s_score + 0.3 * t_mean
        scores[doc_id] = graph_score

    # Min-max normalize graph scores
    if scores:
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        if rng > 1e-10:
            scores = {k: (v - mn) / rng for k, v in scores.items()}

    # Blend with BM25 if provided
    if bm25_scores:
        # Normalize BM25
        bm25_vals = list(bm25_scores.values())
        bm25_mn, bm25_mx = min(bm25_vals), max(bm25_vals)
        bm25_rng = bm25_mx - bm25_mn if bm25_mx > bm25_mn else 1.0
        for doc_id in scores:
            bm25_norm = (bm25_scores.get(doc_id, 0.0) - bm25_mn) / bm25_rng
            scores[doc_id] = (1 - bm25_weight) * scores[doc_id] + bm25_weight * bm25_norm

    return scores


# ===================================================================
# Unified entry point
# ===================================================================

def run_aght(
    titles: list[str],
    sentences_list: list[list[str]],
    query: str,
    nlp,
    doc_id_map: dict[str, str],
    cot_concepts: set[str] | None = None,
    config: AGHTConfig | None = None,
    bm25_scores: dict[str, float] | None = None,
) -> tuple[dict[str, float], AGHTResult]:
    """Full AGHT pipeline: build graph → QKV eval → message passing → score.

    Returns (doc_scores, diagnostics).
    """
    import time

    if config is None:
        config = AGHTConfig()

    # Phase 3': Build unified graph
    t0 = time.time()
    graph = build_unified_graph(
        titles, sentences_list, nlp,
        max_para_freq=config.max_para_freq,
        max_tokens_per_doc=config.max_tokens_per_doc,
        same_lemma_x_max_freq=config.same_lemma_x_max_freq,
        sim_threshold=config.sim_threshold,
    )
    build_ms = (time.time() - t0) * 1000

    # Phase 4': QKV evaluation
    t0 = time.time()
    result = compute_qkv_attention(
        graph, query, nlp,
        cot_concepts=cot_concepts,
        config=config,
    )
    eval_ms = (time.time() - t0) * 1000

    # Phase 5': Message passing
    t0 = time.time()
    graph_attention_propagation(
        graph,
        n_iterations=config.mp_iterations,
        mp_alpha=config.mp_alpha,
        use_unified=config.use_unified_feval,
    )
    mp_ms = (time.time() - t0) * 1000

    # Score documents
    t0 = time.time()
    doc_scores = score_documents(graph, titles, doc_id_map,
                                  bm25_scores=bm25_scores)
    score_ms = (time.time() - t0) * 1000

    result.build_ms = build_ms
    result.eval_ms = eval_ms
    result.mp_ms = mp_ms
    result.score_ms = score_ms

    return doc_scores, result, graph
