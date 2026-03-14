"""Per-document token-level dependency graph scoring (Spec N).

Build a token-level directed graph from spaCy dependency parse for each
candidate document and score it against the query by:
  score = coverage × (1 + proximity_bonus)

Edge types:
  1. dep        — dependency parse (head → child)
  2. root_chain — ROOT-to-ROOT between consecutive sentences (bidirectional)
  3. same_lemma — same content-word lemma across sentences (bidirectional)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

import networkx as nx
import spacy.tokens

logger = logging.getLogger(__name__)

_CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_token_graph_from_doc(doc: spacy.tokens.Doc) -> nx.DiGraph:
    """Build a token-level directed graph from a parsed spaCy Doc.

    Nodes carry ``lemma``, ``pos``, ``sent_idx`` attributes.
    """
    g = nx.DiGraph()

    # --- nodes + dep edges (intra-sentence tree) ---
    for token in doc:
        g.add_node(
            token.i,
            lemma=token.lemma_.lower(),
            pos=token.pos_,
            sent_idx=token.sent.start,
        )
        if token.head.i != token.i:
            g.add_edge(token.head.i, token.i, edge_type="dep")

    # --- root_chain edges (inter-sentence discourse flow) ---
    roots = [sent.root.i for sent in doc.sents]
    for i in range(len(roots) - 1):
        g.add_edge(roots[i], roots[i + 1], edge_type="root_chain")
        g.add_edge(roots[i + 1], roots[i], edge_type="root_chain")

    # --- same_lemma edges (cross-sentence lexical cohesion) ---
    lemma_positions: dict[str, list[int]] = defaultdict(list)
    for token in doc:
        if token.pos_ in _CONTENT_POS and len(token.lemma_) > 2:
            lemma_positions[token.lemma_.lower()].append(token.i)

    for _lemma, positions in lemma_positions.items():
        if len(positions) < 2 or len(positions) > 20:
            continue
        # Group by sentence
        sent_groups: dict[int, list[int]] = defaultdict(list)
        for pos in positions:
            sent_groups[g.nodes[pos]["sent_idx"]].append(pos)
        sent_keys = sorted(sent_groups.keys())
        # Connect representative nodes across nearby sentences (window=3)
        for i in range(len(sent_keys)):
            for j in range(i + 1, min(i + 4, len(sent_keys))):
                src = sent_groups[sent_keys[i]][0]
                dst = sent_groups[sent_keys[j]][0]
                if not g.has_edge(src, dst):
                    g.add_edge(src, dst, edge_type="same_lemma")
                    g.add_edge(dst, src, edge_type="same_lemma")

    return g


# ---------------------------------------------------------------------------
# F-Evaluation based DG/AG classification (Spec N.2)
# ---------------------------------------------------------------------------

def _query_relevance(
    u: int, v: int, g: nx.DiGraph, query_lemmas: set[str],
) -> float:
    """Edge (u,v) の query 関連度 [0, 1].

    0-hop F 評価の IG 成分に対応。端点の直接マッチ + 1-hop 近傍密度。
    """
    rel_u = 1.0 if g.nodes[u]["lemma"] in query_lemmas else 0.0
    rel_v = 1.0 if g.nodes[v]["lemma"] in query_lemmas else 0.0
    direct = max(rel_u, rel_v)

    u_nbrs = set(g.predecessors(u)) | set(g.successors(u))
    v_nbrs = set(g.predecessors(v)) | set(g.successors(v))
    u_density = (
        sum(1 for n in u_nbrs if g.nodes[n]["lemma"] in query_lemmas)
        / max(len(u_nbrs), 1)
    )
    v_density = (
        sum(1 for n in v_nbrs if g.nodes[n]["lemma"] in query_lemmas)
        / max(len(v_nbrs), 1)
    )
    neighborhood = (u_density + v_density) / 2

    return 0.6 * direct + 0.4 * neighborhood


def _edge_structural_cost(u: int, v: int, g: nx.DiGraph) -> float:
    """Edge type に基づく構造コスト [0, 1].

    0-hop F 評価の GED 成分に対応。
    """
    etype = g[u][v].get("edge_type", "dep")
    if etype == "dep":
        return 0.2
    elif etype == "root_chain":
        sent_dist = abs(g.nodes[u]["sent_idx"] - g.nodes[v]["sent_idx"])
        return 0.3 + 0.1 * min(sent_dist, 3)
    elif etype == "same_lemma":
        sent_dist = abs(g.nodes[u]["sent_idx"] - g.nodes[v]["sent_idx"])
        return 0.4 + 0.1 * min(sent_dist, 5)
    elif etype in ("insight_bridge", "insight_query"):
        return 0.3  # insight edges = moderate cost
    return 0.5


def _classify_edges_f_eval(
    ug: nx.Graph,
    doc_graph: nx.DiGraph,
    query_lemmas: set[str],
    f_lambda: float = 1.0,
) -> dict:
    """F-evaluation ベースの DG/AG 分類 (Spec N.2).

    geDIG 原理: f_eval = edge_cost - λ · query_relevance
      f_eval < θ → AG (query 関連が高い = 確認済み → cost=1.0)
      f_eval ≥ θ → DG (不確実 → cost = 1.0 + (f_val - θ))

    コストはデータから自然に導出（マジックナンバー不要）。
    """
    if ug.number_of_edges() == 0:
        return {
            "n_ag": 0, "n_dg": 0, "beta_1": 0,
            "theta": 0.0, "f_values": {},
        }

    # Compute F-value for each edge
    f_values: dict[tuple[int, int], float] = {}
    for u, v in ug.edges():
        # Use directed graph for structural cost and relevance
        if doc_graph.has_edge(u, v):
            e_cost = _edge_structural_cost(u, v, doc_graph)
        elif doc_graph.has_edge(v, u):
            e_cost = _edge_structural_cost(v, u, doc_graph)
        else:
            e_cost = 0.5
        q_rel = _query_relevance(u, v, doc_graph, query_lemmas)
        f_values[(u, v)] = e_cost - f_lambda * q_rel

    # Dynamic threshold: bottom 30% = AG
    vals = sorted(f_values.values())
    theta = vals[int(len(vals) * 0.3)] if vals else 0.0

    n_ag, n_dg = 0, 0
    for (u, v), fv in f_values.items():
        if fv < theta:
            ug[u][v]["cost"] = 1.0  # AG: confirmed, low cost
            n_ag += 1
        else:
            ug[u][v]["cost"] = 1.0 + (fv - theta)  # DG: natural scale
            n_dg += 1

    n_components = nx.number_connected_components(ug)
    beta_1 = ug.number_of_edges() - ug.number_of_nodes() + n_components

    return {
        "n_ag": n_ag, "n_dg": n_dg, "beta_1": beta_1,
        "theta": round(theta, 4), "f_values": f_values,
    }


# ---------------------------------------------------------------------------
# DG/AG edge classification (Spec N.1: geDIG Walk Score)
# ---------------------------------------------------------------------------

def _classify_edges_dg_ag(
    ug: nx.Graph,
    dg_penalty: float = 2.0,
) -> dict:
    """Classify edges as DG (bridge) vs AG (cycle) and set ``cost`` attribute.

    Bridge edges (DG) = structurally critical, single-chain connection.
    Cycle edges (AG) = confirmatory, multiple independent paths exist.

    AG edges get cost=1.0 (favored), DG edges get cost=dg_penalty (penalised).
    This makes weighted shortest paths prefer cycle-rich subgraphs.

    Returns diagnostics dict with n_bridges, n_cycles, beta_1.
    """
    if ug.number_of_edges() == 0:
        return {"n_bridges": 0, "n_cycles": 0, "beta_1": 0}

    bridge_set = set(frozenset(e) for e in nx.bridges(ug))

    n_bridges = 0
    n_cycles = 0
    for u, v in ug.edges():
        is_bridge = frozenset((u, v)) in bridge_set
        ug[u][v]["cost"] = dg_penalty if is_bridge else 1.0
        if is_bridge:
            n_bridges += 1
        else:
            n_cycles += 1

    # β₁ = E - V + C (independent cycle count)
    n_components = nx.number_connected_components(ug)
    beta_1 = ug.number_of_edges() - ug.number_of_nodes() + n_components

    return {"n_bridges": n_bridges, "n_cycles": n_cycles, "beta_1": beta_1}


# ---------------------------------------------------------------------------
# Insight vector generation + injection (Spec N.2)
# ---------------------------------------------------------------------------

def _insight_pattern_a(
    doc_graph: nx.DiGraph,
    ug: nx.Graph,
    f_diag: dict,
    query_lemmas: set[str],
) -> list[str]:
    """Pattern A: DG ブリッジ lemma 集約.

    AG-only サブグラフで query ノードの連結成分を求め、
    異なる query 成分を繋ぐノードの lemma を抽出。
    """
    theta = f_diag["theta"]
    f_values = f_diag["f_values"]

    # AG-only サブグラフ
    ag_edges = [(u, v) for (u, v), fv in f_values.items() if fv < theta]
    if not ag_edges:
        return []
    ag_sub = nx.Graph()
    ag_sub.add_edges_from(ag_edges)

    # query ノードの連結成分
    query_nodes = {
        n for n, d in doc_graph.nodes(data=True)
        if d["lemma"] in query_lemmas
    }
    if not query_nodes:
        return []

    components = list(nx.connected_components(ag_sub))
    query_comps = [c for c in components if c & query_nodes]

    if len(query_comps) < 2:
        return []  # 穴がない

    # 全グラフ上で異なる query 成分を繋ぐノード = bridging nodes
    bridging: set[str] = set()
    for node in ug.nodes():
        connected_comps: set[int] = set()
        for nbr in ug.neighbors(node):
            for ci, comp in enumerate(query_comps):
                if nbr in comp:
                    connected_comps.add(ci)
        if len(connected_comps) >= 2:
            lem = doc_graph.nodes[node]["lemma"]
            if lem not in query_lemmas:  # query lemma 自体は除外
                bridging.add(lem)

    return sorted(bridging)[:5]


def _insight_pattern_b(
    doc_graph: nx.DiGraph,
    ug: nx.Graph,
    f_diag: dict,
    query_lemmas: set[str],
    lemma_repr: dict[str, int],
) -> list[str]:
    """Pattern B: DG パス上の中間 lemma.

    Query lemma ノード対の weighted 最短パスで DG エッジを通過する
    中間ノードの lemma を抽出（推論ステップ）。
    """
    theta = f_diag["theta"]
    f_values = f_diag["f_values"]
    repr_nodes = list(lemma_repr.values())
    if len(repr_nodes) < 2:
        return []

    bridging_lemmas: set[str] = set()
    pairs = list(combinations(repr_nodes[:8], 2))

    for src, dst in pairs:
        try:
            path = nx.shortest_path(ug, src, dst, weight="cost")
        except nx.NetworkXNoPath:
            continue
        # パス上の各エッジをチェック
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            rkey = (path[i + 1], path[i])
            fv = f_values.get(key, f_values.get(rkey, 0.0))
            if fv >= theta:  # DG エッジ
                for node in (path[i], path[i + 1]):
                    lem = doc_graph.nodes[node]["lemma"]
                    if lem not in query_lemmas:
                        bridging_lemmas.add(lem)

    return sorted(bridging_lemmas)[:5]


def _inject_insights(
    doc_graph: nx.DiGraph,
    insight_lemmas: list[str],
    query_lemmas: set[str],
    max_edges_per_insight: int = 10,
) -> nx.DiGraph:
    """洞察 lemma をノードとして inject し、既存ノードとエッジを張る.

    insight ノードは coverage に含めない（水増し防止）。
    proximity_bonus のみに影響 → query ノード間の経路短縮。
    """
    if not insight_lemmas:
        return doc_graph

    g = doc_graph.copy()
    max_id = max(g.nodes()) + 1 if g.nodes() else 0

    for i, lemma in enumerate(insight_lemmas):
        nid = max_id + i
        g.add_node(nid, lemma=lemma, pos="INSIGHT", sent_idx=-1)

        edges_added = 0
        # 同一 lemma のノードと接続
        for existing, attrs in doc_graph.nodes(data=True):
            if attrs["lemma"] == lemma and edges_added < max_edges_per_insight:
                g.add_edge(nid, existing, edge_type="insight_bridge")
                g.add_edge(existing, nid, edge_type="insight_bridge")
                edges_added += 1

        # Query lemma ノードにも接続 (B の穴埋め)
        for existing, attrs in doc_graph.nodes(data=True):
            if (
                attrs["lemma"] in query_lemmas
                and not g.has_edge(nid, existing)
                and edges_added < max_edges_per_insight
            ):
                g.add_edge(nid, existing, edge_type="insight_query")
                g.add_edge(existing, nid, edge_type="insight_query")
                edges_added += 1

    return g


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_single(
    query_lemmas: set[str],
    doc_graph: nx.DiGraph,
    max_sp_pairs: int = 50,
    use_walk_score: bool = False,
    dg_penalty: float = 2.0,
    use_f_eval: bool = False,
    f_lambda: float = 1.0,
    insight_mode: str = "none",
) -> tuple[float, dict]:
    """Score a single document graph against query lemmas.

    Three DG/AG classification modes (mutually exclusive):
      - Default: no classification, uniform edge cost
      - use_walk_score: Tarjan bridge detection (Spec N.1)
      - use_f_eval: F-evaluation based (Spec N.2) — overrides walk_score

    When *use_f_eval* is True, insight_mode controls insight injection:
      - "none": F-eval classification only, no insight injection
      - "graph_agg": Pattern A — DG bridge lemma aggregation
      - "path_bridge": Pattern B — DG path intermediate nodes
      - "both": Pattern A + B combined

    Returns (score, diagnostics_dict).
    """
    empty_diag = {
        "n_matched": 0, "coverage": 0.0,
        "avg_sp": 0.0, "proximity_bonus": 0.0,
        "n_bridges": 0, "n_cycles": 0, "beta_1": 0,
        "n_ag": 0, "n_dg": 0, "f_theta": 0.0,
        "n_insights": 0, "insight_lemmas": [],
    }
    if not query_lemmas or doc_graph.number_of_nodes() == 0:
        return 0.0, empty_diag

    # Find one representative node per matched query lemma
    # (exclude INSIGHT nodes from coverage to prevent inflation)
    lemma_repr: dict[str, int] = {}
    for nid, attrs in doc_graph.nodes(data=True):
        if attrs.get("pos") == "INSIGHT":
            continue
        lem = attrs["lemma"]
        if lem in query_lemmas and lem not in lemma_repr:
            lemma_repr[lem] = nid

    n_matched = len(lemma_repr)
    coverage = n_matched / len(query_lemmas)

    if n_matched < 2:
        return coverage, {
            "n_matched": n_matched, "coverage": round(coverage, 4),
            "avg_sp": 0.0, "proximity_bonus": 0.0,
            "n_bridges": 0, "n_cycles": 0, "beta_1": 0,
            "n_ag": 0, "n_dg": 0, "f_theta": 0.0,
            "n_insights": 0, "insight_lemmas": [],
        }

    # --- DG/AG classification ---
    use_weighted = use_f_eval or use_walk_score
    walk_diag: dict = {"n_bridges": 0, "n_cycles": 0, "beta_1": 0}
    f_diag: dict = {"n_ag": 0, "n_dg": 0, "beta_1": 0, "theta": 0.0, "f_values": {}}

    if use_f_eval:
        # Spec N.2: F-evaluation (overrides Tarjan)
        ug = doc_graph.to_undirected()
        f_diag = _classify_edges_f_eval(ug, doc_graph, query_lemmas, f_lambda=f_lambda)

        # --- Insight injection ---
        insight_lemmas: list[str] = []
        if insight_mode in ("graph_agg", "both"):
            insight_lemmas.extend(
                _insight_pattern_a(doc_graph, ug, f_diag, query_lemmas)
            )
        if insight_mode in ("path_bridge", "both"):
            insight_lemmas.extend(
                _insight_pattern_b(doc_graph, ug, f_diag, query_lemmas, lemma_repr)
            )
        # Deduplicate
        insight_lemmas = list(dict.fromkeys(insight_lemmas))

        if insight_lemmas:
            # Inject insights and re-build undirected graph
            doc_graph = _inject_insights(doc_graph, insight_lemmas, query_lemmas)
            ug = doc_graph.to_undirected()
            # Re-classify edges including new insight edges
            f_diag = _classify_edges_f_eval(ug, doc_graph, query_lemmas, f_lambda=f_lambda)

    elif use_walk_score:
        # Spec N.1: Tarjan bridge detection (legacy)
        ug = doc_graph.to_undirected()
        walk_diag = _classify_edges_dg_ag(ug, dg_penalty=dg_penalty)
        insight_lemmas = []
    else:
        ug = doc_graph.to_undirected()
        insight_lemmas = []

    repr_nodes = list(lemma_repr.values())
    pairs = list(combinations(repr_nodes, 2))

    if len(pairs) > max_sp_pairs:
        import random
        random.seed(42)
        pairs = random.sample(pairs, max_sp_pairs)

    n_nodes = doc_graph.number_of_nodes()
    sp_sum, sp_count = 0.0, 0
    for src, dst in pairs:
        try:
            if use_weighted:
                sp_sum += nx.shortest_path_length(ug, src, dst, weight="cost")
            else:
                sp_sum += nx.shortest_path_length(ug, src, dst)
        except nx.NetworkXNoPath:
            sp_sum += n_nodes * (dg_penalty if use_walk_score else 2.0)
        sp_count += 1

    avg_sp = sp_sum / sp_count if sp_count else 0.0
    proximity_bonus = 1.0 / (1.0 + avg_sp)

    score = coverage * (1.0 + proximity_bonus)
    diag = {
        "n_matched": n_matched,
        "coverage": round(coverage, 4),
        "avg_sp": round(avg_sp, 2),
        "proximity_bonus": round(proximity_bonus, 4),
        "n_bridges": walk_diag.get("n_bridges", 0),
        "n_cycles": walk_diag.get("n_cycles", 0),
        "beta_1": f_diag.get("beta_1", 0) if use_f_eval else walk_diag.get("beta_1", 0),
        "n_ag": f_diag.get("n_ag", 0),
        "n_dg": f_diag.get("n_dg", 0),
        "f_theta": f_diag.get("theta", 0.0),
        "n_insights": len(insight_lemmas) if use_f_eval else 0,
        "insight_lemmas": insight_lemmas[:5] if use_f_eval else [],
    }
    return score, diag


# ---------------------------------------------------------------------------
# Batch API (public)
# ---------------------------------------------------------------------------

def compute_token_scores_batch(
    query: str,
    doc_texts: list[str],
    nlp,
    max_tokens: int = 500,
    use_walk_score: bool = False,
    dg_penalty: float = 2.0,
    use_f_eval: bool = False,
    f_lambda: float = 1.0,
    insight_mode: str = "none",
) -> tuple[list[float], list[dict]]:
    """Score multiple documents against a query.

    Parameters
    ----------
    query : str
        The search query.
    doc_texts : list[str]
        Document texts to score.
    nlp : spacy.Language
        Pre-loaded spaCy model.
    max_tokens : int
        Truncate each document to this many whitespace-separated words.
    use_walk_score : bool
        When True, classify edges as DG/AG via Tarjan bridge detection
        and use weighted shortest paths (geDIG Walk Score, Spec N.1).
    dg_penalty : float
        Cost penalty for bridge (DG) edges when *use_walk_score* is True.
    use_f_eval : bool
        When True, classify edges via F-evaluation (Spec N.2).
        Overrides *use_walk_score* if both are set.
    f_lambda : float
        Lambda weight for F-evaluation (edge_cost - λ·query_relevance).
    insight_mode : str
        Insight injection mode: "none", "graph_agg", "path_bridge", "both".
        Only active when *use_f_eval* is True.

    Returns
    -------
    scores : list[float]
        Raw scores (not normalised).
    diags : list[dict]
        Per-document diagnostics.
    """
    _empty_diag = {
        "n_matched": 0, "coverage": 0.0, "avg_sp": 0.0,
        "proximity_bonus": 0.0, "n_bridges": 0, "n_cycles": 0, "beta_1": 0,
        "n_ag": 0, "n_dg": 0, "f_theta": 0.0,
        "n_insights": 0, "insight_lemmas": [],
    }

    # Extract query content-word lemmas
    query_doc = nlp(query)
    query_lemmas = {
        t.lemma_.lower()
        for t in query_doc
        if t.pos_ in _CONTENT_POS and len(t.lemma_) > 2
    }

    if not query_lemmas:
        logger.warning("Token graph: no content lemmas in query")
        return [0.0] * len(doc_texts), [dict(_empty_diag) for _ in doc_texts]

    # Truncate and batch-parse
    truncated = [" ".join(text.split()[:max_tokens]) for text in doc_texts]
    parsed = list(nlp.pipe(truncated, batch_size=16))

    scores: list[float] = []
    diags: list[dict] = []
    for pdoc in parsed:
        g = _build_token_graph_from_doc(pdoc)
        s, d = _score_single(
            query_lemmas, g,
            use_walk_score=use_walk_score,
            dg_penalty=dg_penalty,
            use_f_eval=use_f_eval,
            f_lambda=f_lambda,
            insight_mode=insight_mode,
        )
        scores.append(s)
        diags.append(d)

    return scores, diags
