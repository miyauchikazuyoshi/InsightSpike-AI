from __future__ import annotations

from typing import Any, Dict, List, Tuple
import sqlite3


def load_ds_graph(db_path: str, namespace: str) -> Tuple[List[List[int]], List[List[List[int]]]]:
    """Load full graph snapshot from SQLite data store for a namespace.

    Returns (nodes, edges) where nodes are [r,c,d] and edges are [[u],[v]].
    """
    nodes: List[List[int]] = []
    edges: List[List[List[int]]] = []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT id FROM graph_nodes WHERE namespace=?", (namespace,))
        for (nid,) in cur.fetchall():
            try:
                parts = [int(p) for p in str(nid).split(',')]
                if len(parts) >= 3:
                    nodes.append(parts[:3])
            except Exception:
                continue
        cur.execute("SELECT source_id, target_id FROM graph_edges WHERE namespace=?", (namespace,))
        for su, sv in cur.fetchall():
            try:
                u = [int(p) for p in str(su).split(',')]
                v = [int(p) for p in str(sv).split(',')]
                if len(u) >= 3 and len(v) >= 3:
                    edges.append([u[:3], v[:3]])
            except Exception:
                continue
        conn.close()
    except Exception:
        pass
    return nodes, edges


def load_ds_graph_upto_step(db_path: str, namespace: str, step: int, *, include_timeline: bool = True) -> Tuple[List[List[int]], List[List[List[int]]]]:
    """Load graph snapshot including all nodes/edges with attribute step <= given step.

    Assumes 'attributes' column contains JSON with an optional integer 'step' field.
    """
    nodes: List[List[int]] = []
    edges: List[List[List[int]]] = []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # Nodes
        cur.execute("SELECT id, attributes FROM graph_nodes WHERE namespace=?", (namespace,))
        for nid, attrs in cur.fetchall():
            try:
                step_attr = None
                if isinstance(attrs, str) and attrs:
                    import json
                    meta = json.loads(attrs)
                    step_attr = int(meta.get('step', meta.get('birth_step', 0)))
                if step_attr is not None and step_attr > step:
                    continue
                parts = [int(p) for p in str(nid).split(',')]
                if len(parts) >= 3:
                    nodes.append(parts[:3])
            except Exception:
                continue
        # Edges
        cur.execute("SELECT source_id, target_id, attributes FROM graph_edges WHERE namespace=?", (namespace,))
        for su, sv, attrs in cur.fetchall():
            try:
                step_attr = None
                edge_type = None
                if isinstance(attrs, str) and attrs:
                    import json
                    meta = json.loads(attrs)
                    step_attr = int(meta.get('step', 0))
                    edge_type = (meta.get('edge_type') or meta.get('stage') or '').lower()
                # Optionally skip timeline edges when strict reconstruction is desired
                if (not include_timeline) and edge_type == 'timeline':
                    continue
                if step_attr is not None and step_attr > step:
                    continue
                u = [int(p) for p in str(su).split(',')]
                v = [int(p) for p in str(sv).split(',')]
                if len(u) >= 3 and len(v) >= 3:
                    edges.append([u[:3], v[:3]])
            except Exception:
                continue
        conn.close()
    except Exception:
        pass
    return nodes, edges


def reconstruct_records(
    db_path: str,
    namespace: str,
    records: List[Dict[str, Any]],
    *,
    mode: str = "strict",
) -> List[Dict[str, Any]]:
    """Reconstruct per-step UI fragments from DS.

    - Adds ds_graph_nodes/edges for each step (up to that step).
    - In 'strict' mode, optionally prunes overlays (timeline/candidate snapshots) so
      UI can rely on DS-only rendering without leakage。
    - Returns a new records list (shallow-copied dicts) to avoid mutating input.
    """
    out: List[Dict[str, Any]] = []
    strict = (str(mode).lower() == "strict")
    for rec in (records or []):
        if not isinstance(rec, dict):
            out.append(rec)
            continue
        step = int(rec.get("step", 0))
        s_nodes, s_edges = load_ds_graph_upto_step(db_path, namespace, step, include_timeline=not strict)
        new_rec = dict(rec)
        if s_nodes or s_edges:
            new_rec["ds_graph_nodes"] = s_nodes
            new_rec["ds_graph_edges"] = s_edges
        if strict:
            # Prune overlays that could cause leakage/混乱（可視化はDSに任せる）
            for k in (
                "timeline_edges",
                "candidate_pool",
                "cand_edges",
                "forced_edges",
                "forced_edges_meta",
                "graph_nodes_eval",
                "graph_edges_eval",
                "graph_nodes_pre",
                "graph_edges_pre",
            ):
                if k in new_rec:
                    try:
                        del new_rec[k]
                    except Exception:
                        pass
            # Also suppress query_node in Strict DS mode so that UI won't render next-Q ring
            if 'query_node' in new_rec:
                try:
                    del new_rec['query_node']
                except Exception:
                    pass
        out.append(new_rec)
    return out
