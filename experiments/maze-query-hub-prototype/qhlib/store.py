from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
import json
import sqlite3


class SQLiteStore:
    def __init__(self, db_path: str, namespace: str) -> None:
        self.db_path = str(db_path)
        self.namespace = str(namespace)
        self._ensure()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure(self) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    node_type TEXT,
                    attributes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT,
                    attributes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def insert_nodes(self, nodes_meta: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        saved: List[Dict[str, Any]] = []
        with self._conn() as conn:
            cur = conn.cursor()
            for nm in nodes_meta:
                rid, cid, did = nm.get('node', [None, None, None])
                if rid is None:
                    continue
                node_id = f"{int(rid)},{int(cid)},{int(did)}"
                attrs = {k: v for k, v in nm.items() if k not in ('node',)}
                cur.execute(
                    "INSERT OR IGNORE INTO graph_nodes (id, namespace, node_type, attributes) VALUES (?, ?, ?, ?)",
                    (node_id, self.namespace, nm.get('node_type') or 'auto', json.dumps(attrs))
                )
                saved.append({"id": node_id, **attrs})
            conn.commit()
        return saved

    def insert_edges(self, edges_meta: Iterable[Dict[str, Any]], edge_type_fallback: str = 'auto') -> List[Dict[str, Any]]:
        saved: List[Dict[str, Any]] = []
        with self._conn() as conn:
            cur = conn.cursor()
            for em in edges_meta:
                nodes = em.get('nodes')
                if not (isinstance(nodes, Sequence) and len(nodes) == 2):
                    continue
                u, v = nodes
                su = f"{int(u[0])},{int(u[1])},{int(u[2])}"
                sv = f"{int(v[0])},{int(v[1])},{int(v[2])}"
                eattrs = {k: v for k, v in em.items() if k not in ('nodes',)}
                cur.execute(
                    "INSERT INTO graph_edges (namespace, source_id, target_id, edge_type, attributes) VALUES (?, ?, ?, ?, ?)",
                    (self.namespace, su, sv, eattrs.get('stage') or edge_type_fallback, json.dumps(eattrs))
                )
                saved.append({"source": su, "target": sv, **eattrs})
            conn.commit()
        return saved

    def insert_forced_edges(self, edges_meta: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        saved: List[Dict[str, Any]] = []
        with self._conn() as conn:
            cur = conn.cursor()
            for em in edges_meta:
                nodes = em.get('nodes')
                if not (isinstance(nodes, Sequence) and len(nodes) == 2):
                    continue
                u, v = nodes
                su = f"{int(u[0])},{int(u[1])},{int(u[2])}"
                sv = f"{int(v[0])},{int(v[1])},{int(v[2])}"
                eattrs = {k: v for k, v in em.items() if k not in ('nodes',)}
                cur.execute(
                    "INSERT INTO graph_edges (namespace, source_id, target_id, edge_type, attributes) VALUES (?, ?, ?, ?, ?)",
                    (self.namespace, su, sv, 'forced_candidate', json.dumps(eattrs))
                )
                saved.append({"source": su, "target": sv, **eattrs, "edge_type": "forced_candidate"})
            conn.commit()
        return saved

    def insert_timeline_edges(self, timeline_edges: Iterable[Sequence[Sequence[int]]]) -> List[Dict[str, Any]]:
        saved: List[Dict[str, Any]] = []
        with self._conn() as conn:
            cur = conn.cursor()
            for te in timeline_edges:
                if not (isinstance(te, Sequence) and len(te) == 2):
                    continue
                u, v = te
                su = f"{int(u[0])},{int(u[1])},{int(u[2])}"
                sv = f"{int(v[0])},{int(v[1])},{int(v[2])}"
                eattrs = {"stage": "timeline"}
                cur.execute(
                    "INSERT INTO graph_edges (namespace, source_id, target_id, edge_type, attributes) VALUES (?, ?, ?, ?, ?)",
                    (self.namespace, su, sv, 'timeline', json.dumps(eattrs))
                )
                saved.append({"source": su, "target": sv, **eattrs, "edge_type": "timeline"})
            conn.commit()
        return saved

    def totals(self) -> Tuple[int, int]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM graph_nodes WHERE namespace=?", (self.namespace,))
            n_nodes = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM graph_edges WHERE namespace=?", (self.namespace,))
            n_edges = int(cur.fetchone()[0])
        return n_nodes, n_edges

    def snapshot_graph(self) -> Tuple[List[List[int]], List[List[List[int]]]]:
        """Return all nodes and edges for the namespace as lists usable by UI."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM graph_nodes WHERE namespace=?", (self.namespace,))
            nodes = [[int(p) for p in row[0].split(',')] for row in cur.fetchall()]
            cur.execute("SELECT source_id, target_id FROM graph_edges WHERE namespace=?", (self.namespace,))
            edges = []
            for su, sv in cur.fetchall():
                u = [int(p) for p in su.split(',')]
                v = [int(p) for p in sv.split(',')]
                edges.append([u, v])
        return nodes, edges

