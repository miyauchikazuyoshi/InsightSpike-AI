from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

Node = Tuple[int, int, int]


def _node_to_id(n: Node | Sequence[int]) -> str:
    if isinstance(n, (list, tuple)) and len(n) >= 3:
        return f"{int(n[0])},{int(n[1])},{int(n[2])}"
    raise ValueError("invalid node id")


def _id_to_node(s: str) -> Node:
    parts = [int(p) for p in s.split(',')]
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    raise ValueError("invalid id string")


@dataclass
class PairRecord:
    u_id: str
    v_id: str
    d_before: float


@dataclass
class Pairset:
    signature: str
    lb_avg: float
    pairs: List[PairRecord]
    node_count: int
    edge_count: int
    scope: str
    boundary: str
    eff_hop: int
    meta: Dict[str, Any] = field(default_factory=dict)


class SPPairsetService:
    def load(self, signature: str) -> Optional[Pairset]:
        raise NotImplementedError

    def save(self, pairset: Pairset) -> None:
        raise NotImplementedError

    def upsert(self, pairset: Pairset) -> None:
        return self.save(pairset)

    def stats(self) -> Dict[str, Any]:
        return {}


class InMemoryPairsetService(SPPairsetService):
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = int(max(1, capacity))
        self._data: Dict[str, Pairset] = {}
        self._hits = 0
        self._miss = 0

    def load(self, signature: str) -> Optional[Pairset]:
        ps = self._data.get(signature)
        if ps is None:
            self._miss += 1
        else:
            self._hits += 1
        return ps

    def save(self, pairset: Pairset) -> None:
        if len(self._data) >= self.capacity and pairset.signature not in self._data:
            # naive eviction: pop arbitrary
            self._data.pop(next(iter(self._data)))
        self._data[pairset.signature] = pairset

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._miss
        return {
            "size": len(self._data),
            "capacity": self.capacity,
            "hits": self._hits,
            "misses": self._miss,
            "hit_rate": (self._hits / total) if total else 0.0,
        }


class SQLitePairsetService(SPPairsetService):
    def __init__(self, db_path: str, namespace: str) -> None:
        self.db_path = str(db_path)
        self.namespace = str(namespace)
        self._ensure()
        self._hits = 0
        self._miss = 0

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure(self) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sp_pairsets (
                  signature TEXT PRIMARY KEY,
                  namespace TEXT NOT NULL,
                  scope TEXT,
                  boundary TEXT,
                  eff_hop INTEGER,
                  node_count INTEGER,
                  edge_count INTEGER,
                  lb_avg REAL,
                  pair_count INTEGER,
                  meta TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sp_pairs (
                  signature TEXT NOT NULL,
                  idx INTEGER NOT NULL,
                  u_id TEXT NOT NULL,
                  v_id TEXT NOT NULL,
                  d_before REAL,
                  PRIMARY KEY(signature, idx)
                )
                """
            )
            conn.commit()

    def load(self, signature: str) -> Optional[Pairset]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT scope,boundary,eff_hop,node_count,edge_count,lb_avg,pair_count,meta,namespace FROM sp_pairsets WHERE signature=?", (signature,))
            row = cur.fetchone()
            if not row:
                self._miss += 1
                return None
            scope, boundary, eff_hop, node_count, edge_count, lb_avg, pair_count, meta_json, ns = row
            if ns != self.namespace:
                # namespace mismatch -> miss
                self._miss += 1
                return None
            cur.execute("SELECT idx,u_id,v_id,d_before FROM sp_pairs WHERE signature=? ORDER BY idx ASC", (signature,))
            pairs: List[PairRecord] = []
            for idx, u_id, v_id, d in cur.fetchall():
                pairs.append(PairRecord(u_id=u_id, v_id=v_id, d_before=float(d)))
            self._hits += 1
            return Pairset(
                signature=signature,
                lb_avg=float(lb_avg),
                pairs=pairs,
                node_count=int(node_count),
                edge_count=int(edge_count),
                scope=str(scope or ''),
                boundary=str(boundary or ''),
                eff_hop=int(eff_hop or 0),
                meta=json.loads(meta_json or '{}'),
            )

    def save(self, pairset: Pairset) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "REPLACE INTO sp_pairsets(signature,namespace,scope,boundary,eff_hop,node_count,edge_count,lb_avg,pair_count,meta) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    pairset.signature,
                    self.namespace,
                    pairset.scope,
                    pairset.boundary,
                    int(pairset.eff_hop),
                    int(pairset.node_count),
                    int(pairset.edge_count),
                    float(pairset.lb_avg),
                    int(len(pairset.pairs)),
                    json.dumps(pairset.meta or {}),
                ),
            )
            cur.execute("DELETE FROM sp_pairs WHERE signature=?", (pairset.signature,))
            for idx, pr in enumerate(pairset.pairs):
                cur.execute(
                    "INSERT INTO sp_pairs(signature,idx,u_id,v_id,d_before) VALUES (?,?,?,?,?)",
                    (pairset.signature, idx, pr.u_id, pr.v_id, float(pr.d_before)),
                )
            conn.commit()

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._miss
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM sp_pairsets")
            n_sets = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM sp_pairs")
            n_pairs = int(cur.fetchone()[0])
        return {
            "pairsets": n_sets,
            "pairs": n_pairs,
            "hits": self._hits,
            "misses": self._miss,
            "hit_rate": (self._hits / total) if total else 0.0,
        }


class SignatureBuilder:
    def __init__(self) -> None:
        pass

    def for_subgraph(
        self,
        g: nx.Graph,
        anchors_core: Sequence[Node],
        eff_hop: int,
        scope: str,
        boundary: str,
    ) -> Tuple[str, Dict[str, Any]]:
        nodes = sorted(_node_to_id(n) for n in g.nodes())
        edges: List[str] = []
        for u, v in g.edges():
            su = _node_to_id(u); sv = _node_to_id(v)
            if su <= sv:
                edges.append(f"{su}|{sv}")
            else:
                edges.append(f"{sv}|{su}")
        edges.sort()
        nac = sorted(_node_to_id(a) for a in anchors_core)
        payload = f"N:{' '.join(nodes)}#E:{' '.join(edges)}#A:{' '.join(nac)}#H:{int(eff_hop)}#S:{scope}#B:{boundary}"
        h = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        meta = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "anchors": nac,
            "eff_hop": int(eff_hop),
            "scope": scope,
            "boundary": boundary,
        }
        return h, meta

