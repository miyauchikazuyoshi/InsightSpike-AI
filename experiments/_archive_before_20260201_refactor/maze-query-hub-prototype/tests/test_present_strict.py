import json
import sqlite3
from pathlib import Path

import importlib.util as _imp_util
import sys as _sys
from pathlib import Path as _Path

_present_path = _Path(__file__).resolve().parents[1] / "qhlib" / "present.py"
_spec = _imp_util.spec_from_file_location("qh_present", str(_present_path))
assert _spec and _spec.loader, "failed to load present.py"
_mod = _imp_util.module_from_spec(_spec)
_sys.modules["qh_present"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
load_ds_graph_upto_step = getattr(_mod, "load_ds_graph_upto_step")


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE graph_nodes (id TEXT, namespace TEXT, attributes TEXT)")
    cur.execute("CREATE TABLE graph_edges (source_id TEXT, target_id TEXT, namespace TEXT, attributes TEXT)")
    # Two nodes
    cur.execute(
        "INSERT INTO graph_nodes (id, namespace, attributes) VALUES (?, ?, ?)",
        ("1,1,0", "ns", json.dumps({"step": 0})),
    )
    cur.execute(
        "INSERT INTO graph_nodes (id, namespace, attributes) VALUES (?, ?, ?)",
        ("1,2,0", "ns", json.dumps({"step": 0})),
    )
    # One timeline edge, one normal
    cur.execute(
        "INSERT INTO graph_edges (source_id, target_id, namespace, attributes) VALUES (?, ?, ?, ?)",
        ("1,1,0", "1,2,0", "ns", json.dumps({"step": 0, "edge_type": "timeline"})),
    )
    cur.execute(
        "INSERT INTO graph_edges (source_id, target_id, namespace, attributes) VALUES (?, ?, ?, ?)",
        ("1,2,0", "1,1,0", "ns", json.dumps({"step": 0, "edge_type": "graph"})),
    )
    conn.commit()
    conn.close()


def test_load_ds_graph_upto_step_excludes_timeline(tmp_path: Path) -> None:
    db = tmp_path / "ds.sqlite"
    _init_db(db)
    # include_timeline=False should drop timeline edges
    nodes, edges = load_ds_graph_upto_step(str(db), "ns", step=0, include_timeline=False)
    assert nodes and isinstance(nodes[0], list)
    # expect only one non-timeline edge
    assert len(edges) == 1
    # include_timeline=True returns both
    nodes2, edges2 = load_ds_graph_upto_step(str(db), "ns", step=0, include_timeline=True)
    assert len(edges2) == 2
