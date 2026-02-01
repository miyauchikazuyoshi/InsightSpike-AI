#!/usr/bin/env python3
"""
Simple builder to generate interactive/report HTML for maze query-hub experiments.

It reads the summary JSON (produced by run_experiment_query.py --output)
and the step log JSON (--step-log), bundles them into a single experimentData
object, and injects it into query_interactive_template.html.

Usage:
  python experiments/maze-query-hub-prototype/build_reports.py \
    --summary experiments/maze-query-hub-prototype/results/xxx_summary.json \
    --steps   experiments/maze-query-hub-prototype/results/xxx_steps.json \
    --out     experiments/maze-query-hub-prototype/results/xxx_interactive.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from importlib import import_module
import importlib.util as _imp_util
import sys as _sys

# Optional metrics for maze exploration効率（ファイルパス経由で安全に読み込む）
extract_exploration_metrics = None  # type: ignore
detect_backtracks = None  # type: ignore
summarize_backtrack = None  # type: ignore
summarize_gates = None  # type: ignore
try:  # best-effort: load qhlib/metrics.py by path (hyphenated package name prevents dotted import)
    _metrics_path = (Path(__file__).with_name("qhlib") / "metrics.py").resolve()
    if _metrics_path.exists():
        _spec = _imp_util.spec_from_file_location("qh_metrics", str(_metrics_path))
        if _spec and _spec.loader:
            _mod = _imp_util.module_from_spec(_spec)
            _sys.modules["qh_metrics"] = _mod
            _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
            extract_exploration_metrics = getattr(_mod, "extract_exploration_metrics", None)
            detect_backtracks = getattr(_mod, "detect_backtracks", None)
            summarize_backtrack = getattr(_mod, "summarize_backtrack", None)
            summarize_gates = getattr(_mod, "summarize_gates", None)
except Exception:
    pass


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _lighten_steps(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a lightweight copy of step records for HTML embedding.
    Retain DS snapshotsと最小メタのみ。重いスナップショットは削除する。
    """
    drop_keys = {
        # Graph snapshots (eval/pre/committed diff)
        'graph_nodes_preselect', 'graph_edges_preselect',
        'graph_nodes_pre', 'graph_edges_pre',
        'graph_nodes_eval', 'graph_edges_eval',
        'committed_only_nodes', 'committed_only_edges', 'committed_only_edges_meta',
        # Diagnostic pools
        'candidate_pool', 'cand_edges', 'forced_edges', 'forced_edges_meta',
        'sp_diagnostics', 'debug_hop0', 'graph_nodes_post', 'graph_edges_post',
        # Post eval hop series (rarely used in HTML)
        'hop_series_post',
    }
    out: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            out.append(rec)
            continue
        lite = {k: v for k, v in rec.items() if k not in drop_keys}
        out.append(lite)
    return out


def assemble_experiment_data(
    summary: Dict[str, Any],
    steps: List[Dict[str, Any]],
    *,
    ds_nodes: List[List[int]] | None = None,
    ds_edges: List[List[List[int]]] | None = None,
    ui_defaults: Dict[str, Any] | None = None,
    light_steps: bool = False,
) -> Dict[str, Any]:
    # Derive pre/post query nodes for legacy logs (so HTML can use pre-step Q without rerun)
    try:
        by_seed: Dict[str, List[Dict[str, Any]]] = {}
        for rec in steps:
            by_seed.setdefault(str(rec.get('seed', 0)), []).append(rec)
        for seed, recs in by_seed.items():
            recs.sort(key=lambda r: int(r.get('step', 0)))
            prev_pos = None
            for idx, rec in enumerate(recs):
                pos = rec.get('position')
                # post-step (for reference)
                if isinstance(pos, list) and len(pos) >= 2:
                    rec.setdefault('query_node_post', [int(pos[0]), int(pos[1]), -1])
                # pre-step = previous record's position if available
                if prev_pos is not None and isinstance(prev_pos, list) and len(prev_pos) >= 2:
                    rec.setdefault('query_node_pre_derived', [int(prev_pos[0]), int(prev_pos[1]), -1])
                # If still missing, try derive from ds_edges_saved (timeline at this step)
                if 'query_node_pre_derived' not in rec:
                    try:
                        saved = rec.get('ds_edges_saved') or []
                        for em in saved:
                            stage = (em.get('edge_type') or em.get('stage') or '').lower()
                            if stage != 'timeline':
                                continue
                            su = em.get('source'); sv = em.get('target')
                            def parse_node(s):
                                if not isinstance(s, str): return None
                                try:
                                    parts = [int(p) for p in s.split(',')]
                                    return parts if len(parts) >= 3 else None
                                except Exception:
                                    return None
                            au = parse_node(su); bv = parse_node(sv)
                            for node in (au, bv):
                                if node and len(node) >= 3 and int(node[2]) == -1:
                                    rec['query_node_pre_derived'] = [int(node[0]), int(node[1]), -1]
                                    break
                            if 'query_node_pre_derived' in rec:
                                break
                    except Exception:
                        pass
                # If still missing, try derive from ds_graph_edges endpoints (prefer dir=-1, nearest to post-step)
                if 'query_node_pre_derived' not in rec:
                    try:
                        edges = rec.get('ds_graph_edges') or []
                        q_candidates = []
                        for e in edges:
                            if not (isinstance(e, list) and len(e) == 2):
                                continue
                            for node in e:
                                if isinstance(node, list) and len(node) >= 3 and int(node[2]) == -1:
                                    q_candidates.append([int(node[0]), int(node[1]), -1])
                        if q_candidates:
                            # pick closest to post-step position
                            if isinstance(pos, list) and len(pos) >= 2:
                                def md(n):
                                    return abs(int(n[0]) - int(pos[0])) + abs(int(n[1]) - int(pos[1]))
                                q_candidates.sort(key=md)
                            rec['query_node_pre_derived'] = q_candidates[0]
                    except Exception:
                        pass
                prev_pos = pos
            # For first step, if no derived pre exists, fall back to rec.query_node or rec.query_node_post
            if recs:
                r0 = recs[0]
                if 'query_node_pre_derived' not in r0:
                    qn = r0.get('query_node') or r0.get('query_node_post') or r0.get('position')
                    if isinstance(qn, list) and len(qn) >= 2:
                        r0['query_node_pre_derived'] = [int(qn[0]), int(qn[1]), -1]
    except Exception:
        pass
    # Group steps by seed
    seed_data: Dict[str, Dict[str, Any]] = {}
    for rec in steps:
        seed = str(rec.get("seed", 0))
        seed_entry = seed_data.setdefault(seed, {"records": []})
        seed_entry["records"].append(rec)
    # Attach per-seed run summaries and counts if available
    try:
        runs = summary.get('runs') or []
        for run in runs:
            seed = str(run.get('seed', 0))
            entry = seed_data.setdefault(seed, {"records": []})
            entry["run_summary"] = run
            # Expose total steps and a simple positive‑k indicator if present
            entry["total_steps"] = int(run.get('steps', len(entry.get('records') or [])))
    except Exception:
        pass

    out = {
        "config": summary.get("config", {}),
        "summary": summary.get("summary", {}),
        "runs": summary.get("runs", []),
        "maze_data": summary.get("maze_data", {}),
        "seed_data": seed_data,
    }
    # Inject UI defaults so template can initialize toggles/state
    if ui_defaults:
        out["config"]["ui_defaults"] = ui_defaults
    if ds_nodes is not None or ds_edges is not None:
        out["ds_graph"] = {
            "nodes": ds_nodes or [],
            "edges": ds_edges or [],
        }
    # Ensure graph_mode present for template rendering
    if "graph_mode" not in out["config"]:
        out["config"]["graph_mode"] = "query_hub"
    # Optionally lighten step payload for HTML embedding only
    if light_steps:
        for seed, entry in out.get("seed_data", {}).items():
            recs = entry.get("records") or []
            entry["records"] = _lighten_steps(recs)
    return out


def inject_into_template(template_path: Path, out_path: Path, payload: Dict[str, Any]) -> None:
    import re
    html = template_path.read_text(encoding="utf-8")
    data_blob = json.dumps(payload, ensure_ascii=False)
    # Replace only the RHS of the const assignment, preserving the rest of the script
    pattern = re.compile(r"(const\s+experimentData\s*=\s*)(.*?)(;)", re.DOTALL)
    if pattern.search(html):
        injected = pattern.sub(rf"\1{data_blob}\3", html, count=1)
    else:
        # Fallback: inject a small script defining experimentData near the top
        injected = html.replace(
            "<script>", f"<script>\nconst experimentData = {data_blob};\n",
            1,
        )
    out_path.write_text(injected, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=Path(__file__).with_name("query_interactive_template.html"))
    ap.add_argument("--sqlite", type=Path, default=None, help="Optional SQLite DB path to load DS graph for UI")
    ap.add_argument("--namespace", type=str, default="maze_query_hub", help="Namespace for DS graph in SQLite")
    ap.add_argument("--strict", action="store_true", help="Prefer DS graph in HTML and default Strict DS ON")
    ap.add_argument("--relaxed", action="store_true", help="Prefer step snapshots/overlays and default Strict DS OFF")
    ap.add_argument("--light-steps", action="store_true", help="Embed lightweight step records in HTML (DSで復元する前提)")
    ap.add_argument("--present-mode", type=str, default="none", choices=["none","strict","relaxed"], help="Reconstruct per-step UI fragments from DS with present layer")
    args = ap.parse_args()

    summary_raw = load_json(args.summary)
    # When reading summary from run_experiment_query.py, the JSON root is the whole payload
    # and matches expected keys; pass through
    summary = summary_raw
    steps = load_json(args.steps)

    # Enrich summary with exploration/backtrack/gating metrics if possible
    try:
        maze_size = None
        if isinstance(summary, dict):
            maze_size = (summary.get("config") or {}).get("maze_size")
            if not maze_size:
                md = summary.get("maze_data") or {}
                if isinstance(md, dict) and md:
                    # pick first seed entry
                    any_seed = next(iter(md.values()))
                    size = (any_seed or {}).get("size")
                    if isinstance(size, list) and len(size) >= 2:
                        maze_size = int(size[0])  # assume square
        if extract_exploration_metrics and maze_size:
            ee = extract_exploration_metrics(steps, int(maze_size))
            bt_segments = detect_backtracks(steps) if detect_backtracks else []
            bt = summarize_backtrack(bt_segments) if summarize_backtrack else {}
            gs = summarize_gates(steps) if summarize_gates else {}
            # Attach under summary.summary
            if isinstance(summary, dict):
                summary.setdefault("summary", {})
                summary["summary"]["exploration_efficiency"] = ee
                summary["summary"]["backtrack_analysis"] = bt
                summary["summary"]["gate_statistics"] = gs
    except Exception:
        pass
    ds_nodes = None
    ds_edges = None
    if args.sqlite is not None:
        try:
            # Try dotted import (may fail due to hyphen in path); fallback to path import
            try:
                present = import_module('experiments.maze-query-hub-prototype.qhlib.present')
            except Exception:
                _present_path = (Path(__file__).with_name("qhlib") / "present.py").resolve()
                if _present_path.exists():
                    _spec = _imp_util.spec_from_file_location("qh_present", str(_present_path))
                    if _spec and _spec.loader:
                        _modp = _imp_util.module_from_spec(_spec)
                        _sys.modules["qh_present"] = _modp
                        _spec.loader.exec_module(_modp)  # type: ignore[attr-defined]
                        present = _modp
                else:
                    raise ImportError("present.py not found for DS reconstruction")
            # Optionally reconstruct per-step fragments from DS (present layer)
            if args.present_mode and args.present_mode != 'none':
                steps = present.reconstruct_records(str(args.sqlite), args.namespace, steps, mode=args.present_mode)
            else:
                # Enrich each step with DS snapshot up to that step
                for rec in steps:
                    if not isinstance(rec, dict):
                        continue
                    step_idx = int(rec.get('step', 0))
                    s_nodes, s_edges = present.load_ds_graph_upto_step(str(args.sqlite), args.namespace, step_idx)
                    if s_nodes or s_edges:
                        rec['ds_graph_nodes'] = s_nodes
                        rec['ds_graph_edges'] = s_edges
            # Also provide a global DS fallback snapshot
            ds_nodes, ds_edges = present.load_ds_graph(str(args.sqlite), args.namespace)
        except Exception:
            ds_nodes, ds_edges = None, None

    # Decide UI defaults
    ui_defaults = None
    if args.strict and not args.relaxed:
        ui_defaults = {"dsGraph": True, "dsStrict": True, "timeline": False, "evalGraph": False, "useMhOnly": True, "showAllQueries": False}
    elif args.relaxed and not args.strict:
        ui_defaults = {"dsGraph": False, "dsStrict": False, "timeline": True, "evalGraph": False, "useMhOnly": True, "showAllQueries": False}
    payload = assemble_experiment_data(
        summary,
        steps,
        ds_nodes=ds_nodes,
        ds_edges=ds_edges,
        ui_defaults=ui_defaults,
        light_steps=bool(args.light_steps),
    )
    # Ensure a sensible default seed is selected (one that has records) to avoid a blank view
    try:
        sd = payload.get("seed_data") or {}
        first_with_records = None
        for k, entry in sd.items():
            recs = entry.get("records") or []
            if recs:
                first_with_records = k
                break
        if first_with_records is not None:
            cfg = payload.setdefault("config", {})
            ui = cfg.setdefault("ui_defaults", {})
            ui.setdefault("selectedSeed", str(first_with_records))
    except Exception:
        pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inject_into_template(args.template, args.out, payload)
    print(f"Wrote interactive HTML: {args.out}")


if __name__ == "__main__":
    main()
