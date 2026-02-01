#!/usr/bin/env python3
"""
Select seeds by AG activity from a steps.json produced by run_experiment_query.py.

Usage:
  PYTHONPATH=src python experiments/maze-query-hub-prototype/tools/select_seeds_by_ag.py \
    --steps experiments/maze-query-hub-prototype/results/l3_fast/_51x51_probe_steps.json \
    --top 5 --out experiments/maze-query-hub-prototype/results/l3_fast/_51x51_probe_topseeds.json

Outputs a JSON with sorted seeds by AG count (desc), and basic timing stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_steps(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        out: List[Dict[str, Any]] = []
        for v in data.values():
            if isinstance(v, list):
                out.extend([x for x in v if isinstance(x, dict)])
        return out
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Select seeds by AG activity")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    steps = load_steps(args.steps)
    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in steps:
        try:
            s = int(r.get("seed", 0))
        except Exception:
            s = 0
        by_seed.setdefault(s, []).append(r)

    rows: List[Tuple[int, int, float, float]] = []
    for seed, recs in by_seed.items():
        ag = sum(1 for r in recs if r.get("ag_fire"))
        times = [float(r.get("time_ms_eval", 0.0)) for r in recs]
        avg = (sum(times) / len(times)) if times else 0.0
        p95 = 0.0
        if times:
            ts = sorted(times)
            idx = int(round(0.95 * (len(ts) - 1)))
            p95 = ts[idx]
        rows.append((seed, ag, avg, p95))

    rows.sort(key=lambda t: (-t[1], -t[3], -t[2], t[0]))
    topn = rows[: max(1, int(args.top))]
    result = {
        "count": len(rows),
        "top": [
            {"seed": s, "ag_count": ag, "avg_time_ms_eval": avg, "p95_time_ms_eval": p95}
            for (s, ag, avg, p95) in topn
        ],
        "all": [
            {"seed": s, "ag_count": ag, "avg_time_ms_eval": avg, "p95_time_ms_eval": p95}
            for (s, ag, avg, p95) in rows
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

