#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple


def _iter_transforms(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        # benchmark_results.json style
        for row in payload:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "unknown")
            cost = float(row.get("transform_cost", 0.0))
            yield {
                "domain": "graph_pattern",
                "task": "benchmark_results_json",
                "source": {"id": f"benchmark:{name}:source"},
                "target": {"id": f"benchmark:{name}:target"},
                "transform": {"cost": cost},
            }
        return

    if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        # novel_analogies.json style
        for cand in payload["candidates"]:
            if not isinstance(cand, dict):
                continue
            src = str(cand.get("source") or "unknown")
            tgt = str(cand.get("target") or "unknown")
            cost = float(cand.get("transform_cost", 0.0))
            yield {
                "domain": "graph_pattern",
                "task": "novel_analogies_json",
                "source": {"id": f"kb:{src}"},
                "target": {"id": f"kb:{tgt}"},
                "transform": {"cost": cost},
            }
        return

    raise SystemExit(f"Unsupported input format: {path}")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sleep triplets from graph-pattern transform dumps.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Transform dumps (.jsonl/.json).")
    parser.add_argument("--out-dir", type=Path, default=Path("results/sleep/graph_pattern"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs: List[Tuple[str, str, float]] = []

    for path in args.input:
        for row in _iter_transforms(path):
            src = str(((row.get("source") or {}).get("id")) or "")
            tgt = str(((row.get("target") or {}).get("id")) or "")
            cost = float(((row.get("transform") or {}).get("cost")) or 0.0)
            if not src or not tgt:
                continue
            pairs.append((src, tgt, cost))

    adj: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
    for a, b, cost in pairs:
        adj[a].append((b, cost))
        adj[b].append((a, cost))

    triplets: List[Dict[str, Any]] = []
    for anchor, nbrs in adj.items():
        nbrs_sorted = sorted(nbrs, key=lambda x: x[1])
        if len(nbrs_sorted) < 2:
            continue
        pos_id, pos_cost = nbrs_sorted[0]

        # hard negative = nearest "almost" neighbor that is not the best one
        neg_id = None
        neg_cost = None
        for cand_id, cand_cost in nbrs_sorted[1:]:
            if cand_id == pos_id:
                continue
            neg_id, neg_cost = cand_id, float(cand_cost)
            break
        if neg_id is None or neg_cost is None:
            continue

        triplets.append(
            {
                "domain": "graph_pattern",
                "type": "transform_triplet",
                "anchor_id": anchor,
                "positive_id": pos_id,
                "hard_negative_id": neg_id,
                "cost_positive": float(pos_cost),
                "cost_negative": float(neg_cost),
            }
        )

    out_dir = args.out_dir
    n_trip = _write_jsonl(out_dir / "triplets.jsonl", triplets)
    n_pairs = _write_jsonl(out_dir / "pairs.jsonl", ({"source_id": a, "target_id": b, "cost": c} for a, b, c in pairs))

    stats = {
        "inputs": [str(p) for p in args.input],
        "counts": {"pairs": n_pairs, "anchors": len(adj), "triplets": n_trip},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

