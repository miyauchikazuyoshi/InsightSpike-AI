"""Format ablation JSON bundle into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation JSON -> Markdown table")
    ap.add_argument("--ablations", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = _load_json(args.ablations)
    rows = []
    for key, payload in data.items():
        # payload is a result JSON with results per baseline; pull geDIG and/or the first entry
        results = payload.get("results", {})
        # Prefer gedig_ag_dg, else any
        name = "gedig_ag_dg" if "gedig_ag_dg" in results else (list(results.keys())[0] if results else None)
        if not name:
            continue
        stats = results[name]
        rows.append(
            {
                "variant": key,
                "per_mean": float(stats.get("per_mean", 0.0)),
                "acceptance_rate": float(stats.get("acceptance_rate", 0.0)),
                "fmr": float(stats.get("fmr", 1.0)),
                "latency_p50": float(stats.get("latency_p50", 0.0)),
                "latency_p95": float(stats.get("latency_p95", 0.0)),
            }
        )

    out = args.out or args.ablations.with_suffix(".md")
    with out.open("w", encoding="utf-8") as fh:
        fh.write("| variant | per_mean | acceptance | fmr | lat_p50 | lat_p95 |\n|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            fh.write(
                f"| {r['variant']} | {r['per_mean']:.4f} | {r['acceptance_rate']:.3f} | {r['fmr']:.3f} | {r['latency_p50']:.1f} | {r['latency_p95']:.1f} |\n"
            )
    print(f"Wrote ablation markdown: {out}")


if __name__ == "__main__":
    main()

