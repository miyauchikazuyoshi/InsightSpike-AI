#!/usr/bin/env python3
"""
Generate a tiny baseline comparison (Static RAG vs geDIG) from existing lite experiment
results under experiments/exp2to4_lite/results, and optionally update docs/phase1.md
with a small Markdown table.

Usage:
  python scripts/run_baseline_compare.py \
    --results experiments/exp2to4_lite/results/exp23_lite_*.json \
    --out results/baseline/baseline_compare.json \
    --update-docs

Notes:
  - Prefers the most recent exp23_lite_*.json if --results is not provided.
  - Does not run any experiments; reads already-produced JSON to remain cloud-safe.
  - Updates docs/phase1.md between markers:
      <!-- BASELINE:BEGIN --> ... <!-- BASELINE:END -->
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


DEFAULT_RESULTS_GLOB = "experiments/exp2to4_lite/results/exp23_lite_*.json"
DEFAULT_OUT = "results/baseline/baseline_compare.json"
DOCS_PHASE1 = Path("docs/phase1.md")


@dataclass
class Metrics:
    acceptance_rate: float
    fmr: float
    latency_p50: float

    @classmethod
    def from_dict(cls, d: Dict) -> "Metrics":
        return cls(
            acceptance_rate=float(d.get("acceptance_rate", 0.0)),
            fmr=float(d.get("fmr", 0.0)),
            latency_p50=float(d.get("latency_p50", 0.0)),
        )


def pick_latest(path_glob: str) -> Optional[Path]:
    files = sorted(glob.glob(path_glob))
    return Path(files[-1]) if files else None


def load_compare(results_path: Path) -> Dict[str, Metrics]:
    data = json.loads(results_path.read_text())
    res = data.get("results", {})
    if not isinstance(res, dict):
        raise SystemExit(f"Unexpected results format in {results_path}")
    try:
        static = Metrics.from_dict(res["static_rag"])  # naive baseline
        gedig = Metrics.from_dict(res["gedig_ag_dg"])  # geDIG
    except KeyError as e:
        raise SystemExit(f"Missing key in results: {e} in {results_path}")
    return {"Static RAG": static, "geDIG": gedig}


def ensure_out_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def save_json(out_path: Path, payload: Dict) -> None:
    ensure_out_dir(out_path)
    out_path.write_text(json.dumps(payload, indent=2))


def fmt_rate(x: float) -> str:
    return f"{x:.3f}"


def fmt_ms(x: float) -> str:
    return f"{x:.0f} ms"


def render_table(static: Metrics, gedig: Metrics) -> str:
    lines = []
    lines.append("| Metric | Static RAG | geDIG |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Acceptance | {fmt_rate(static.acceptance_rate)} | {fmt_rate(gedig.acceptance_rate)} |")
    lines.append(f"| FMR | {fmt_rate(static.fmr)} | {fmt_rate(gedig.fmr)} |")
    lines.append(f"| Latency P50 | {fmt_ms(static.latency_p50)} | {fmt_ms(gedig.latency_p50)} |")
    return "\n".join(lines) + "\n"


def update_docs_phase1(table_md: str) -> None:
    if not DOCS_PHASE1.exists():
        print(f"[warn] docs/phase1.md not found; skipping docs update")
        return
    begin = "<!-- BASELINE:BEGIN -->"
    end = "<!-- BASELINE:END -->"
    text = DOCS_PHASE1.read_text()
    block = (
        begin
        + "\n\n### Baseline Comparison (Static RAG vs geDIG)\n\n"
        + table_md
        + "\n"
        + end
    )
    if begin in text and end in text:
        new = text.split(begin)[0] + block + text.split(end, 1)[1]
    else:
        # Append to end with a separator
        new = text.rstrip() + "\n\n" + block + "\n"
    DOCS_PHASE1.write_text(new)
    print(f"[ok] Updated docs/phase1.md baseline table")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default=DEFAULT_RESULTS_GLOB, help="Path or glob to exp23_lite JSON")
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT), help="Output JSON path")
    ap.add_argument("--update-docs", action="store_true", help="Update docs/phase1.md table")
    args = ap.parse_args()

    rp: Optional[Path]
    if any(ch in args.results for ch in "*?[]"):
        rp = pick_latest(args.results)
        if rp is None:
            raise SystemExit(f"No results matched: {args.results}")
    else:
        rp = Path(args.results)
        if not rp.exists():
            raise SystemExit(f"Results file not found: {rp}")

    compare = load_compare(rp)
    payload = {
        "source": str(rp),
        "static_rag": compare["Static RAG"].__dict__,
        "gedig": compare["geDIG"].__dict__,
    }
    save_json(args.out, payload)
    print(f"[ok] Wrote {args.out}")

    if args.update_docs:
        table_md = render_table(compare["Static RAG"], compare["geDIG"]) 
        update_docs_phase1(table_md)


if __name__ == "__main__":
    main()

