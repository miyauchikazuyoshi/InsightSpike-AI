"""Emit an equal-resources table (CSV and Markdown) for a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit equal-resources table")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = args.config.read_text(encoding="utf-8")
    js = _load_json(args.results)

    # Extract a few fields; config is YAML but we can show as text snippet
    c = js.get("config", {})
    rows = [
        ("dataset", str(c.get("dataset"))),
        ("num_queries", str(c.get("num_queries"))),
        ("embedding_model", "see YAML: embedding.model"),
        ("top_k", "see YAML: retrieval.top_k"),
        ("bm25_weight", "see YAML: retrieval.bm25_weight"),
        ("embedding_weight", "see YAML: retrieval.embedding_weight"),
        ("lambda", "see YAML: gedig.lambda"),
        ("use_multihop", "see YAML: gedig.use_multihop"),
        ("max_hops", "see YAML: gedig.max_hops"),
        ("theta_ag", "see YAML: gedig.theta_ag"),
        ("theta_dg", "see YAML: gedig.theta_dg"),
    ]

    out = args.out or args.results.with_name(args.results.stem + "_resources.md")
    with out.open("w", encoding="utf-8") as fh:
        fh.write("| Key | Value |\n|---|---|\n")
        for k, v in rows:
            fh.write(f"| {k} | {v} |\n")
        fh.write("\n\n<details><summary>Raw YAML config</summary>\n\n")
        fh.write("```yaml\n")
        fh.write(cfg)
        fh.write("\n```\n\n</details>\n")

    print(f"Wrote resources table to {out}")


if __name__ == "__main__":
    main()

