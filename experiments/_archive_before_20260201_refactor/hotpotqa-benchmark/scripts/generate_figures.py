#!/usr/bin/env python3
"""Generate simple figures from aggregated results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from summary_by_method.json")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "summary_by_method.json",
        help="Path to summary_by_method.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "figures",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install matplotlib: pip install matplotlib")

    rows = load_summary(args.summary)
    if not rows:
        print("[warn] No rows found in summary.")
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [row.get("method", "unknown") for row in rows]

    em = [row.get("em") for row in rows]
    f1 = [row.get("f1") for row in rows]
    if any(v is not None for v in em + f1):
        x = range(len(methods))
        width = 0.4
        plt.figure(figsize=(8, 4))
        plt.bar([i - width / 2 for i in x], em, width=width, label="EM")
        plt.bar([i + width / 2 for i in x], f1, width=width, label="F1")
        plt.xticks(list(x), methods, rotation=15, ha="right")
        plt.ylabel("Score")
        plt.title("EM/F1 by Method")
        plt.legend()
        plt.tight_layout()
        out_path = output_dir / "em_f1.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[done] Wrote {out_path}")

    latency = [row.get("latency_p50_ms") for row in rows]
    if any(v is not None for v in latency):
        x = range(len(methods))
        plt.figure(figsize=(8, 4))
        plt.bar(list(x), latency)
        plt.xticks(list(x), methods, rotation=15, ha="right")
        plt.ylabel("Latency p50 (ms)")
        plt.title("Latency p50 by Method")
        plt.tight_layout()
        out_path = output_dir / "latency_p50.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[done] Wrote {out_path}")


if __name__ == "__main__":
    main()
