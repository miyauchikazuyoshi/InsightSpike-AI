#!/usr/bin/env python3
"""Generate figures from latest comparison CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from latest comparison")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "latest_comparison.csv",
        help="Path to latest_comparison.csv",
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

    rows = load_rows(args.input)
    if not rows:
        print("[warn] No rows found in latest comparison.")
        return

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("data", "unknown"), []).append(row)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for data_name, data_rows in grouped.items():
        methods = [r["method"] for r in data_rows]
        em_vals = [float(r["em"]) if r.get("em") not in (None, "") else 0.0 for r in data_rows]
        f1_vals = [float(r["f1"]) if r.get("f1") not in (None, "") else 0.0 for r in data_rows]

        x = range(len(methods))
        width = 0.4
        plt.figure(figsize=(8, 4))
        plt.bar([i - width / 2 for i in x], em_vals, width=width, label="EM")
        plt.bar([i + width / 2 for i in x], f1_vals, width=width, label="F1")
        plt.xticks(list(x), methods, rotation=15, ha="right")
        plt.ylabel("Score")
        plt.title(f"Latest EM/F1 ({data_name})")
        plt.legend()
        plt.tight_layout()
        out_path = output_dir / f"latest_comparison_{Path(data_name).stem}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[done] Wrote {out_path}")


if __name__ == "__main__":
    main()
