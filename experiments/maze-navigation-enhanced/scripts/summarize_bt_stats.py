#!/usr/bin/env python3
"""Summarize BT statistics from CSV logs."""

import argparse
import csv
from pathlib import Path
from statistics import mean


def summarize(csv_path):
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            # convert types
            converted = {}
            for k, v in row.items():
                if k == 'goal':
                    converted[k] = v.lower() == 'true'
                else:
                    try:
                        converted[k] = float(v)
                    except ValueError:
                        converted[k] = v
            rows.append(converted)
    groups = {}
    for row in rows:
        size = int(row['size'])
        groups.setdefault(size, []).append(row)
    return groups


def produce_table(groups, out_tex=None):
    header = "size & success rate & steps & edges & BT count & coverage & gedig$_{min}$\\\\"
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule", "Maze size & Success & Steps & Edges & BT & Coverage & $g_{\min}$ \\ ", "\\midrule"]
    for size in sorted(groups):
        group = groups[size]
        n = len(group)
        success = sum(1 for r in group if r['goal']) / n
        avg_steps = mean(r['steps'] for r in group)
        avg_edges = mean(r['edges'] for r in group)
        avg_bt = mean(r.get('bt_triggers', 0.0) for r in group)
        avg_cov = mean(r.get('coverage', 0.0) for r in group)
        avg_gmin = mean(r.get('gedig_min', 0.0) for r in group)
        lines.append(f"{size} & {success:.2f} & {avg_steps:.1f} & {avg_edges:.0f} & {avg_bt:.1f} & {avg_cov:.3f} & {avg_gmin:.3f} \\ ")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    table = "\n".join(lines)
    if out_tex:
        out_tex.write_text(table)
    return table


def main():
    parser = argparse.ArgumentParser(description="Summarize BT statistics")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    groups = summarize(args.csv)
    table = produce_table(groups, args.output)
    print(table)
    if args.output:
        print(f"Saved LaTeX table to {args.output}")


if __name__ == "__main__":
    main()
