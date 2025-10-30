"""Utility to refresh maze fixed-k★ experiment reports.

Given the summary JSON (`run_experiment.py --output ...`) and step log JSON
(`--step-log ...`), this script injects the new data into the interactive HTML
viewer and regenerates the static HTML report so that they stay in sync with
the latest run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_seed_data(runs: List[Dict[str, Any]], steps: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    run_map: Dict[int, Dict[str, Any]] = {int(run["seed"]): dict(run) for run in runs}
    seed_records: Dict[int, Dict[str, Any]] = {}

    for rec in steps:
        seed = int(rec["seed"])
        entry = seed_records.setdefault(
            seed,
            {
                "records": [],
                "run_summary": run_map.get(seed, {}),
                "k_positive": 0,
                "total_steps": 0,
            },
        )
        entry["records"].append(rec)
        k_star = rec.get("candidate_selection", {}).get("k_star")
        if isinstance(k_star, (int, float)) and k_star >= 1.0:
            entry["k_positive"] += 1
        entry["total_steps"] += 1

    # Ensure seeds without step records still carry run summaries
    for seed, run in run_map.items():
        entry = seed_records.setdefault(
            seed,
            {
                "records": [],
                "k_positive": 0,
                "total_steps": 0,
                "run_summary": run,
            },
        )
        entry.setdefault("run_summary", run)
        entry.setdefault("records", [])
        entry["records"].sort(key=lambda item: int(item.get("step", 0)))
        entry["total_steps"] = len(entry["records"])

    return {str(seed): value for seed, value in sorted(seed_records.items())}, seed_records


def update_interactive_html(html_path: Path, experiment_data: Dict[str, Any]) -> None:
    if not html_path.exists():
        raise FileNotFoundError(f"Interactive HTML not found: {html_path}")

    text = html_path.read_text(encoding="utf-8")
    payload = json.dumps(experiment_data, ensure_ascii=False)
    pattern = re.compile(r"const experimentData = (.*?);", re.S)
    new_text, count = pattern.subn(f"const experimentData = {payload};", text, count=1)
    if count == 0:
        raise RuntimeError("Failed to inject experiment data into interactive HTML.")
    html_path.write_text(new_text, encoding="utf-8")


def render_report_html(
    summary: Dict[str, Any],
    runs: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    seed_stats: Dict[int, Dict[str, Any]],
) -> str:
    config_json = json.dumps(summary["config"], ensure_ascii=False, indent=2)
    metrics = summary["summary"]
    total_steps = len(steps)
    k_positive = sum(entry["k_positive"] for entry in seed_stats.values())
    k_ratio = (k_positive / total_steps) if total_steps else 0.0
    best_hop_counter = Counter(int(rec.get("best_hop", 0)) for rec in steps)
    max_best_hop_ratio = max(best_hop_counter.values(), default=0)

    def metric_block(label: str, value: float) -> str:
        return f'<div class="metric"><strong>{label}</strong><br/><span>{value:.4f}</span></div>'

    metric_html = "\n".join(
        [
            metric_block("Success Rate", metrics.get("success_rate", 0.0)),
            metric_block("Average Steps", metrics.get("avg_steps", 0.0)),
            metric_block("Average Edges", metrics.get("avg_edges", 0.0)),
            metric_block("g₀ Mean", metrics.get("g0_mean", 0.0)),
            metric_block("g_min Mean", metrics.get("gmin_mean", 0.0)),
            metric_block("k★ Mean", metrics.get("k_star_mean", 0.0)),
            metric_block("Multihop Usage", metrics.get("multihop_usage", 0.0)),
            metric_block("Dead-end Steps (avg)", metrics.get("dead_end_steps_avg", 0.0)),
            metric_block("Dead-end Escape Rate", metrics.get("dead_end_escape_rate_avg", 0.0)),
            (
                f'<div class="metric"><strong>k★ ≥ 1 Ratio</strong><br/><span>{k_ratio:.4f}</span>'
                f'<div class="note">{k_positive}/{total_steps} steps</div></div>'
            ),
        ]
    )

    run_rows = []
    for run in sorted(runs, key=lambda item: int(item["seed"])):
        seed = int(run["seed"])
        seed_summary = seed_stats.get(seed, {})
        dead_steps = seed_summary.get("run_summary", {}).get("dead_end_steps", 0)
        dead_rate = seed_summary.get("run_summary", {}).get("dead_end_escape_rate", 0.0)
        run_rows.append(
            "<tr>"
            f"<td>{seed}</td>"
            f"<td>{'✓' if run.get('success') else '✗'}</td>"
            f"<td>{run.get('steps', 0)}</td>"
            f"<td>{run.get('edges', 0)}</td>"
            f"<td>{dead_steps}</td>"
            f"<td>{dead_rate:.3f}</td>"
            "</tr>"
        )
    run_table = "\n".join(run_rows)

    hop_rows = []
    if best_hop_counter:
        max_count = max(best_hop_counter.values())
        for hop, count in sorted(best_hop_counter.items()):
            width = 300 * (count / max_count) if max_count else 0
            hop_rows.append(
                f'<div class="bar-row"><div style="width:52px;">hop {hop}</div>'
                f'<div class="bar" style="width:{width:.0f}px"></div>'
                f'<div>{count / total_steps if total_steps else 0.0:.4f}</div></div>'
            )
    else:
        hop_rows.append('<div class="note">Step log is empty.</div>')
    hop_html = "\n".join(hop_rows)

    observations: List[str] = []
    if total_steps == 0:
        observations.append("ステップログが空です。実験の実行を確認してください。")
    else:
        if k_ratio < 0.05:
            observations.append(
                "候補集合が閾値をほぼ満たしておらず、k★ は停滞ぎみです。θ_cand/θ_link や類似度重みの再調整を検討してください。"
            )
        else:
            observations.append(
                "k★ は十分に立ち上がっており、候補スコアリングが機能しています。さらなる最適化は候補品質に集中できます。"
            )
        multihop_usage = metrics.get("multihop_usage", 0.0)
        if multihop_usage < 0.1:
            observations.append(
                "best_hop の大半が 0 で推移しています。decay_factor や max_hops の調整で multi-hop の感度を高められるか検証してください。"
            )
        else:
            observations.append(
                "multi-hop 評価が安定的に利用されています。hop 分布と g₀/g_min の差分を合わせて評価してください。"
            )

    obs_html = "\n".join(f"<li>{item}</li>" for item in observations)

    from string import Template

    report_template = Template("""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<title>Maze Fixed K★ Experiment Report</title>
<style>body{font-family:Arial,Helvetica,sans-serif;background:#fafafa;color:#222;padding:24px;}h1{font-size:24px;margin-top:0;}table{border-collapse:collapse;margin-top:16px;width:100%;}th,td{border:1px solid #ccc;padding:8px;text-align:left;}th{background:#f0f0f0;} .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:16px;} .metric{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,0.05);} .bar-row{display:flex;align-items:center;gap:8px;} .bar{height:14px;background:#4a90e2;border-radius:3px;} .tag{display:inline-block;padding:2px 6px;border-radius:4px;background:#e0f7fa;font-size:12px;margin-left:6px;color:#006064;} .note{font-size:12px;color:#666;margin-top:4px;} pre{background:#f7f7f7;padding:12px;border-radius:6px;overflow:auto;} </style>
</head>
<body>
<h1>Maze Fixed K★ Experiment Report</h1>
<section><h2>Config</h2><pre>${config}</pre></section>
<section><h2>Summary Metrics</h2><div class="metric-grid">
${metric_blocks}
</div></section>
<section><h2>Per-run Overview</h2><table><thead><tr><th>Seed</th><th>Success</th><th>Steps</th><th>Edges</th><th>Dead-end Steps</th><th>Dead-end Escape</th></tr></thead><tbody>
${run_rows}
</tbody></table></section>
<section><h2>best_hop 分布</h2><div class="note">Widthは相対頻度（最大300px）。</div>
${hop_rows}
</section>
<section><h2>Observations</h2>
<ul>${observations}</ul></section>
</body></html>
""")

    return report_template.safe_substitute(
        config=config_json,
        metric_blocks=metric_html,
        run_rows=run_table,
        hop_rows=hop_html,
        observations=obs_html,
    )


def generate_reports(
    summary_path: Path,
    steps_path: Path,
    interactive_path: Path,
    report_path: Path,
) -> None:
    summary = load_json(summary_path)
    steps = load_json(steps_path)

    seed_data, seed_stats = build_seed_data(summary["runs"], steps)
    experiment_data = OrderedDict(
        [
            ("config", summary["config"]),
            ("summary", summary["summary"]),
            ("runs", summary["runs"]),
            ("maze_data", summary["maze_data"]),
            ("seed_data", seed_data),
        ]
    )

    update_interactive_html(interactive_path, experiment_data)
    report_html = render_report_html(summary, summary["runs"], steps, seed_stats)
    report_path.write_text(report_html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh maze fixed-k★ experiment HTML reports.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary JSON output.")
    parser.add_argument("--steps", type=Path, required=True, help="Path to per-step JSON log.")
    parser.add_argument("--interactive", type=Path, required=True, help="Interactive HTML path to update.")
    parser.add_argument("--report", type=Path, required=True, help="Static report HTML output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_reports(args.summary, args.steps, args.interactive, args.report)


if __name__ == "__main__":
    main()
