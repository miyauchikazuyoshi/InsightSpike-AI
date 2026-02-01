#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess as sp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def run(cmd: list[str], env: dict | None = None):
    print("+", " ".join(cmd))
    sp.run(cmd, check=True, env={**os.environ, **(env or {})})


def build(summary: Path, steps: Path, out: Path):
    run([
        "python", str(ROOT / "build_reports.py"),
        "--summary", str(summary),
        "--steps", str(steps),
        "--out", str(out),
    ])


def main():
    RES.mkdir(parents=True, exist_ok=True)
    # A/B: after_before (baseline)
    s1 = RES / "cmp_ab_after_summary.json"; t1 = RES / "cmp_ab_after_steps.json"; h1 = RES / "cmp_ab_after_interactive.html"
    run([
        "python", str(ROOT / "run_experiment_query.py"),
        "--maze-size", "25", "--max-steps", "60",
        "--linkset-mode", "--norm-base", "link",
        "--output", str(s1), "--step-log", str(t1),
    ])
    build(s1, t1, h1)

    # L3 lite (cached_incr, global)
    s2 = RES / "cmp_l3lite_summary.json"; t2 = RES / "cmp_l3lite_steps.json"; h2 = RES / "cmp_l3lite_interactive.html"
    run([
        "python", str(ROOT / "run_experiment_query.py"),
        "--maze-size", "25", "--max-steps", "60",
        "--sp-cache", "--sp-cache-mode", "cached_incr", "--sp-pair-samples", "80", "--commit-budget", "2",
        "--use-main-l3",
        "--output", str(s2), "--step-log", str(t2),
    ], env={"INSIGHTSPIKE_SP_ENGINE": "cached_incr", "INSIGHTSPIKE_SP_REGISTRY": str(RES / "pairsets.json")})
    build(s2, t2, h2)

    # L3 lite (cached_incr + union-of-k-hop)
    s3 = RES / "cmp_l3lite_union_summary.json"; t3 = RES / "cmp_l3lite_union_steps.json"; h3 = RES / "cmp_l3lite_union_interactive.html"
    run([
        "python", str(ROOT / "run_experiment_query.py"),
        "--maze-size", "25", "--max-steps", "60",
        "--sp-cache", "--sp-cache-mode", "cached_incr", "--sp-pair-samples", "80", "--commit-budget", "2",
        "--use-main-l3",
        "--output", str(s3), "--step-log", str(t3),
    ], env={"INSIGHTSPIKE_SP_ENGINE": "cached_incr", "INSIGHTSPIKE_SP_REGISTRY": str(RES / "pairsets.json"), "INSIGHTSPIKE_SP_ADAPTIVE": "1"})
    build(s3, t3, h3)

    print("Done. Reports:")
    print(" -", h1)
    print(" -", h2)
    print(" -", h3)


if __name__ == "__main__":
    main()
