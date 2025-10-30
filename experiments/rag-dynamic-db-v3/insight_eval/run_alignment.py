#!/usr/bin/env python3
"""Wrapper to run the paper's alignment script in this experiment directory."""
import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True, help="QA JSON (question/response pairs)")
    ap.add_argument("--outfig", type=str, default="results/figures/rag_insight_alignment.pdf")
    ap.add_argument("--kb", type=str, default="data/insight_store/knowledge_base/initial/insight_dataset.txt")
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--residual_ans", action="store_true")
    ap.add_argument("--neg", nargs="*", default=[], choices=[
        "shuffle_answers", "random_adj", "degree_preserve"
    ], help="Negative controls to compute additionally")
    ap.add_argument("--outjson", type=str, default=str(Path("results/outputs/rag_insight_alignment_stats.json")), help="JSON stats output path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]  # repo root
    script = root / "docs/paper/run_insight_vector_alignment.py"
    outfig = Path(args.outfig)
    outfig.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", str(script),
        "--inputs", args.inputs,
        "--outfig", str(outfig),
        "--kb", args.kb,
        "--H", str(args.H),
        "--gamma", str(args.gamma),
        "--tau", str(args.tau),
        "--outjson", args.outjson,
    ]
    for neg in (args.neg or []):
        cmd.extend(["--neg", neg])
    if args.residual_ans:
        cmd.append("--residual_ans")
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
