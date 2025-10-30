#!/usr/bin/env python3
"""Convenience wrapper: QA JSONL -> alignment inputs -> Δs evaluation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from .prepare_alignment_inputs import load_qa_pairs, save_alignment_json  # type: ignore
except ImportError:  # pragma: no cover
    from prepare_alignment_inputs import load_qa_pairs, save_alignment_json  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supplemental insight-alignment pipeline end-to-end.")
    parser.add_argument("--qa-jsonl", type=Path, required=True, help="QA JSONL (question/answer).")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Output directory.")
    parser.add_argument("--kb", type=str, default="data/insight_store/knowledge_base/initial/insight_dataset.txt")
    parser.add_argument("--H", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=0.35)
    parser.add_argument("--residual-ans", action="store_true", help="Use residual answer embedding.")
    parser.add_argument("--neg", nargs="*", default=[], choices=["shuffle_answers", "random_adj", "degree_preserve"], help="Negative controls.")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    alignment_json = outdir / "alignment_inputs.json"
    pairs = load_qa_pairs(args.qa_jsonl)
    if not pairs:
        raise SystemExit(f"No QA pairs found in {args.qa_jsonl}")
    save_alignment_json(pairs, alignment_json)

    root = Path(__file__).resolve().parents[3]
    script = root / "docs" / "paper" / "run_insight_vector_alignment.py"
    outfig = outdir / "rag_insight_alignment.pdf"
    outjson = outdir / "rag_insight_alignment_stats.json"

    cmd = [
        "python3",
        str(script),
        "--inputs",
        str(alignment_json),
        "--outfig",
        str(outfig),
        "--kb",
        args.kb,
        "--H",
        str(args.H),
        "--gamma",
        str(args.gamma),
        "--tau",
        str(args.tau),
        "--outjson",
        str(outjson),
    ]
    for neg in args.neg:
        cmd.extend(["--neg", neg])
    if args.residual_ans:
        cmd.append("--residual_ans")

    print("[run]", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise SystemExit(
            f"Alignment script failed (returncode={exc.returncode}). "
            "Ensure numpy/matplotlib/sentence-transformers など必要な依存がインストールされているか確認してください。"
        ) from exc
    print(f"Finished alignment. Outputs in {outdir}")


if __name__ == "__main__":
    main()
