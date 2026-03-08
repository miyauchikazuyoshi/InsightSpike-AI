#!/usr/bin/env python3
"""Run BM25 or Static GraphRAG baseline on HotpotQA.

Usage::

    PYTHONPATH=src python experiments/hotpotqa_v2/scripts/run_baseline.py \
        --baseline bm25 --data data/hotpotqa_sample_100.jsonl --limit 10

    LLM_PROVIDER=mock PYTHONPATH=src python experiments/hotpotqa_v2/scripts/run_baseline.py \
        --baseline graphrag --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_EXP_ROOT = Path(__file__).parent.parent
_REPO_ROOT = _EXP_ROOT.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.hotpotqa_v2.src.data_loader import HotpotQALoader
from experiments.hotpotqa_v2.src.evaluator import HotpotQAEvaluator
from experiments.hotpotqa_v2.baselines.bm25_gpt import BM25GPTBaseline
from experiments.hotpotqa_v2.baselines.static_graphrag import StaticGraphRAGBaseline
from experiments.hotpotqa_v2.baselines.ircot import IRCoTBaseline
from experiments.hotpotqa_v2.baselines.react_baseline import ReActBaseline


_BASELINES = {
    "bm25": BM25GPTBaseline,
    "graphrag": StaticGraphRAGBaseline,
    "ircot": IRCoTBaseline,
    "react": ReActBaseline,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline on HotpotQA")
    parser.add_argument("--baseline", type=str, required=True, choices=list(_BASELINES.keys()))
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--window", type=int, default=1, help="GraphRAG window size")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps for IRCoT/ReAct")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model name (default: gpt-4o-mini)")
    args = parser.parse_args()

    # Data — CLI paths resolve from repo root
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = _REPO_ROOT / data_path
    else:
        data_path = _EXP_ROOT / "data" / "hotpotqa_distractor_dev.jsonl"

    # Output
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = _REPO_ROOT / output_dir
    else:
        output_dir = _EXP_ROOT / "results" / f"baseline_{args.baseline}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"[baseline] Loading data from {data_path}")
    loader = HotpotQALoader(data_path)
    examples = loader.load()
    if args.limit:
        examples = examples[: args.limit]
    print(f"[baseline] Loaded {len(examples)} examples")

    # Setup baseline
    cls = _BASELINES[args.baseline]
    if args.baseline == "graphrag":
        baseline = cls(top_k=args.top_k, window=args.window, model=args.model)
    elif args.baseline in ("ircot", "react"):
        kwargs = {"top_k": args.top_k, "model": args.model}
        if args.max_steps is not None:
            kwargs["max_steps"] = args.max_steps
        baseline = cls(**kwargs)
    else:
        baseline = cls(top_k=args.top_k, model=args.model)
    baseline.setup(examples)

    evaluator = HotpotQAEvaluator()
    total = len(examples)
    t0 = time.time()
    results_file = output_dir / "results.jsonl"

    with open(results_file, "w") as fout:
        for i, example in enumerate(examples):
            try:
                result = baseline.process(example)
                eval_result = evaluator.evaluate_single(
                    example_id=example.id,
                    prediction=result.answer,
                    ground_truth=example.answer,
                    predicted_facts=result.retrieved_facts,
                    gold_facts=example.supporting_facts,
                    latency_ms=result.latency_ms,
                    question_type=example.question_type,
                )
                record = {
                    "example_id": example.id,
                    "question_type": example.question_type,
                    "ground_truth": example.answer,
                    "prediction": result.answer,
                    "em": eval_result.em,
                    "f1": eval_result.f1,
                    "sf_f1": eval_result.sf_f1,
                    "latency_ms": result.latency_ms,
                    "baseline": args.baseline,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                elapsed = time.time() - t0
                avg_ms = (elapsed / (i + 1)) * 1000
                print(f"  [{i+1}/{total}] EM={eval_result.em:.0f} F1={eval_result.f1:.3f} ({avg_ms:.0f}ms avg)")

            except Exception as exc:
                print(f"  [{i+1}/{total}] ERROR: {exc}")

    # Summary
    agg = evaluator.aggregate_by_type()
    summary = {
        "baseline": args.baseline,
        "total_examples": total,
        "results": {k: v.to_dict() for k, v in agg.items()},
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Baseline: {args.baseline}")
    for label, res in agg.items():
        d = res.to_dict()
        print(f"  {label:>12s}: n={d['count']:>5d}  EM={d['em']:.4f}  F1={d['f1']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
