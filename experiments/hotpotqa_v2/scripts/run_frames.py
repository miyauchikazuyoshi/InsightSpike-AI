#!/usr/bin/env python3
"""v12: Open-World Topology-Guided Retrieval experiments on FRAMES.

Modes:
  gold_only   : Answer using gold articles directly (upper bound baseline)
  wiki_bm25   : Wikipedia search → direct LLM answer (no graph, lower bound)
  iterative   : Wikipedia search → graph → β₀-driven bridge retrieval → answer

Usage:
  PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \\
      experiments/hotpotqa_v2/scripts/run_frames.py \\
      --mode iterative \\
      --data experiments/hotpotqa_v2/data/frames_benchmark.jsonl \\
      --output experiments/hotpotqa_v2/results/v12_frames_iter \\
      --limit 50 --initial-top-k 5 --max-iterations 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wiki_retriever import WikipediaRetriever
from open_world_pipeline import OpenWorldPipeline, GoldOnlyPipeline


# ---------------------------------------------------------------------------
# Evaluation helpers (from run_allcontext.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    import string
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt)
    r = n / len(gt)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v12: FRAMES Open-World Retrieval Experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["gold_only", "wiki_bm25", "iterative"],
        required=True,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="experiments/hotpotqa_v2/data/frames_benchmark.jsonl",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o")

    # Retrieval parameters
    parser.add_argument("--initial-top-k", type=int, default=5)
    parser.add_argument("--bridge-top-k", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=3)

    # Graph parameters (from v11 P2d best)
    parser.add_argument("--k-hop", type=int, default=3)
    parser.add_argument("--max-subgraph-paras", type=int, default=15)
    parser.add_argument("--max-para-freq", type=int, default=3)
    parser.add_argument("--theta-f", type=float, default=999.0)

    # Convergence
    parser.add_argument("--delta-f-epsilon", type=float, default=0.05)

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    total = len(examples) if args.limit is None else min(args.limit, len(examples))
    examples = examples[:total]
    print(f"  Loaded {total} examples")

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    if args.mode == "gold_only":
        pipeline = GoldOnlyPipeline(
            k_hop=args.k_hop,
            max_subgraph_paras=args.max_subgraph_paras,
            max_para_freq=args.max_para_freq,
            theta_f=args.theta_f,
            model=args.model,
        )
    else:
        wiki_retriever = WikipediaRetriever(
            max_results=args.initial_top_k + 5,
            request_delay=0.15,
        )
        pipeline = OpenWorldPipeline(
            wiki_retriever=wiki_retriever,
            initial_top_k=args.initial_top_k,
            bridge_top_k=args.bridge_top_k,
            max_iterations=args.max_iterations if args.mode == "iterative" else 0,
            k_hop=args.k_hop,
            max_subgraph_paras=args.max_subgraph_paras,
            max_para_freq=args.max_para_freq,
            theta_f=args.theta_f,
            delta_f_epsilon=args.delta_f_epsilon,
            model=args.model,
        )

    # Save config
    config = {
        "mode": args.mode,
        "model": args.model,
        "initial_top_k": args.initial_top_k,
        "bridge_top_k": args.bridge_top_k,
        "max_iterations": args.max_iterations,
        "k_hop": args.k_hop,
        "max_subgraph_paras": args.max_subgraph_paras,
        "max_para_freq": args.max_para_freq,
        "theta_f": args.theta_f,
        "delta_f_epsilon": args.delta_f_epsilon,
        "limit": total,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run experiments
    results_file = out_dir / "results.jsonl"
    records: list[dict] = []

    # Resume support
    done_ids: set[str] = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["example_id"])
                records.append(r)
        print(f"  Resuming: {len(done_ids)} already done")

    print(f"\nRunning {args.mode} on {total} examples...\n")

    for idx, ex in enumerate(examples):
        example_id = ex["id"]
        if example_id in done_ids:
            continue

        question = ex["question"]
        answer = ex["answer"]
        reasoning_type = ex.get("type", "unknown")
        gold_titles = list(set(ex.get("supporting_facts", {}).get("title", [])))

        print(f"  [{idx+1}/{total}] {example_id}: {question[:80]}...")

        t0 = time.time()

        try:
            if args.mode == "gold_only":
                result = pipeline.run(
                    question,
                    ex["context"]["title"],
                    ex["context"]["sentences"],
                    gold_titles=gold_titles,
                )
            else:
                result = pipeline.run(question, gold_titles=gold_titles)

            pred = result.answer
            em = compute_em(pred, answer)
            f1 = compute_f1(pred, answer)

            record = {
                "example_id": example_id,
                "question": question,
                "ground_truth": answer,
                "prediction": pred,
                "em": em,
                "f1": f1,
                "mode": args.mode,
                "reasoning_type": reasoning_type,
                "latency_ms": round(result.latency_ms, 1),
                "n_llm_calls": result.n_llm_calls,
                "system_used": result.system_used,
                "context_tokens_est": result.context_tokens_est,
                "n_articles_retrieved": result.retrieval_state.n_articles,
                "n_iterations": result.retrieval_state.iteration,
                "convergence_reason": result.retrieval_state.convergence_reason,
                "beta_0": result.retrieval_state.beta_0,
                "beta_1": result.retrieval_state.beta_1,
                "f_value": result.retrieval_state.f_value,
                "search_queries": result.retrieval_state.search_queries,
                "bridge_queries": result.retrieval_state.bridge_queries,
                "retrieved_titles": result.retrieval_state.titles,
                "gold_titles": gold_titles,
                "gold_recall": result.retrieval_state.gold_recall,
                "gold_precision": result.retrieval_state.gold_precision,
                "subgraph_n_paras": result.subgraph_n_paras,
                "subgraph_gold_precision": result.subgraph_gold_precision,
                "subgraph_gold_recall": result.subgraph_gold_recall,
                "error": result.error,
            }

            status = "✓" if em > 0 else "✗"
            print(
                f"    {status} EM={em:.0f} F1={f1:.3f} "
                f"articles={result.retrieval_state.n_articles} "
                f"iter={result.retrieval_state.iteration} "
                f"β₀={result.retrieval_state.beta_0} "
                f"gold_recall={result.retrieval_state.gold_recall or 0:.2f} "
                f"({result.latency_ms:.0f}ms)"
            )

        except Exception as e:
            record = {
                "example_id": example_id,
                "question": question,
                "ground_truth": answer,
                "prediction": "ERROR",
                "em": 0.0,
                "f1": 0.0,
                "mode": args.mode,
                "reasoning_type": reasoning_type,
                "error": str(e),
            }
            print(f"    ERROR: {e}")

        records.append(record)

        # Write incrementally
        with open(results_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid = [r for r in records if r.get("error") is None]
    errors = [r for r in records if r.get("error") is not None]

    if valid:
        avg_em = sum(r["em"] for r in valid) / len(valid)
        avg_f1 = sum(r["f1"] for r in valid) / len(valid)
        print(f"  N={len(valid)} (valid), {len(errors)} errors")
        print(f"  EM  = {avg_em:.1%}")
        print(f"  F1  = {avg_f1:.3f}")

        # Retrieval metrics
        if any("gold_recall" in r and r["gold_recall"] is not None for r in valid):
            recalls = [r["gold_recall"] for r in valid if r.get("gold_recall") is not None]
            precisions = [r["gold_precision"] for r in valid if r.get("gold_precision") is not None]
            print(f"\n  Retrieval:")
            print(f"    avg gold_recall    = {sum(recalls)/len(recalls):.3f}")
            print(f"    avg gold_precision = {sum(precisions)/len(precisions):.3f}")

        if any("n_iterations" in r for r in valid):
            iters = [r.get("n_iterations", 0) for r in valid]
            articles = [r.get("n_articles_retrieved", 0) for r in valid]
            llm_calls = [r.get("n_llm_calls", 0) for r in valid]
            print(f"\n  Iterations:")
            print(f"    avg iterations     = {sum(iters)/len(iters):.2f}")
            print(f"    avg articles       = {sum(articles)/len(articles):.1f}")
            print(f"    avg LLM calls      = {sum(llm_calls)/len(llm_calls):.2f}")

        # Convergence reasons
        if any("convergence_reason" in r for r in valid):
            reasons = Counter(r.get("convergence_reason", "") for r in valid)
            print(f"\n  Convergence reasons:")
            for reason, count in reasons.most_common():
                print(f"    {reason}: {count} ({count/len(valid):.1%})")

        # By reasoning type
        type_groups: dict[str, list] = {}
        for r in valid:
            t = r.get("reasoning_type", "unknown")
            # Simplify multi-type labels
            primary = t.split("|")[0].strip() if "|" in t else t
            type_groups.setdefault(primary, []).append(r)

        if len(type_groups) > 1:
            print(f"\n  By primary reasoning type:")
            for t, group in sorted(type_groups.items(), key=lambda x: -len(x[1])):
                em = sum(r["em"] for r in group) / len(group)
                f1 = sum(r["f1"] for r in group) / len(group)
                print(f"    {t:30s}  N={len(group):3d}  EM={em:.1%}  F1={f1:.3f}")

        # Save summary
        summary = {
            "mode": args.mode,
            "model": args.model,
            "n": len(valid),
            "n_errors": len(errors),
            "em": round(avg_em, 4),
            "f1": round(avg_f1, 4),
        }

        if recalls:
            summary["avg_gold_recall"] = round(sum(recalls) / len(recalls), 4)
            summary["avg_gold_precision"] = round(sum(precisions) / len(precisions), 4)

        if iters:
            summary["avg_iterations"] = round(sum(iters) / len(iters), 2)
            summary["avg_articles"] = round(sum(articles) / len(articles), 1)
            summary["avg_llm_calls"] = round(sum(llm_calls) / len(llm_calls), 2)

        summary["convergence_reasons"] = dict(Counter(
            r.get("convergence_reason", "") for r in valid
        ))

        by_type = {}
        for t, group in type_groups.items():
            by_type[t] = {
                "n": len(group),
                "em": round(sum(r["em"] for r in group) / len(group), 4),
                "f1": round(sum(r["f1"] for r in group) / len(group), 4),
            }
        summary["by_reasoning_type"] = by_type

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n  Results: {results_file}")
        print(f"  Summary: {out_dir / 'summary.json'}")
    else:
        print("  No valid results!")


if __name__ == "__main__":
    main()
