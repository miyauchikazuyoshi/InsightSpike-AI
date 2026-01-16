#!/usr/bin/env python3
"""Run baseline experiments on HotpotQA."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from src.config import load_yaml, resolve_path
from src.data_loader import HotpotQALoader
from src.evaluator import EvaluationResult, HotpotQAEvaluator, exact_match, f1_score


METHOD_CONFIG_KEY = {
    "bm25": "bm25_gpt",
    "closed_book": "closed_book_gpt",
    "contriever": "contriever_gpt",
    "static_graphrag": "static_graphrag",
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "rate limit" in message
        or "rate_limit" in message
        or "rpd" in message
        or "429" in message
        or "too many requests" in message
    )


def load_resume_state(path: Path, evaluator: HotpotQAEvaluator) -> set[str]:
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            ex_id = record.get("id")
            if not ex_id or ex_id in seen_ids:
                continue
            seen_ids.add(ex_id)

            prediction = record.get("prediction") or ""
            ground_truth = record.get("ground_truth") or ""

            em = record.get("em")
            f1 = record.get("f1")
            precision = record.get("precision")
            recall = record.get("recall")

            if f1 is None or precision is None or recall is None:
                f1_calc, precision_calc, recall_calc = f1_score(
                    str(prediction), str(ground_truth)
                )
                f1 = f1_calc if f1 is None else f1
                precision = precision_calc if precision is None else precision
                recall = recall_calc if recall is None else recall

            if em is None:
                em = exact_match(str(prediction), str(ground_truth))

            sf_f1 = record.get("sf_f1", 0.0)
            sf_em = record.get("sf_em")
            sf_precision = record.get("sf_precision")
            sf_recall = record.get("sf_recall")

            if sf_em is None:
                sf_em = 1.0 if float(sf_f1) >= 0.999999 else 0.0
            if sf_precision is None:
                sf_precision = sf_f1
            if sf_recall is None:
                sf_recall = sf_f1

            latency_ms = record.get("latency_ms", 0.0) or 0.0

            evaluator.results.append(
                EvaluationResult(
                    example_id=str(ex_id),
                    em=float(em),
                    f1=float(f1),
                    precision=float(precision),
                    recall=float(recall),
                    sf_em=float(sf_em),
                    sf_f1=float(sf_f1),
                    sf_precision=float(sf_precision),
                    sf_recall=float(sf_recall),
                    latency_ms=float(latency_ms),
                )
            )

    return seen_ids


def get_baseline(method: str, method_config: dict):
    """Get baseline implementation by name and config."""
    retrieval_cfg = method_config.get("retrieval", {})
    llm_cfg = method_config.get("llm", {})

    if method == "bm25":
        from baselines.bm25_gpt import BM25GPTBaseline

        return BM25GPTBaseline(
            top_k=retrieval_cfg.get("top_k", 5),
            model=llm_cfg.get("model", "gpt-4o-mini"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 256),
        )
    if method == "closed_book":
        from baselines.closed_book_gpt import ClosedBookGPTBaseline

        return ClosedBookGPTBaseline(
            model=llm_cfg.get("model", "gpt-4o-mini"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 256),
        )
    if method == "contriever":
        from baselines.contriever_gpt import ContrieverGPTBaseline

        return ContrieverGPTBaseline(
            retriever_model=retrieval_cfg.get("model", "facebook/contriever"),
            top_k=retrieval_cfg.get("top_k", 5),
            llm_model=llm_cfg.get("model", "gpt-4o-mini"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 256),
            device=retrieval_cfg.get("device", "auto"),
            max_length=retrieval_cfg.get("max_length", 256),
            batch_size=retrieval_cfg.get("batch_size", 16),
        )
    if method == "static_graphrag":
        from baselines.static_graphrag import StaticGraphRAGBaseline

        return StaticGraphRAGBaseline(
            top_k=retrieval_cfg.get("top_k", 5),
            window=retrieval_cfg.get("window", 1),
            model=llm_cfg.get("model", "gpt-4o-mini"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 256),
        )
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline on HotpotQA")
    parser.add_argument("--method", type=str, required=True,
                        choices=["bm25", "closed_book", "contriever", "static_graphrag"],
                        help="Baseline method to run")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "baselines.yaml",
        help="Path to baseline config YAML",
    )
    parser.add_argument("--data", type=Path, default=None, help="Path to data file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: results/)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for testing)")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from an existing JSONL file (skips processed ids)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry count for rate limit errors (0 disables retries)",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=10.0,
        help="Seconds to wait before retrying after a rate limit error",
    )
    args = parser.parse_args()

    config = load_yaml(args.config) if args.config else {}
    baseline_config = config.get("baselines", config)
    method_key = METHOD_CONFIG_KEY.get(args.method, args.method)
    method_cfg = baseline_config.get(method_key, {}) or {}

    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    experiment_cfg = config.get("experiment", {})

    data_path = args.data or resolve_path(EXPERIMENT_DIR, data_cfg.get("path"))
    if data_path is None:
        data_path = EXPERIMENT_DIR / "data" / "hotpotqa_sample_100.jsonl"
    output_dir = args.output or resolve_path(EXPERIMENT_DIR, output_cfg.get("results_dir"))
    if output_dir is None:
        output_dir = EXPERIMENT_DIR / "results"
    seed = args.seed if args.seed is not None else experiment_cfg.get("seed", 42)

    set_seed(int(seed))

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_mode = "w"
    resume_path = args.resume_from
    if resume_path:
        resume_path = resume_path.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
        output_file = resume_path
        output_mode = "a"
    else:
        output_file = output_dir / f"{args.method}_{timestamp}.jsonl"
    summary_file = output_dir / f"{args.method}_{timestamp}_summary.json"

    print(f"[run] Method: {args.method}")
    print(f"[run] Data: {data_path}")
    print(f"[run] Output: {output_file}")
    if args.config:
        print(f"[run] Config: {args.config}")

    # Load data
    loader = HotpotQALoader(data_path)
    examples = loader.load()

    if args.limit:
        examples = examples[:args.limit]

    print(f"[run] Loaded {len(examples)} examples")

    # Setup baseline
    baseline = get_baseline(args.method, method_cfg)
    baseline.setup(examples)

    # Run evaluation
    evaluator = HotpotQAEvaluator()
    seen_ids: set[str] = set()
    if resume_path:
        seen_ids = load_resume_state(resume_path, evaluator)
        if seen_ids:
            print(f"[run] Resume loaded {len(seen_ids)} records from {resume_path}")

    remaining = [ex for ex in examples if ex.id not in seen_ids]
    if seen_ids:
        print(f"[run] Remaining {len(remaining)} examples (skipped {len(seen_ids)})")

    with open(output_file, output_mode, encoding="utf-8") as f:
        for i, example in enumerate(remaining):
            print(f"[run] Processing {i+1}/{len(remaining)}: {example.id[:16]}...", end="\r")
            attempts = 0
            while True:
                try:
                    result = baseline.process(example)

                    eval_result = evaluator.evaluate_single(
                        example_id=example.id,
                        prediction=result.answer,
                        ground_truth=example.answer,
                        predicted_facts=result.retrieved_facts,
                        gold_facts=example.supporting_facts,
                        latency_ms=result.latency_ms,
                    )

                    record = {
                        "id": example.id,
                        "question": example.question,
                        "prediction": result.answer,
                        "ground_truth": example.answer,
                        "em": eval_result.em,
                        "f1": eval_result.f1,
                        "precision": eval_result.precision,
                        "recall": eval_result.recall,
                        "sf_em": eval_result.sf_em,
                        "sf_f1": eval_result.sf_f1,
                        "sf_precision": eval_result.sf_precision,
                        "sf_recall": eval_result.sf_recall,
                        "latency_ms": result.latency_ms,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    break

                except Exception as e:
                    if is_rate_limit_error(e) and attempts < args.max_retries:
                        attempts += 1
                        wait_s = max(0.0, float(args.retry_wait))
                        print(
                            f"\n[warn] Rate limit on {example.id}, retry "
                            f"{attempts}/{args.max_retries} in {wait_s:.1f}s"
                        )
                        if wait_s:
                            time.sleep(wait_s)
                        continue

                    print(f"\n[error] Failed on {example.id}: {e}")
                    break

    print()

    # Aggregate and save summary
    aggregated = evaluator.aggregate()
    expected_count = len(examples)
    summary = {
        "method": args.method,
        "data": str(data_path),
        "count": aggregated.count,
        "timestamp": timestamp,
        "config": str(args.config) if args.config else None,
        "parameters": method_cfg,
        **aggregated.to_dict(),
    }
    if aggregated.count < expected_count:
        summary["incomplete"] = True
        summary["expected_count"] = expected_count

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] Results saved to {output_file}")
    print(f"[done] Summary saved to {summary_file}")
    print(f"[results] EM={aggregated.em:.4f}, F1={aggregated.f1:.4f}, SF-F1={aggregated.sf_f1:.4f}")


if __name__ == "__main__":
    main()
