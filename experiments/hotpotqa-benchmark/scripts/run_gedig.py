#!/usr/bin/env python3
"""Run geDIG experiments on HotpotQA."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.config import load_yaml, resolve_path
from src.data_loader import HotpotQALoader
from src.evaluator import EvaluationResult, HotpotQAEvaluator, exact_match, f1_score
from src.hotpotqa_adapter import GeDIGHotpotQAAdapter


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_resume_state(
    path: Path, evaluator: HotpotQAEvaluator
) -> tuple[set[str], int, int, int, int, float, int]:
    seen_ids: set[str] = set()
    initial_ag_fires = 0
    initial_dg_fires = 0
    final_ag_fires = 0
    final_dg_fires = 0
    total_gedig = 0.0
    total_edges = 0

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

            initial_ag_fires += int(bool(record.get("initial_ag_fired", False)))
            initial_dg_fires += int(bool(record.get("initial_dg_fired", False)))
            final_ag_fires += int(bool(record.get("ag_fired", False)))
            final_dg_fires += int(bool(record.get("dg_fired", False)))
            total_gedig += float(record.get("gedig_score", 0.0) or 0.0)
            total_edges += int(record.get("graph_edges", 0) or 0)

    return (
        seen_ids,
        initial_ag_fires,
        initial_dg_fires,
        final_ag_fires,
        final_dg_fires,
        total_gedig,
        total_edges,
    )


def main():
    parser = argparse.ArgumentParser(description="Run geDIG on HotpotQA")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "gedig_hotpotqa.yaml",
        help="Path to geDIG config YAML",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to data file",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output directory (default: results/)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples (for testing)"
    )
    # geDIG parameters
    parser.add_argument("--lambda-weight", type=float, default=None, help="Lambda weight")
    parser.add_argument("--theta-ag", type=float, default=None, help="AG threshold")
    parser.add_argument("--theta-dg", type=float, default=None, help="DG threshold")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k retrieval")
    parser.add_argument("--max-expansions", type=int, default=None, help="Max AG expansions")
    parser.add_argument("--expansion-seeds", type=int, default=None, help="Expansion seed count")
    parser.add_argument("--tfidf-dim", type=int, default=None, help="TF-IDF hash dimension")
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune AG/DG thresholds")
    parser.add_argument("--tune-size", type=int, default=None, help="Tuning sample size")
    parser.add_argument("--tune-ag-percentile", type=float, default=None, help="AG percentile")
    parser.add_argument("--tune-dg-percentile", type=float, default=None, help="DG percentile")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from an existing JSONL file (skips processed ids)",
    )
    args = parser.parse_args()

    config = load_yaml(args.config) if args.config else {}
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    experiment_cfg = config.get("experiment", {})
    gedig_cfg = config.get("gedig", {})
    retrieval_cfg = config.get("retrieval", {})
    llm_cfg = config.get("llm", {})
    tuning_cfg = config.get("tuning", {})

    data_path = args.data or resolve_path(EXPERIMENT_DIR, data_cfg.get("path"))
    if data_path is None:
        data_path = EXPERIMENT_DIR / "data" / "hotpotqa_sample_100.jsonl"
    output_dir = args.output or resolve_path(EXPERIMENT_DIR, output_cfg.get("results_dir"))
    if output_dir is None:
        output_dir = EXPERIMENT_DIR / "results"
    seed = args.seed if args.seed is not None else experiment_cfg.get("seed", 42)

    lambda_weight = args.lambda_weight if args.lambda_weight is not None else gedig_cfg.get("lambda", 1.0)
    theta_ag = args.theta_ag if args.theta_ag is not None else gedig_cfg.get("theta_ag", 0.4)
    theta_dg = args.theta_dg if args.theta_dg is not None else gedig_cfg.get("theta_dg", 0.0)
    top_k = args.top_k if args.top_k is not None else retrieval_cfg.get("top_k", 5)
    max_hops = gedig_cfg.get("max_hops", 2)
    max_expansions = args.max_expansions if args.max_expansions is not None else gedig_cfg.get("max_expansions", 1)
    expansion_seeds = args.expansion_seeds if args.expansion_seeds is not None else gedig_cfg.get("expansion_seeds", 2)
    tfidf_dim = args.tfidf_dim if args.tfidf_dim is not None else gedig_cfg.get("tfidf_dim", 64)
    llm_model = llm_cfg.get("model", "gpt-4o-mini")
    llm_temperature = llm_cfg.get("temperature", 0.0)
    llm_max_tokens = llm_cfg.get("max_tokens", 256)
    gamma = gedig_cfg.get("gamma", 1.0)

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
        output_file = output_dir / f"gedig_{timestamp}.jsonl"
    summary_file = output_dir / f"gedig_{timestamp}_summary.json"

    print(f"[run] Method: geDIG")
    print(f"[run] Data: {data_path}")
    print(f"[run] Output: {output_file}")
    if args.config:
        print(f"[run] Config: {args.config}")
    print(
        " ".join(
            [
                f"[run] Parameters: lambda={lambda_weight}",
                f"theta_ag={theta_ag}",
                f"theta_dg={theta_dg}",
                f"top_k={top_k}",
                f"max_expansions={max_expansions}",
                f"expansion_seeds={expansion_seeds}",
                f"tfidf_dim={tfidf_dim}",
            ]
        )
    )

    # Load data
    loader = HotpotQALoader(data_path)
    examples = loader.load()

    if args.limit:
        examples = examples[: args.limit]

    print(f"[run] Loaded {len(examples)} examples")

    # Setup geDIG adapter
    adapter = GeDIGHotpotQAAdapter(
        lambda_weight=lambda_weight,
        gamma=gamma,
        theta_ag=theta_ag,
        theta_dg=theta_dg,
        max_hops=max_hops,
        top_k=top_k,
        llm_model=llm_model,
        max_expansions=max_expansions,
        expansion_seeds=expansion_seeds,
        tfidf_dim=tfidf_dim,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
    )
    adapter.setup(examples)

    def _percentile_value(values: list[float], pct: float) -> float:
        p = pct * 100 if pct <= 1 else pct
        return float(np.percentile(values, p))

    tune_enabled = bool(args.tune_thresholds or tuning_cfg.get("enable", False))
    tuning_info = {"enabled": False}
    if tune_enabled:
        sample_size = args.tune_size or tuning_cfg.get("sample_size", 100)
        ag_pct = args.tune_ag_percentile or tuning_cfg.get("ag_percentile", 50)
        dg_pct = args.tune_dg_percentile or tuning_cfg.get("dg_percentile", 30)

        sample_size = min(int(sample_size), len(examples))
        sampled = random.sample(examples, sample_size) if sample_size < len(examples) else examples

        g0_values = []
        b_values = []
        for ex in sampled:
            g0, gmin = adapter.score_example(ex)
            g0_values.append(g0)
            b_values.append(min(g0, gmin))

        if g0_values:
            theta_ag = _percentile_value(g0_values, float(ag_pct)) - 1e-6
            theta_dg = _percentile_value(b_values, float(dg_pct))
            adapter.theta_ag = theta_ag
            adapter.theta_dg = theta_dg

        tuning_info = {
            "enabled": True,
            "sample_size": sample_size,
            "ag_percentile": ag_pct,
            "dg_percentile": dg_pct,
            "theta_ag": theta_ag,
            "theta_dg": theta_dg,
        }
        print(
            " ".join(
                [
                    "[run] Tuned thresholds:",
                    f"theta_ag={theta_ag:.4f}",
                    f"theta_dg={theta_dg:.4f}",
                    f"sample_size={sample_size}",
                ]
            )
        )

    # Run evaluation
    evaluator = HotpotQAEvaluator()

    # Track geDIG-specific metrics
    initial_ag_fires = 0
    initial_dg_fires = 0
    final_ag_fires = 0
    final_dg_fires = 0
    total_gedig = 0.0
    total_edges = 0
    seen_ids: set[str] = set()

    if resume_path:
        (
            seen_ids,
            initial_ag_fires,
            initial_dg_fires,
            final_ag_fires,
            final_dg_fires,
            total_gedig,
            total_edges,
        ) = load_resume_state(resume_path, evaluator)
        if seen_ids:
            print(f"[run] Resume loaded {len(seen_ids)} records from {resume_path}")

    remaining = [ex for ex in examples if ex.id not in seen_ids]
    if seen_ids:
        print(f"[run] Remaining {len(remaining)} examples (skipped {len(seen_ids)})")

    with open(output_file, output_mode, encoding="utf-8") as f:
        for i, example in enumerate(remaining):
            print(
                f"[run] Processing {i+1}/{len(remaining)}: {example.id[:16]}...", end="\r"
            )

            try:
                result = adapter.process(example)

                eval_result = evaluator.evaluate_single(
                    example_id=example.id,
                    prediction=result.answer,
                    ground_truth=example.answer,
                    predicted_facts=result.retrieved_facts,
                    gold_facts=example.supporting_facts,
                    latency_ms=result.latency_ms,
                )

                # Track geDIG metrics
                initial_ag = bool(getattr(result, "initial_ag_fired", False))
                initial_dg = bool(getattr(result, "initial_dg_fired", False))
                if not initial_ag and not initial_dg:
                    initial_ag = bool(result.metadata.get("initial_ag", False))
                    initial_dg = bool(result.metadata.get("initial_dg", False))

                initial_ag_fires += int(initial_ag)
                initial_dg_fires += int(initial_dg)
                final_ag_fires += int(result.ag_fired)
                final_dg_fires += int(result.dg_fired)
                total_gedig += result.gedig_score
                total_edges += result.graph_edges

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
                    "gedig_score": result.gedig_score,
                    "initial_ag_fired": initial_ag,
                    "initial_dg_fired": initial_dg,
                    "ag_fired": result.ag_fired,
                    "dg_fired": result.dg_fired,
                    "graph_edges": result.graph_edges,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"\n[error] Failed on {example.id}: {e}")
                continue

    print()

    # Aggregate and save summary
    aggregated = evaluator.aggregate()
    n = aggregated.count if aggregated.count > 0 else 1

    summary = {
        "method": "gedig",
        "data": str(data_path),
        "count": aggregated.count,
        "timestamp": timestamp,
        "config": str(args.config) if args.config else None,
        "parameters": {
            "lambda_weight": lambda_weight,
            "theta_ag": theta_ag,
            "theta_dg": theta_dg,
            "top_k": top_k,
            "max_expansions": max_expansions,
            "max_hops": max_hops,
            "gamma": gamma,
            "llm_model": llm_model,
            "llm_temperature": llm_temperature,
            "llm_max_tokens": llm_max_tokens,
            "expansion_seeds": expansion_seeds,
            "tfidf_dim": tfidf_dim,
        },
        "tuning": tuning_info,
        **aggregated.to_dict(),
        # geDIG-specific metrics
        "ag_fire_rate": initial_ag_fires / n,
        "dg_fire_rate": initial_dg_fires / n,
        "final_ag_fire_rate": final_ag_fires / n,
        "final_dg_fire_rate": final_dg_fires / n,
        "avg_gedig_score": total_gedig / n,
        "avg_graph_edges": total_edges / n,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] Results saved to {output_file}")
    print(f"[done] Summary saved to {summary_file}")
    print(f"[results] EM={aggregated.em:.4f}, F1={aggregated.f1:.4f}, SF-F1={aggregated.sf_f1:.4f}")
    print(
        f"[gedig] AG fire rate={initial_ag_fires/n:.2%} (initial), DG fire rate={initial_dg_fires/n:.2%} (initial)"
    )
    print(
        f"[gedig] Final AG fire rate={final_ag_fires/n:.2%}, Final DG fire rate={final_dg_fires/n:.2%}"
    )
    print(f"[gedig] Avg geDIG score={total_gedig/n:.4f}, Avg edges={total_edges/n:.1f}")


if __name__ == "__main__":
    main()
