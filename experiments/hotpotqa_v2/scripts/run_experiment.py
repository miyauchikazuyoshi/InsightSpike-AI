#!/usr/bin/env python3
"""Unified geDIG v2 experiment runner for HotpotQA.

Usage::

    # From repo root:
    PYTHONPATH=src python experiments/hotpotqa_v2/scripts/run_experiment.py \
        --config experiments/hotpotqa_v2/configs/condition_d_betti_full.yaml \
        --output experiments/hotpotqa_v2/results/condition_d/

    # Mock mode (no LLM calls):
    LLM_PROVIDER=mock PYTHONPATH=src python experiments/hotpotqa_v2/scripts/run_experiment.py \
        --config experiments/hotpotqa_v2/configs/condition_a_sp.yaml \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure experiment src is importable
_EXP_ROOT = Path(__file__).parent.parent
_REPO_ROOT = _EXP_ROOT.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.hotpotqa_v2.src.config import load_yaml, resolve_path
from experiments.hotpotqa_v2.src.data_loader import HotpotQALoader
from experiments.hotpotqa_v2.src.evaluator import HotpotQAEvaluator
from experiments.hotpotqa_v2.src.adapter import GeDIGv2Adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run geDIG v2 experiment on HotpotQA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    condition_name = cfg.get("condition_name", config_path.stem)

    # Data path — CLI args resolve from repo root; config paths resolve from experiment root
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = _REPO_ROOT / data_path
    else:
        raw = cfg.get("data_path", "data/hotpotqa_distractor_dev.jsonl")
        data_path = Path(raw)
        if not data_path.is_absolute():
            data_path = _EXP_ROOT / data_path

    # Output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = _REPO_ROOT / output_dir
    else:
        output_dir = _EXP_ROOT / "results" / condition_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"[run] Loading data from {data_path}")
    loader = HotpotQALoader(data_path)
    examples = loader.load()
    if args.limit:
        examples = examples[: args.limit]
    print(f"[run] Loaded {len(examples)} examples")

    # Resume support
    completed_ids: set[str] = set()
    results_file = output_dir / "results.jsonl"
    if args.resume and results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                record = json.loads(line)
                completed_ids.add(record["example_id"])
        print(f"[run] Resuming: {len(completed_ids)} already completed")

    # Build adapter
    adapter_cfg = cfg.get("adapter", {})
    adapter = GeDIGv2Adapter(
        structural_mode=adapter_cfg.get("structural_mode", "betti_full"),
        gamma_0=adapter_cfg.get("gamma_0", 1.0),
        gamma_1=adapter_cfg.get("gamma_1", 1.0),
        lambda_weight=adapter_cfg.get("lambda_weight", 1.0),
        theta_ag=adapter_cfg.get("theta_ag", 0.4),
        theta_dg=adapter_cfg.get("theta_dg", 0.0),
        max_hops=adapter_cfg.get("max_hops", 2),
        top_k=adapter_cfg.get("top_k", 5),
        max_expansions=adapter_cfg.get("max_expansions", 1),
        expansion_seeds=adapter_cfg.get("expansion_seeds", 2),
        tfidf_dim=adapter_cfg.get("tfidf_dim", 64),
        q_link_top_k=adapter_cfg.get("q_link_top_k", 3),
        entity_overlap_threshold=adapter_cfg.get("entity_overlap_threshold", 0.3),
        llm_model=adapter_cfg.get("llm_model", "gpt-4o-mini"),
        llm_temperature=adapter_cfg.get("llm_temperature", 0.0),
        llm_max_tokens=adapter_cfg.get("llm_max_tokens", 256),
        llm_retry_max=adapter_cfg.get("llm_retry_max", 3),
        # v3 Hybrid mode
        hybrid_mode=adapter_cfg.get("hybrid_mode", False),
        max_cot_steps=adapter_cfg.get("max_cot_steps", 2),
        # v4 Adaptive Depth
        adaptive_depth=adapter_cfg.get("adaptive_depth", False),
        depth_alpha=adapter_cfg.get("depth_alpha", 0.5),
        max_depth=adapter_cfg.get("max_depth", 4),
        # v5 Two-Edge Architecture
        two_edge_mode=adapter_cfg.get("two_edge_mode", False),
        rerank_alpha=adapter_cfg.get("rerank_alpha", 0.5),
        ctx_max_sent_distance=adapter_cfg.get("ctx_max_sent_distance", 6),
        sim_alpha=adapter_cfg.get("sim_alpha", 0.6),
        sim_beta=adapter_cfg.get("sim_beta", 0.4),
        sim_edge_threshold=adapter_cfg.get("sim_edge_threshold", 0.25),
    )

    # Evaluator
    evaluator = HotpotQAEvaluator()

    # Run
    print(f"[run] Condition: {condition_name}")
    print(f"[run] structural_mode={adapter.structural_mode}, "
          f"gamma_0={adapter.gamma_0}, gamma_1={adapter.gamma_1}, "
          f"hybrid={adapter.hybrid_mode}, "
          f"adaptive_depth={adapter.adaptive_depth}"
          + (f" (alpha={adapter.depth_alpha}, max_depth={adapter.max_depth})"
             if adapter.adaptive_depth else "")
          + (f", two_edge_mode={adapter.two_edge_mode}"
             f" (rerank_alpha={adapter.rerank_alpha})"
             if adapter.two_edge_mode else ""))
    print(f"[run] Output: {output_dir}")

    total = len(examples)
    t0 = time.time()

    with open(results_file, "a" if args.resume else "w") as fout:
        for i, example in enumerate(examples):
            if example.id in completed_ids:
                continue

            try:
                result = adapter.process(example)

                # Evaluate
                eval_result = evaluator.evaluate_single(
                    example_id=example.id,
                    prediction=result.answer,
                    ground_truth=example.answer,
                    predicted_facts=result.retrieved_facts,
                    gold_facts=example.supporting_facts,
                    latency_ms=result.latency_ms,
                    question_type=example.question_type,
                )

                # Write per-example record
                record = {
                    "example_id": example.id,
                    "question": example.question,
                    "question_type": example.question_type,
                    "level": example.level,
                    "ground_truth": example.answer,
                    "prediction": result.answer,
                    "em": eval_result.em,
                    "f1": eval_result.f1,
                    "sf_f1": eval_result.sf_f1,
                    "condition": condition_name,
                    "structural_mode": adapter.structural_mode,
                    "gamma_0": adapter.gamma_0,
                    "gamma_1": adapter.gamma_1,
                    "gedig_score": result.gedig_score,
                    "extended_f": result.extended_f,
                    "betti_0_before": result.betti_0_before,
                    "betti_0_after": result.betti_0_after,
                    "delta_betti_0": result.delta_betti_0,
                    "betti_1_before": result.betti_1_before,
                    "betti_1_after": result.betti_1_after,
                    "delta_betti_1": result.delta_betti_1,
                    "ag_fired": result.ag_fired,
                    "dg_fired": result.dg_fired,
                    "graph_nodes": result.graph_nodes,
                    "graph_edges": result.graph_edges,
                    "expansions": result.expansions,
                    "system_used": result.system_used,
                    "cot_steps": result.cot_steps,
                    "cot_depth": result.cot_depth,
                    "two_edge_mode": adapter.two_edge_mode,
                    "ctx_edges": result.metadata.get("ctx_edges", 0),
                    "sim_edges": result.metadata.get("sim_edges", 0),
                    "latency_ms": result.latency_ms,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                elapsed = time.time() - t0
                avg_ms = (elapsed / (i + 1)) * 1000
                print(
                    f"  [{i+1}/{total}] EM={eval_result.em:.0f} F1={eval_result.f1:.3f} "
                    f"Δβ₀={result.delta_betti_0:+d} Δβ₁={result.delta_betti_1:+d} "
                    f"F={result.extended_f:.3f} "
                    f"({result.latency_ms:.0f}ms, avg {avg_ms:.0f}ms)"
                )

            except Exception as exc:
                print(f"  [{i+1}/{total}] ERROR on {example.id}: {exc}")

            adapter.reset()

    # Aggregate summary
    agg = evaluator.aggregate_by_type()
    summary = {
        "condition": condition_name,
        "structural_mode": adapter.structural_mode,
        "gamma_0": adapter.gamma_0,
        "gamma_1": adapter.gamma_1,
        "total_examples": total,
        "results": {k: v.to_dict() for k, v in agg.items()},
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name}")
    for label, res in agg.items():
        d = res.to_dict()
        print(f"  {label:>12s}: n={d['count']:>5d}  EM={d['em']:.4f}  F1={d['f1']:.4f}  SF-F1={d['sf_f1']:.4f}")
    print(f"{'='*60}")
    print(f"Results: {results_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
