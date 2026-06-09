#!/usr/bin/env python3
"""Pre-compute attention-based F-trajectories for preset sentences.

Matches the canonical formula used in the JSAI 2026 paper Section 3
(experiments/transformer/extract_and_score.py and
src/insightspike/algorithms/gedig/attention.py).

Usage:
    python compute_presets.py [--model bert-base-uncased] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lib import attention_f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_",
                        help="Match Phase 1 setting (0.5)")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--percentile", type=float, default=0.9,
                        help="Top-X percentile for thresholding attention")
    parser.add_argument("--n-random", type=int, default=3,
                        help="Random baseline matrices per layer")
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--input",
        default=Path(__file__).parent / "preset_sentences.json",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).parent / "presets.json",
        type=Path,
    )
    args = parser.parse_args()

    with open(args.input) as f:
        config = json.load(f)

    print(f"Loading model: {args.model}", flush=True)
    t0 = time.time()
    model, tokenizer = attention_f.load_model(args.model, device=args.device)
    print(f"  → ready ({time.time() - t0:.1f}s)", flush=True)

    results: dict = {
        "config": {
            "model": args.model,
            "lambda": args.lambda_,
            "gamma": args.gamma,
            "percentile": args.percentile,
            "n_random_per_layer": args.n_random,
            "rng_seed": args.rng_seed,
            "formula": "F = ΔEPC − λ·γ·ΔSP − λ·ΔH  (attention-based, Phase 1)",
        },
        "categories": {},
    }

    n_processed = 0
    for cat_name, items in config["categories"].items():
        results["categories"][cat_name] = []
        for item in items:
            t0 = time.time()
            traj = attention_f.compute(
                model,
                tokenizer,
                item["text"],
                model_name=args.model,
                lambda_=args.lambda_,
                gamma=args.gamma,
                percentile=args.percentile,
                device=args.device,
                n_random=args.n_random,
                rng_seed=args.rng_seed,
            )
            elapsed = time.time() - t0
            results["categories"][cat_name].append({
                "id": item["id"],
                "text": item["text"],
                "note": item.get("note", ""),
                "trajectory": traj.to_dict(),
            })
            n_processed += 1
            print(
                f"  [{cat_name}/{item['id']}] "
                f"L={traj.num_layers} H={traj.num_heads} tok={traj.num_tokens}  "
                f"F_real={traj.f_mean_real:+.4f}  F_rand={traj.f_mean_random:+.4f}  "
                f"ΔF={traj.delta_f:+.4f}  win={traj.win_rate:.0%}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {n_processed} presets → {args.output}", flush=True)


if __name__ == "__main__":
    main()
