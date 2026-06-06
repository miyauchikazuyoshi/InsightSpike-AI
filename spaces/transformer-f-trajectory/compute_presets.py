#!/usr/bin/env python3
"""Pre-compute F-trajectory for preset sentences.

Run once before deploying / showing the demo so the preset switcher in
app.py is instantaneous (no model inference at switch time).

Usage:
    python compute_presets.py [--model bert-base-uncased] [--device mps]

Output:
    presets.json — { preset_id: { trajectory: {...}, note: ..., text: ... } }
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lib import f_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--anchor-idx", type=int, default=0)
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epc-method", default="vector", choices=["vector", "similarity"])
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
    model, tokenizer = f_trajectory.load_model(args.model, device=args.device)
    print(f"  → ready ({time.time() - t0:.1f}s)", flush=True)

    results: dict = {
        "config": {
            "model": args.model,
            "anchor_idx": args.anchor_idx,
            "lambda": args.lambda_,
            "gamma": args.gamma,
            "epc_method": args.epc_method,
        },
        "categories": {},
    }

    n_processed = 0
    for cat_name, items in config["categories"].items():
        results["categories"][cat_name] = []
        for item in items:
            t0 = time.time()
            traj = f_trajectory.compute(
                model,
                tokenizer,
                item["text"],
                model_name=args.model,
                anchor_idx=args.anchor_idx,
                lambda_=args.lambda_,
                gamma=args.gamma,
                epc_method=args.epc_method,
                device=args.device,
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
                f"  [{cat_name}/{item['id']}] {len(item['text'])} chars, "
                f"{traj.num_layers} layers, total_F={traj.total_f:.3f} ({elapsed:.1f}s)",
                flush=True,
            )

    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {n_processed} presets → {args.output}", flush=True)


if __name__ == "__main__":
    main()
