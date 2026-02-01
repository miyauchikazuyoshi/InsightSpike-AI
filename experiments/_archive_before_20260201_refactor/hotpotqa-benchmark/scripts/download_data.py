#!/usr/bin/env python3
"""Download HotpotQA dataset and prepare for geDIG experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def download_hotpotqa() -> Path:
    """Download HotpotQA distractor dev set using datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Please install datasets: pip install datasets")
        raise

    print("[download] Loading HotpotQA distractor dev set...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "hotpotqa_distractor_dev.jsonl"

    print(f"[download] Converting to JSONL format: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            record = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": {
                    "title": item["supporting_facts"]["title"],
                    "sent_id": item["supporting_facts"]["sent_id"],
                },
                "context": {
                    "title": item["context"]["title"],
                    "sentences": item["context"]["sentences"],
                },
                "type": item["type"],
                "level": item["level"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[download] Saved {len(dataset)} examples to {output_path}")
    return output_path


def create_sample(input_path: Path, sample_size: int, seed: int = 42) -> Path:
    """Create a random sample from the full dataset."""
    random.seed(seed)

    print(f"[sample] Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if sample_size >= len(lines):
        print(f"[sample] Requested {sample_size} but only {len(lines)} available. Using all.")
        sample_size = len(lines)

    sampled = random.sample(lines, sample_size)

    output_path = DATA_DIR / f"hotpotqa_sample_{sample_size}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(sampled)

    print(f"[sample] Saved {sample_size} examples to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare HotpotQA data")
    parser.add_argument("--sample", type=int, nargs="*", default=[100, 500],
                        help="Sample sizes to create (default: 100, 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if data already exists")
    args = parser.parse_args()

    full_path = DATA_DIR / "hotpotqa_distractor_dev.jsonl"

    if args.skip_download and full_path.exists():
        print(f"[download] Skipping download, using existing {full_path}")
    else:
        full_path = download_hotpotqa()

    for size in args.sample:
        create_sample(full_path, size, args.seed)

    print("[done] Data preparation complete!")
    print(f"  Full dataset: {full_path}")
    for size in args.sample:
        print(f"  Sample ({size}): {DATA_DIR / f'hotpotqa_sample_{size}.jsonl'}")


if __name__ == "__main__":
    main()
