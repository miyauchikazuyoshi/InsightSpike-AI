#!/usr/bin/env python3
"""Download and prepare BRIGHT benchmark data from HuggingFace.

Creates per-domain JSONL files for queries and documents.

Usage:
  .venv/bin/python3 experiments/hotpotqa_v2/scripts/prepare_bright.py \
      --domains biology,stackoverflow,economics \
      --output experiments/hotpotqa_v2/data/bright/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ALL_DOMAINS = [
    "biology", "earth_science", "economics", "psychology", "robotics",
    "stackoverflow", "sustainable_living", "pony", "leetcode", "aops",
    "theoremqa_theorems", "theoremqa_questions",
]


def main():
    parser = argparse.ArgumentParser(description="Prepare BRIGHT data")
    parser.add_argument(
        "--domains", type=str, default="biology,stackoverflow,economics",
        help="Comma-separated list of domains",
    )
    parser.add_argument(
        "--output", type=str,
        default="experiments/hotpotqa_v2/data/bright/",
    )
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install: pip install datasets")
        return

    for domain in domains:
        print(f"\n--- {domain} ---")

        # Queries
        queries_file = out_dir / f"{domain}_queries.jsonl"
        if queries_file.exists():
            n = sum(1 for _ in open(queries_file))
            print(f"  Queries: {queries_file} already exists ({n} rows)")
        else:
            print(f"  Loading queries...")
            ds_q = load_dataset("xlangai/BRIGHT", "examples", split=domain)
            with open(queries_file, "w") as f:
                for row in ds_q:
                    record = {
                        "id": row["id"],
                        "query": row["query"],
                        "gold_ids": row["gold_ids"],
                        "excluded_ids": row["excluded_ids"],
                        "gold_answer": row["gold_answer"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  Queries: {len(ds_q)} rows -> {queries_file}")

        # Documents
        docs_file = out_dir / f"{domain}_docs.jsonl"
        if docs_file.exists():
            n = sum(1 for _ in open(docs_file))
            print(f"  Docs: {docs_file} already exists ({n} rows)")
        else:
            print(f"  Loading documents...")
            ds_d = load_dataset("xlangai/BRIGHT", "documents", split=domain)
            with open(docs_file, "w") as f:
                for row in ds_d:
                    record = {
                        "id": row["id"],
                        "content": row["content"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  Docs: {len(ds_d)} rows -> {docs_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
