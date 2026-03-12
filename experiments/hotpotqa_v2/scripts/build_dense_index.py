#!/usr/bin/env python3
"""Build dense (E5-base-v2 + FAISS) indices for BRIGHT domains.

Encodes all documents per domain and saves FAISS index, doc IDs, and
embedding matrix to disk.

Usage:
  PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
      experiments/hotpotqa_v2/scripts/build_dense_index.py \
      --data-dir experiments/hotpotqa_v2/data/bright \
      --index-dir experiments/hotpotqa_v2/data/bright/dense_index \
      --domains biology,economics,stackoverflow
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dense_retriever import DenseRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build dense indices for BRIGHT")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing {domain}_docs.jsonl files",
    )
    parser.add_argument(
        "--index-dir", type=str, required=True,
        help="Output directory for FAISS indices",
    )
    parser.add_argument(
        "--domains", type=str, default="biology,economics,stackoverflow",
        help="Comma-separated list of domains",
    )
    parser.add_argument(
        "--model", type=str, default="intfloat/e5-base-v2",
        help="SentenceTransformer model name",
    )
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    data_dir = Path(args.data_dir)

    retriever = DenseRetriever(model_name=args.model, index_dir=args.index_dir)

    total_t0 = time.time()

    for domain in domains:
        docs_path = data_dir / f"{domain}_docs.jsonl"
        if not docs_path.exists():
            logger.error("Not found: %s — skipping", docs_path)
            continue

        logger.info("=== Building index for %s ===", domain)
        t0 = time.time()
        retriever.build_index(docs_path, domain)
        elapsed = time.time() - t0
        logger.info("  Done in %.1fs", elapsed)

    total_elapsed = time.time() - total_t0
    logger.info("=== All done in %.1fs ===", total_elapsed)


if __name__ == "__main__":
    main()
