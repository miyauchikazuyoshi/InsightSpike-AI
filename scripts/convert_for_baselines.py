#!/usr/bin/env python3
"""Convert RAG JSONL datasets into baseline-friendly formats.

Currently supports:
  - GraphRAG: documents.tsv + questions.jsonl

Usage:
  python scripts/convert_for_baselines.py \
      --input data/sample_queries.jsonl \
      --output-dir experiments/rag-baselines-data/graph_rag
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_samples(path: Path) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def extract_documents(samples: Iterable[Dict]) -> Dict[str, Tuple[str, Dict]]:
    """Collect unique documents from dataset samples."""
    docs: Dict[str, Tuple[str, Dict]] = {}
    for sample in samples:
        # Newer datasets store episodes
        if "episodes" in sample:
            for episode in sample["episodes"]:
                doc_id = episode.get("id")
                text = episode.get("text")
                if not doc_id or not text:
                    continue
                metadata = {k: v for k, v in episode.items() if k not in {"id", "text"}}
                docs.setdefault(doc_id, (text, metadata))
        # Legacy structure: documents array
        elif "documents" in sample:
            for doc in sample["documents"]:
                doc_id = doc.get("id")
                text = doc.get("text")
                if not doc_id or not text:
                    continue
                metadata = {k: v for k, v in doc.get("metadata", {}).items()}
                docs.setdefault(doc_id, (text, metadata))
    return docs


def write_graph_rag_artifacts(samples: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write documents.tsv
    docs = extract_documents(samples)
    docs_path = out_dir / "documents.tsv"
    with docs_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["doc_id", "title", "content"])
        for doc_id, (text, metadata) in docs.items():
            title = metadata.get("goal") or metadata.get("context") or doc_id
            writer.writerow([doc_id, title, text])

    # Write questions.jsonl
    questions_path = out_dir / "questions.jsonl"
    with questions_path.open("w", encoding="utf-8") as fh:
        for idx, sample in enumerate(samples):
            payload = {
                "id": sample.get("id", f"q_{idx:04d}"),
                "question": sample["query"],
                "answer": sample.get("ground_truth", ""),
                "metadata": {
                    "template": sample.get("template"),
                    "domain": sample.get("domain"),
                },
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Input JSONL dataset")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    ap.add_argument(
        "--targets",
        nargs="*",
        default=["graphrag"],
        choices=["graphrag"],
        help="Conversion targets (default: graphrag)",
    )
    args = ap.parse_args()

    samples = load_samples(args.input)
    if "graphrag" in args.targets:
        write_graph_rag_artifacts(samples, args.output_dir / "graphrag")

    print(f"[convert] wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
