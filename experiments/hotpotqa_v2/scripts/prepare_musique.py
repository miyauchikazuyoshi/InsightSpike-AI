"""Prepare MuSiQue benchmark data in HotpotQA-compatible JSONL format.

MuSiQue (Multihop Questions via Single-hop Question Composition) provides
20 paragraphs per question (2-4 gold + 16-18 distractors) with is_supporting
annotations. This is an ideal benchmark for geDIG's topology-based gauge.

Usage:
    PYTHONPATH=src .venv/bin/python3 experiments/hotpotqa_v2/scripts/prepare_musique.py \
        --output experiments/hotpotqa_v2/data/musique_dev.jsonl

    # With limit:
    PYTHONPATH=src .venv/bin/python3 experiments/hotpotqa_v2/scripts/prepare_musique.py \
        --output experiments/hotpotqa_v2/data/musique_sample_100.jsonl --limit 100
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _split_paragraph_into_sentences(text: str) -> list[str]:
    """Split paragraph text into sentences."""
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r"(?<=[.!?])\s+", text)

    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < 5:
            continue
        sentences.append(s)

    return sentences


def convert_musique_to_hotpotqa(examples: list[dict], limit: int | None = None) -> list[dict]:
    """Convert MuSiQue examples to HotpotQA JSONL format.

    MuSiQue format:
        - paragraphs: list of {idx, title, paragraph_text, is_supporting}
        - question_decomposition: list of {id, question, answer, paragraph_support_idx}

    HotpotQA format:
        - context: {title: [...], sentences: [[...], ...]}
        - supporting_facts: {title: [...], sent_id: [...]}
    """
    results = []
    total = len(examples) if limit is None else min(limit, len(examples))

    hop_dist = {}

    for idx, ex in enumerate(examples[:total]):
        question = ex["question"]
        answer = ex["answer"]
        paragraphs = ex["paragraphs"]
        decomposition = ex["question_decomposition"]
        n_hops = len(decomposition)

        hop_dist[n_hops] = hop_dist.get(n_hops, 0) + 1

        # Build context: all 20 paragraphs
        context_titles = []
        context_sentences = []
        for para in paragraphs:
            title = para["title"]
            sents = _split_paragraph_into_sentences(para["paragraph_text"])
            context_titles.append(title)
            context_sentences.append(sents)

        # Build supporting facts from is_supporting + decomposition
        sf_titles = []
        sf_sent_ids = []

        for para in paragraphs:
            if para["is_supporting"]:
                title = para["title"]
                # First sentence of supporting paragraph is the key fact
                # (MuSiQue paragraphs are typically 1-3 sentences)
                sf_titles.append(title)
                sf_sent_ids.append(0)

        # Determine question type based on hop count
        if n_hops == 2:
            q_type = "bridge"  # 2-hop = bridge-like
        elif n_hops == 3:
            q_type = "bridge_3hop"
        else:
            q_type = "bridge_4hop"

        record = {
            "id": ex["id"],
            "question": question,
            "answer": answer,
            "supporting_facts": {
                "title": sf_titles,
                "sent_id": sf_sent_ids,
            },
            "context": {
                "title": context_titles,
                "sentences": context_sentences,
            },
            "type": q_type,
            "level": "hard" if n_hops >= 3 else "medium",
            # MuSiQue-specific metadata
            "musique_hops": n_hops,
            "musique_decomposition": [
                {"question": d["question"], "answer": d["answer"]}
                for d in decomposition
            ],
        }
        results.append(record)

    return results, hop_dist


def main():
    parser = argparse.ArgumentParser(description="Prepare MuSiQue benchmark data")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/hotpotqa_v2/data/musique_dev.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    # Load MuSiQue from HuggingFace
    print("Loading MuSiQue dataset from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    musique_data = [dict(ex) for ex in ds]
    print(f"Loaded {len(musique_data)} questions")

    # Convert
    print("Converting to HotpotQA format...")
    records, hop_dist = convert_musique_to_hotpotqa(musique_data, limit=args.limit)

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(records)} examples to {output_path}")

    # Stats
    print(f"\nHop distribution:")
    for h, c in sorted(hop_dist.items()):
        print(f"  {h}-hop: {c} ({100*c/len(records):.1f}%)")

    # Supporting facts stats
    sf_counts = [len(r["supporting_facts"]["title"]) for r in records]
    avg_sf = sum(sf_counts) / len(sf_counts)
    print(f"\nSupporting facts per question: avg={avg_sf:.1f}, min={min(sf_counts)}, max={max(sf_counts)}")

    # Context stats
    ctx_counts = [len(r["context"]["title"]) for r in records]
    avg_ctx = sum(ctx_counts) / len(ctx_counts)
    print(f"Context paragraphs per question: avg={avg_ctx:.1f}")

    sent_counts = [sum(len(s) for s in r["context"]["sentences"]) for r in records]
    avg_sent = sum(sent_counts) / len(sent_counts)
    print(f"Total sentences per question: avg={avg_sent:.1f}")


if __name__ == "__main__":
    main()
