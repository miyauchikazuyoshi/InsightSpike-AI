#!/usr/bin/env python3
"""Build scaled MuSiQue data with additional distractor paragraphs.

Takes musique_random_500.jsonl (20 para/q) and creates a scaled version
(e.g. 50 para/q) by adding random distractor paragraphs from OTHER questions.

Gold supporting paragraphs are always preserved.
All paragraphs are shuffled deterministically for position-bias control.

Usage:
    python build_scaled_data.py \
        --input data/musique_random_500.jsonl \
        --output data/musique_50para_500.jsonl \
        --target-paras 50 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_examples(path: str) -> list[dict]:
    """Load JSONL examples."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def build_paragraph_pool(
    examples: list[dict],
    exclude_id: str,
) -> list[tuple[str, list[str]]]:
    """Collect (title, sentences) from all examples except exclude_id.

    Returns deduplicated pool of paragraphs available as distractors.
    Deduplication by title (case-sensitive) to avoid exact duplicates.
    """
    seen_titles: set[str] = set()
    pool: list[tuple[str, list[str]]] = []

    for ex in examples:
        if ex["id"] == exclude_id:
            continue
        ctx = ex["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            if title not in seen_titles:
                seen_titles.add(title)
                pool.append((title, sents))

    return pool


def augment_example(
    ex: dict,
    pool: list[tuple[str, list[str]]],
    target_paras: int,
    rng: random.Random,
) -> dict:
    """Clone example with additional distractors up to target_paras.

    Steps:
      1. Keep original paragraphs
      2. Sample (target_paras - n_original) from pool
      3. Filter: skip if title already in example (avoids title collision)
      4. Shuffle ALL paragraphs deterministically
      5. Return new dict with augmented context

    Returns a new dict (does not modify the original).
    """
    ctx = ex["context"]
    orig_titles = list(ctx["title"])
    orig_sents = list(ctx["sentences"])
    n_original = len(orig_titles)
    n_needed = target_paras - n_original

    if n_needed <= 0:
        # Already at or above target — just shuffle
        combined = list(zip(orig_titles, orig_sents))
        rng.shuffle(combined)
        new_titles, new_sents = zip(*combined) if combined else ([], [])
        result = dict(ex)
        result["context"] = {
            "title": list(new_titles),
            "sentences": list(new_sents),
        }
        result["original_n_paras"] = n_original
        result["scaled_n_paras"] = n_original
        return result

    # Existing titles to avoid collision
    existing_titles: set[str] = set(orig_titles)

    # Filter pool to exclude already-present titles
    available = [(t, s) for t, s in pool if t not in existing_titles]

    # Sample distractors
    if len(available) < n_needed:
        # Not enough unique paragraphs — use all available
        selected = available
    else:
        selected = rng.sample(available, n_needed)

    # Combine original + distractors
    all_titles = orig_titles + [t for t, _ in selected]
    all_sents = orig_sents + [s for _, s in selected]

    # Shuffle all paragraphs deterministically
    combined = list(zip(all_titles, all_sents))
    rng.shuffle(combined)
    new_titles, new_sents = zip(*combined) if combined else ([], [])

    # Build result
    result = dict(ex)
    result["context"] = {
        "title": list(new_titles),
        "sentences": list(new_sents),
    }
    result["original_n_paras"] = n_original
    result["scaled_n_paras"] = len(new_titles)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build scaled MuSiQue data with additional distractor paragraphs"
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--target-paras", type=int, default=50,
        help="Target number of paragraphs per question (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    print(f"[build] Loading {args.input}")
    examples = load_examples(args.input)
    print(f"[build] Loaded {len(examples)} examples")

    # Stats
    orig_para_counts = [len(ex["context"]["title"]) for ex in examples]
    print(f"[build] Original paragraphs: min={min(orig_para_counts)}, "
          f"max={max(orig_para_counts)}, avg={sum(orig_para_counts)/len(orig_para_counts):.1f}")

    # Process each example
    augmented = []
    for i, ex in enumerate(examples):
        rng = random.Random(args.seed + i)  # Per-example deterministic seed
        pool = build_paragraph_pool(examples, exclude_id=ex["id"])
        aug = augment_example(ex, pool, args.target_paras, rng)
        augmented.append(aug)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(examples)}] processed")

    # Validate
    n_validated = 0
    for aug in augmented:
        titles = aug["context"]["title"]
        sents = aug["context"]["sentences"]
        # Paragraph count
        assert len(titles) == args.target_paras, \
            f'{aug["id"]}: expected {args.target_paras} paras, got {len(titles)}'
        assert len(sents) == args.target_paras, \
            f'{aug["id"]}: sents count mismatch'
        # Gold paragraphs preserved
        for sf_title in aug["supporting_facts"]["title"]:
            assert sf_title in titles, \
                f'{aug["id"]}: gold paragraph "{sf_title}" missing'
        # Note: original MuSiQue data already contains duplicate titles
        # within examples, so we don't assert title uniqueness.
        # The augment logic avoids adding pool paras whose title already
        # exists in the example (via existing_titles set check).
        n_validated += 1

    print(f"[build] Validated {n_validated}/{len(augmented)} examples")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for aug in augmented:
            f.write(json.dumps(aug, ensure_ascii=False) + "\n")

    # Stats
    new_para_counts = [len(aug["context"]["title"]) for aug in augmented]
    avg_tokens = sum(
        sum(len(" ".join(s).split()) for s in aug["context"]["sentences"])
        for aug in augmented
    ) / len(augmented)

    print(f"\n[build] Output: {out_path}")
    print(f"[build] Scaled paragraphs: {new_para_counts[0]} per question")
    print(f"[build] Avg tokens per question: ~{avg_tokens:.0f}")
    print(f"[build] Estimated context window usage: ~{avg_tokens/128000*100:.1f}% of GPT-4o 128K")


if __name__ == "__main__":
    main()
