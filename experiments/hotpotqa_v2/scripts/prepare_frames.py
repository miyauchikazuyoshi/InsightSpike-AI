"""Prepare FRAMES benchmark data in HotpotQA-compatible JSONL format.

Downloads the FRAMES dataset from HuggingFace, fetches Wikipedia article
content from the gold links, and converts to the same JSONL schema used
by our HotpotQA pipeline.

Usage:
    PYTHONPATH=src .venv/bin/python3 experiments/hotpotqa_v2/scripts/prepare_frames.py \
        --output experiments/hotpotqa_v2/data/frames_benchmark.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def fetch_wikipedia_text(url: str, max_retries: int = 3) -> tuple[str, list[str]]:
    """Fetch plain text from a Wikipedia URL via the MediaWiki API.

    Returns (title, list_of_sentences).
    """
    # Extract title from URL
    # e.g. https://en.wikipedia.org/wiki/James_Buchanan -> James_Buchanan
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    if "/wiki/" in path:
        title = path.split("/wiki/")[-1]
        title = urllib.parse.unquote(title)
    else:
        return ("", [])

    # Use MediaWiki API to get plain text extract
    api_url = (
        "https://en.wikipedia.org/w/api.php?"
        "action=query&format=json&prop=extracts&explaintext=1"
        f"&titles={urllib.parse.quote(title)}"
    )

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "InsightSpike-AI/1.0 (research)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    return (title.replace("_", " "), [])
                page_title = page_data.get("title", title.replace("_", " "))
                extract = page_data.get("extract", "")
                sentences = _split_into_sentences(extract)
                return (page_title, sentences)

            return (title.replace("_", " "), [])

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                print(f"  WARN: Failed to fetch {title}: {e}", file=sys.stderr)
                return (title.replace("_", " "), [])

    return (title.replace("_", " "), [])


def _split_into_sentences(text: str) -> list[str]:
    """Split Wikipedia extract into sentences, filtering noise."""
    if not text:
        return []

    # Remove section headers (== Header ==)
    text = re.sub(r"={2,}[^=]+=+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Simple sentence splitting
    raw = re.split(r"(?<=[.!?])\s+", text)

    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < 10:
            continue
        if len(s) > 500:
            # Split long sentences at semicolons or commas
            parts = re.split(r"[;,]\s+", s)
            for p in parts:
                p = p.strip()
                if len(p) >= 10:
                    sentences.append(p)
        else:
            sentences.append(s)

    # Cap at ~50 sentences per article to keep context manageable
    return sentences[:50]


def convert_frames_to_hotpotqa(
    frames_data: list[dict],
    limit: int | None = None,
) -> list[dict]:
    """Convert FRAMES examples to HotpotQA JSONL format."""
    results = []
    total = len(frames_data) if limit is None else min(limit, len(frames_data))

    for idx, ex in enumerate(frames_data[:total]):
        question = ex["Prompt"]
        answer = ex["Answer"]
        reasoning_type = ex.get("reasoning_types", "unknown")

        # Get wiki links
        wiki_links = ex.get("wiki_links")
        if isinstance(wiki_links, str):
            try:
                wiki_links = ast.literal_eval(wiki_links)
            except (ValueError, SyntaxError):
                wiki_links = []
        if not wiki_links:
            # Fallback: collect from individual link columns
            wiki_links = []
            for i in range(1, 16):
                link = ex.get(f"wikipedia_link_{i}")
                if link:
                    wiki_links.append(link)

        print(f"  [{idx+1}/{total}] Fetching {len(wiki_links)} articles...", end="", flush=True)

        # Fetch Wikipedia articles
        context_titles = []
        context_sentences = []
        for url in wiki_links:
            if not url or not isinstance(url, str):
                continue
            title, sents = fetch_wikipedia_text(url)
            if title and sents:
                context_titles.append(title)
                context_sentences.append(sents)
            time.sleep(0.1)  # Be nice to Wikipedia API

        print(f" got {len(context_titles)} articles")

        if not context_titles:
            print(f"  WARN: No context for question {idx}, skipping")
            continue

        # Build HotpotQA-compatible record
        # supporting_facts: treat first sentence of each article as "supporting"
        # (FRAMES doesn't have sentence-level gold labels)
        sf_titles = context_titles[:]
        sf_sent_ids = [0] * len(context_titles)

        record = {
            "id": f"frames_{idx:04d}",
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
            "type": reasoning_type,
            "level": "hard",
        }
        results.append(record)

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare FRAMES benchmark data")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/hotpotqa_v2/data/frames_benchmark.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    # Load FRAMES from HuggingFace
    print("Loading FRAMES dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("google/frames-benchmark", split="test")
    frames_data = [dict(ex) for ex in ds]
    print(f"Loaded {len(frames_data)} questions")

    # Convert
    print("Fetching Wikipedia articles and converting...")
    records = convert_frames_to_hotpotqa(frames_data, limit=args.limit)

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(records)} examples to {output_path}")

    # Stats
    types = {}
    for r in records:
        t = r["type"]
        types[t] = types.get(t, 0) + 1
    print("\nReasoning type distribution:")
    for t, c in sorted(types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
