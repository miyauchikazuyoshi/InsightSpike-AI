#!/usr/bin/env python3
"""Episodify BRIGHT queries using synchronous LLM calls.

Decomposes query text into atomic reasoning episodes for geDIG routing.
Queries are typically 1-5 sentences, so all are processed via LLM.

Usage::

    PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
        experiments/hotpotqa_v2/scripts/episodify_queries.py \
        --data-dir experiments/hotpotqa_v2/data/bright/ \
        --domain biology \
        --output-dir experiments/hotpotqa_v2/data/bright/episodes/ \
        --model gpt-4o-mini

Outputs:
    {output_dir}/{domain}_query_episodes.jsonl
    {output_dir}/{domain}_query_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt for query episodification
# ------------------------------------------------------------------ #

_QUERY_EPISODIFY_PROMPT = """Decompose the following query into atomic reasoning episodes.
Each episode is ONE reasoning step: a premise, hypothesis, question, evidence, or constraint.

Rules:
- Each episode must be self-contained
- Use "connects_to" to reference earlier episode IDs that this one depends on
- Maximum 8 episodes per query
- Types: premise, hypothesis, question, evidence, constraint

Output ONLY a JSON array (no markdown, no explanation):
[{{"id": 0, "text": "...", "type": "...", "connects_to": []}}, ...]

Query:
{query}"""


def parse_query_episodes(response_text: str, query_id: str) -> dict | None:
    """Parse LLM response into episode structure."""
    try:
        text = response_text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()
        episodes = json.loads(text)
        if not isinstance(episodes, list) or len(episodes) == 0:
            raise ValueError("Empty or non-list response")

        valid_types = {"premise", "hypothesis", "question", "evidence",
                       "constraint", "definition", "mechanism", "claim"}
        normalized = []
        for ep in episodes[:8]:
            ep_type = ep.get("type", "premise")
            if ep_type not in valid_types:
                ep_type = "premise"
            connects = ep.get("connects_to", [])
            if not isinstance(connects, list):
                connects = []
            connects = [c for c in connects
                        if isinstance(c, int) and 0 <= c < ep.get("id", 0)]
            normalized.append({
                "id": len(normalized),
                "text": str(ep.get("text", ""))[:500],
                "type": ep_type,
                "connects_to": connects,
            })
        return {"query_id": query_id, "episodes": normalized, "method": "llm"}

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        logger.warning("Parse error for query %s: %s", query_id, e)
        return None


def episodify_query_fallback(query_id: str, query_text: str) -> dict:
    """Fallback: treat entire query as single episode."""
    return {
        "query_id": query_id,
        "episodes": [
            {"id": 0, "text": query_text[:500], "type": "question",
             "connects_to": []}
        ],
        "method": "fallback",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Episodify BRIGHT queries"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with {domain}_queries.jsonl")
    parser.add_argument("--domain", required=True,
                        help="Domain to process (e.g. biology)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for episodes")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit queries to process (for testing)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries_path = data_dir / f"{args.domain}_queries.jsonl"
    if not queries_path.exists():
        logger.error("Queries file not found: %s", queries_path)
        sys.exit(1)

    # Load queries
    queries = []
    with open(queries_path) as f:
        for line in f:
            queries.append(json.loads(line))
    logger.info("Loaded %d queries for domain=%s", len(queries), args.domain)

    if args.limit:
        queries = queries[:args.limit]
        logger.info("  Limited to %d queries", len(queries))

    # Initialize LLM
    from answerer import LLMAnswerer
    llm = LLMAnswerer(model=args.model, temperature=0.0, max_tokens=600)

    results = []
    n_llm = 0
    n_fallback = 0

    for i, q in enumerate(queries):
        query_id = q["id"]
        query_text = q["query"]

        prompt = _QUERY_EPISODIFY_PROMPT.format(query=query_text[:2000])
        try:
            response = llm._llm_call_raw(prompt, max_tokens=600)
            result = parse_query_episodes(response, query_id)
            if result is None:
                result = episodify_query_fallback(query_id, query_text)
                n_fallback += 1
            else:
                n_llm += 1
        except Exception as e:
            logger.warning("LLM error for query %s: %s", query_id, e)
            result = episodify_query_fallback(query_id, query_text)
            n_fallback += 1

        results.append(result)
        if (i + 1) % 20 == 0:
            logger.info("  Processed %d/%d queries (llm=%d, fallback=%d)",
                        i + 1, len(queries), n_llm, n_fallback)

    # Write results
    output_path = output_dir / f"{args.domain}_query_episodes.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d query episode records to %s", len(results), output_path)

    # Statistics
    import numpy as np
    ep_counts = [len(r["episodes"]) for r in results]
    ep_arr = np.array(ep_counts) if ep_counts else np.array([0])

    type_counts: dict[str, int] = {}
    connects_counts = []
    for r in results:
        for ep in r["episodes"]:
            t = ep.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            connects_counts.append(len(ep.get("connects_to", [])))

    stats = {
        "domain": args.domain,
        "n_queries": len(results),
        "n_llm": n_llm,
        "n_fallback": n_fallback,
        "total_episodes": sum(ep_counts),
        "episodes_per_query": {
            "mean": round(float(ep_arr.mean()), 2),
            "median": round(float(np.median(ep_arr)), 1),
            "min": int(ep_arr.min()),
            "max": int(ep_arr.max()),
        },
        "type_distribution": dict(sorted(type_counts.items(),
                                          key=lambda x: -x[1])),
        "avg_connects_to": round(float(np.mean(connects_counts)), 2)
        if connects_counts else 0,
    }
    stats_path = output_dir / f"{args.domain}_query_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Statistics: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
