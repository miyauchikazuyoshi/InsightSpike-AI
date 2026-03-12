#!/usr/bin/env python3
"""Episodify BRIGHT corpus documents using OpenAI Batch API.

Decomposes documents into atomic knowledge episodes for geDIG routing.
Short documents (≤3 sentences) are heuristically converted without LLM.
Longer documents use gpt-4o-mini via the Batch API (50% discount).

Usage::

    PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
        experiments/hotpotqa_v2/scripts/episodify_corpus.py \
        --data-dir experiments/hotpotqa_v2/data/bright/ \
        --domain biology \
        --output-dir experiments/hotpotqa_v2/data/bright/episodes/ \
        --model gpt-4o-mini --batch-api

Outputs:
    {output_dir}/{domain}_episodes.jsonl
    {output_dir}/{domain}_batch_requests.jsonl  (Batch API requests)
    {output_dir}/{domain}_batch_id.txt          (Batch job ID)
    {output_dir}/{domain}_stats.json            (Statistics)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt for document episodification
# ------------------------------------------------------------------ #

_EPISODIFY_PROMPT = (
    "Decompose the following document into atomic knowledge episodes.\n"
    "Each episode is ONE atomic fact, definition, mechanism, or claim.\n\n"
    "Rules:\n"
    "- Each episode must be self-contained (understandable without other episodes)\n"
    '- Use "connects_to" to reference earlier episodes that this one logically depends on\n'
    "- Maximum 10 episodes per document\n"
    "- Types: definition, mechanism, evidence, claim, example, constraint\n\n"
    'Output ONLY a JSON array (no markdown, no explanation):\n'
    '[{{"id": 0, "text": "...", "type": "...", "connects_to": []}}, ...]\n\n'
    "Document:\n{content}"
)

# Sentence splitting pattern
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    sents = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 10]


def episodify_heuristic(doc_id: str, content: str) -> dict:
    """Heuristic episodification for short documents (≤3 sentences)."""
    sents = split_sentences(content)
    if len(sents) <= 1:
        episodes = [{"id": 0, "text": content.strip(), "type": "single",
                      "connects_to": []}]
    else:
        episodes = []
        for i, sent in enumerate(sents):
            ep = {
                "id": i,
                "text": sent,
                "type": "claim",
                "connects_to": [i - 1] if i > 0 else [],
            }
            episodes.append(ep)
    return {"doc_id": doc_id, "episodes": episodes, "method": "heuristic"}


def parse_llm_episodes(response_text: str, doc_id: str, content: str) -> dict:
    """Parse LLM response into episode structure with validation."""
    try:
        # Try to extract JSON array from response
        text = response_text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()
        parsed = json.loads(text)
        # Handle multiple response formats:
        # 1. Direct list: [{"id": 0, ...}, ...]
        # 2. Wrapper object: {"episodes": [{"id": 0, ...}, ...]}
        # 3. Single object: {"id": 0, "text": "..."}
        if isinstance(parsed, list):
            episodes = parsed
        elif isinstance(parsed, dict):
            if "episodes" in parsed and isinstance(parsed["episodes"], list):
                episodes = parsed["episodes"]
            elif "text" in parsed:
                # Single episode object
                episodes = [parsed]
            else:
                raise ValueError("Dict response without 'episodes' key or 'text'")
        else:
            raise ValueError(f"Unexpected response type: {type(parsed)}")
        if len(episodes) == 0:
            raise ValueError("Empty episode list")

        # Validate and normalize
        valid_types = {"definition", "mechanism", "evidence", "claim",
                       "example", "constraint", "single", "premise",
                       "hypothesis", "question"}
        normalized = []
        for ep in episodes[:10]:  # Max 10 episodes
            # Handle string items (e.g. LLM returned ["text1", "text2"])
            if isinstance(ep, str):
                normalized.append({
                    "id": len(normalized),
                    "text": ep[:500],
                    "type": "claim",
                    "connects_to": [],
                })
                continue
            if not isinstance(ep, dict):
                continue
            ep_type = ep.get("type", "claim")
            if ep_type not in valid_types:
                ep_type = "claim"
            connects = ep.get("connects_to", [])
            if not isinstance(connects, list):
                connects = []
            # Filter out invalid references
            connects = [c for c in connects
                        if isinstance(c, int) and 0 <= c < ep.get("id", 0)]
            normalized.append({
                "id": len(normalized),
                "text": str(ep.get("text", ""))[:500],
                "type": ep_type,
                "connects_to": connects,
            })
        return {"doc_id": doc_id, "episodes": normalized, "method": "llm"}

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        logger.warning("Parse error for %s: %s. Falling back to heuristic.", doc_id, e)
        return episodify_heuristic(doc_id, content)


# ------------------------------------------------------------------ #
# Batch API operations
# ------------------------------------------------------------------ #

def create_batch_requests(
    docs_jsonl_path: Path,
    output_path: Path,
    model: str = "gpt-4o-mini",
) -> tuple[int, int, list[dict]]:
    """Create Batch API request JSONL and heuristic results.

    Returns
    -------
    n_heuristic, n_llm, heuristic_results
    """
    heuristic_results = []
    n_heuristic = 0
    n_llm = 0

    with open(docs_jsonl_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            doc = json.loads(line)
            doc_id = doc["id"]
            content = doc["content"]

            sents = split_sentences(content)
            if len(sents) <= 3:
                # Heuristic — no LLM needed
                result = episodify_heuristic(doc_id, content)
                heuristic_results.append(result)
                n_heuristic += 1
            else:
                # LLM request for Batch API
                prompt = _EPISODIFY_PROMPT.format(content=content[:1500])
                request = {
                    "custom_id": doc_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.0,
                        "max_tokens": 800,
                        "response_format": {"type": "json_object"},
                    },
                }
                fout.write(json.dumps(request) + "\n")
                n_llm += 1

            if (i + 1) % 10000 == 0:
                logger.info("  Processed %d docs (heuristic=%d, llm=%d)",
                            i + 1, n_heuristic, n_llm)

    logger.info("Total: %d docs — heuristic=%d, llm=%d", n_heuristic + n_llm,
                n_heuristic, n_llm)
    return n_heuristic, n_llm, heuristic_results


def submit_batch(requests_path: Path, domain: str) -> str:
    """Upload request file and submit batch job."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)

    logger.info("Uploading batch request file: %s", requests_path)
    with open(requests_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info("  File ID: %s", file_obj.id)

    logger.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"domain": domain, "task": "episodify_corpus"},
    )
    logger.info("  Batch ID: %s, status: %s", batch.id, batch.status)
    return batch.id


def poll_batch(batch_id: str, poll_interval: int = 60) -> dict:
    """Poll batch job until completion."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        logger.info("  Batch %s: status=%s, completed=%s/%s, failed=%s",
                     batch_id, status,
                     counts.completed if counts else "?",
                     counts.total if counts else "?",
                     counts.failed if counts else "?")

        if status == "completed":
            return {
                "status": "completed",
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "counts": {
                    "total": counts.total if counts else 0,
                    "completed": counts.completed if counts else 0,
                    "failed": counts.failed if counts else 0,
                },
            }
        elif status in ("failed", "expired", "cancelled"):
            errors = []
            if hasattr(batch, "errors") and batch.errors:
                errors = [str(e) for e in batch.errors.data[:5]]
            return {"status": status, "errors": errors}
        else:
            time.sleep(poll_interval)


def download_batch_results(output_file_id: str) -> list[dict]:
    """Download and parse batch output file."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    logger.info("Downloading batch results (file_id=%s)...", output_file_id)
    content = client.files.content(output_file_id)
    raw_text = content.text

    results = []
    for line in raw_text.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        results.append(obj)

    logger.info("  Downloaded %d results", len(results))
    return results


def process_batch_results(
    raw_results: list[dict],
    docs_jsonl_path: Path,
) -> list[dict]:
    """Convert batch API results to episode records."""
    # Build doc_id → content lookup for fallback
    doc_contents = {}
    with open(docs_jsonl_path) as f:
        for line in f:
            doc = json.loads(line)
            doc_contents[doc["id"]] = doc["content"]

    episode_results = []
    n_success = 0
    n_fallback = 0

    for obj in raw_results:
        doc_id = obj["custom_id"]
        content = doc_contents.get(doc_id, "")

        resp = obj.get("response", {})
        status_code = resp.get("status_code", 0)
        body = resp.get("body", {})

        if status_code == 200 and body.get("choices"):
            response_text = body["choices"][0]["message"]["content"]
            result = parse_llm_episodes(response_text, doc_id, content)
            if result["method"] == "llm":
                n_success += 1
            else:
                n_fallback += 1
        else:
            # API error → heuristic fallback
            error_msg = obj.get("error", {}).get("message", "unknown")
            logger.warning("API error for %s: %s. Falling back.", doc_id, error_msg)
            result = episodify_heuristic(doc_id, content)
            n_fallback += 1

        episode_results.append(result)

    logger.info("Processed %d results: llm_success=%d, fallback=%d",
                len(episode_results), n_success, n_fallback)
    return episode_results


# ------------------------------------------------------------------ #
# Synchronous mode (for small batches / testing)
# ------------------------------------------------------------------ #

def episodify_sync(
    docs_jsonl_path: Path,
    model: str = "gpt-4o-mini",
    limit: int | None = None,
) -> list[dict]:
    """Synchronous episodification (for testing / small runs)."""
    from answerer import LLMAnswerer

    llm = LLMAnswerer(model=model, temperature=0.0, max_tokens=800)
    results = []

    with open(docs_jsonl_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            doc = json.loads(line)
            doc_id = doc["id"]
            content = doc["content"]

            sents = split_sentences(content)
            if len(sents) <= 3:
                result = episodify_heuristic(doc_id, content)
            else:
                prompt = _EPISODIFY_PROMPT.format(content=content[:1500])
                try:
                    response = llm._llm_call_raw(prompt, max_tokens=800)
                    result = parse_llm_episodes(response, doc_id, content)
                except Exception as e:
                    logger.warning("LLM error for %s: %s", doc_id, e)
                    result = episodify_heuristic(doc_id, content)

            results.append(result)
            if (i + 1) % 100 == 0:
                n_llm = sum(1 for r in results if r["method"] == "llm")
                logger.info("  Processed %d docs (llm=%d, heuristic=%d)",
                            i + 1, n_llm, len(results) - n_llm)

    return results


# ------------------------------------------------------------------ #
# Resume mode: combine heuristic + batch results into final output
# ------------------------------------------------------------------ #

def resume_and_combine(
    output_dir: Path,
    domain: str,
    docs_jsonl_path: Path,
) -> None:
    """Resume from a completed batch: download results and merge with heuristic."""
    batch_id_path = output_dir / f"{domain}_batch_id.txt"
    heuristic_path = output_dir / f"{domain}_heuristic.jsonl"
    final_path = output_dir / f"{domain}_episodes.jsonl"

    if not batch_id_path.exists():
        logger.error("No batch ID found at %s", batch_id_path)
        sys.exit(1)

    batch_id = batch_id_path.read_text().strip()
    logger.info("Resuming batch %s for domain=%s", batch_id, domain)

    # Poll until done
    result = poll_batch(batch_id)
    if result["status"] != "completed":
        logger.error("Batch failed: %s", result)
        sys.exit(1)

    # Download batch results
    raw_results = download_batch_results(result["output_file_id"])
    episode_results = process_batch_results(raw_results, docs_jsonl_path)

    # Load heuristic results
    heuristic_results = []
    if heuristic_path.exists():
        with open(heuristic_path) as f:
            for line in f:
                heuristic_results.append(json.loads(line))
    logger.info("Heuristic results: %d", len(heuristic_results))

    # Combine and write
    all_results = heuristic_results + episode_results
    _write_episodes(all_results, final_path, output_dir, domain)


def _write_episodes(
    results: list[dict],
    output_path: Path,
    output_dir: Path,
    domain: str,
) -> None:
    """Write episode results and statistics."""
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d episode records to %s", len(results), output_path)

    # Statistics
    n_heuristic = sum(1 for r in results if r["method"] == "heuristic")
    n_llm = sum(1 for r in results if r["method"] == "llm")
    ep_counts = [len(r["episodes"]) for r in results]
    total_eps = sum(ep_counts)
    type_counts: dict[str, int] = {}
    connects_counts = []
    for r in results:
        for ep in r["episodes"]:
            t = ep.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            connects_counts.append(len(ep.get("connects_to", [])))

    import numpy as np
    ep_arr = np.array(ep_counts) if ep_counts else np.array([0])

    stats = {
        "domain": domain,
        "n_docs": len(results),
        "n_heuristic": n_heuristic,
        "n_llm": n_llm,
        "total_episodes": total_eps,
        "episodes_per_doc": {
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
    stats_path = output_dir / f"{domain}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Statistics saved to %s", stats_path)
    logger.info("  %d docs, %d episodes (%.1f per doc)",
                len(results), total_eps, ep_arr.mean())
    logger.info("  Types: %s", type_counts)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Episodify BRIGHT corpus documents"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with {domain}_docs.jsonl")
    parser.add_argument("--domain", required=True,
                        help="Domain to process (e.g. biology)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for episodes")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--batch-api", action="store_true",
                        help="Use OpenAI Batch API (50%% off)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: poll batch and download results")
    parser.add_argument("--sync", action="store_true",
                        help="Synchronous mode (for testing)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit docs to process (for testing)")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Batch poll interval in seconds (default: 300)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = data_dir / f"{args.domain}_docs.jsonl"
    if not docs_path.exists():
        logger.error("Docs file not found: %s", docs_path)
        sys.exit(1)

    # --resume: download completed batch results and merge
    if args.resume:
        resume_and_combine(output_dir, args.domain, docs_path)
        return

    # --sync: synchronous mode for testing
    if args.sync:
        logger.info("Synchronous mode: domain=%s, limit=%s", args.domain, args.limit)
        results = episodify_sync(docs_path, model=args.model, limit=args.limit)
        final_path = output_dir / f"{args.domain}_episodes.jsonl"
        _write_episodes(results, final_path, output_dir, args.domain)
        return

    # --batch-api: create requests and submit
    if args.batch_api:
        requests_path = output_dir / f"{args.domain}_batch_requests.jsonl"

        logger.info("Creating batch requests: domain=%s", args.domain)
        n_heuristic, n_llm, heuristic_results = create_batch_requests(
            docs_path, requests_path, model=args.model
        )

        # Save heuristic results
        heuristic_path = output_dir / f"{args.domain}_heuristic.jsonl"
        with open(heuristic_path, "w") as f:
            for r in heuristic_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Saved %d heuristic results to %s",
                     len(heuristic_results), heuristic_path)

        if n_llm == 0:
            logger.info("No LLM requests needed. Writing final output.")
            final_path = output_dir / f"{args.domain}_episodes.jsonl"
            _write_episodes(heuristic_results, final_path, output_dir, args.domain)
            return

        # Submit batch
        batch_id = submit_batch(requests_path, args.domain)
        batch_id_path = output_dir / f"{args.domain}_batch_id.txt"
        batch_id_path.write_text(batch_id)
        logger.info("Batch submitted. ID saved to %s", batch_id_path)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Wait for batch to complete (~few hours)")
        logger.info("  2. Run with --resume to download results:")
        logger.info("     PYTHONPATH=experiments/hotpotqa_v2/src "
                     ".venv/bin/python3 %s \\", sys.argv[0])
        logger.info("       --data-dir %s --domain %s \\",
                     args.data_dir, args.domain)
        logger.info("       --output-dir %s --resume", args.output_dir)
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
