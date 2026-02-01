#!/usr/bin/env python3
"""Generate a qualitative scoring table for HotpotQA cross-genre samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "science": [
        "science",
        "scientific",
        "research",
        "laboratory",
        "experiment",
        "physics",
        "quantum",
        "chemistry",
        "chemical",
        "biology",
        "genetics",
        "genetic",
        "genome",
        "cell",
        "protein",
        "molecule",
        "astronomy",
        "astronomer",
        "planet",
        "galaxy",
        "space",
        "energy",
        "theorem",
        "equation",
        "neuron",
        "geology",
        "climate",
    ],
    "technology": [
        "technology",
        "technological",
        "computer",
        "software",
        "hardware",
        "algorithm",
        "network",
        "internet",
        "encryption",
        "machine",
        "robot",
        "ai",
        "database",
        "programming",
        "code",
        "coding",
        "system",
        "engineer",
        "engineering",
        "device",
        "digital",
        "electronics",
        "application",
        "app",
        "operating system",
        "telecommunications",
    ],
    "economics": [
        "economy",
        "economic",
        "market",
        "finance",
        "financial",
        "trade",
        "investment",
        "inflation",
        "bank",
        "business",
        "capital",
        "company",
        "corporation",
        "industry",
        "enterprise",
        "stock",
        "exchange",
        "profit",
        "revenue",
        "budget",
        "tax",
        "gdp",
        "commerce",
        "export",
        "import",
        "insurance",
    ],
    "history": [
        "history",
        "historical",
        "empire",
        "war",
        "revolution",
        "dynasty",
        "ancient",
        "medieval",
        "century",
        "colonial",
        "civilization",
        "king",
        "queen",
        "emperor",
        "president",
        "prime minister",
        "governor",
        "battle",
        "independence",
        "treaty",
        "civil war",
        "world war",
        "roman",
        "ottoman",
        "military",
        "army",
        "navy",
        "occupation",
        "uprising",
        "regime",
        "monarchy",
    ],
    "arts": [
        "art",
        "artist",
        "music",
        "painting",
        "novel",
        "poem",
        "literature",
        "composer",
        "dance",
        "theatre",
        "theater",
        "sculpture",
        "sculptor",
        "musician",
        "song",
        "album",
        "band",
        "orchestra",
        "opera",
        "film",
        "movie",
        "cinema",
        "television",
        "tv",
        "series",
        "episode",
        "actor",
        "actress",
        "director",
        "play",
        "book",
        "writer",
        "author",
        "drama",
        "comedy",
        "animation",
        "museum",
        "gallery",
    ],
    "psychology": [
        "psychology",
        "cognition",
        "behavior",
        "memory",
        "emotion",
        "perception",
        "mind",
        "mental",
        "psychological",
        "psychologist",
        "psychiatry",
        "psychiatrist",
        "cognitive",
        "behavioral",
        "personality",
        "therapy",
    ],
    "philosophy": [
        "philosophy",
        "ethics",
        "metaphysics",
        "epistemology",
        "ontology",
        "logic",
        "paradox",
        "philosopher",
        "ethical",
        "morality",
        "moral",
        "existential",
        "existentialism",
        "rationalism",
        "empiricism",
        "meaning",
        "truth",
    ],
    "daily_life": [
        "health",
        "food",
        "nutrition",
        "sport",
        "exercise",
        "education",
        "medical",
        "disease",
        "family",
        "medicine",
        "hospital",
        "diet",
        "cooking",
        "sports",
        "athlete",
        "football",
        "footballer",
        "soccer",
        "basketball",
        "baseball",
        "cricket",
        "rugby",
        "tennis",
        "golf",
        "olympic",
        "league",
        "team",
        "club",
        "coach",
        "school",
        "college",
        "university",
        "student",
    ],
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)


def _domain_hits(query: str) -> int:
    text = query.lower()
    tokens = _tokenize(text)
    hits = 0
    for keywords in DOMAIN_KEYWORDS.values():
        if any((kw in text) if " " in kw else (kw in tokens) for kw in keywords):
            hits += 1
    return hits


def _score_relevance(avg_overlap: float) -> int:
    if avg_overlap >= 0.08:
        return 2
    if avg_overlap >= 0.04:
        return 1
    return 0


def _score_cross_domain(domains: List[str]) -> int:
    if len(domains) < 2:
        return 0
    if all(d != "other" for d in domains):
        return 2
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualitative audit scorer (HotpotQA)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(args.seed)
    sample = rng.sample(rows, k=min(args.sample_size, len(rows)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "sample_id",
            "query",
            "mapped_support_domains",
            "support_title_1",
            "support_domain_1",
            "support_snippet_1",
            "support_title_2",
            "support_domain_2",
            "support_snippet_2",
            "overlap_avg",
            "domain_hit_count",
            "score_relevance",
            "score_cross_domain",
            "score_query_domain_signal",
            "score_total",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(sample, start=1):
            query = str(row.get("query", ""))
            domains = row.get("mapped_support_domains", [])
            support_docs = [d for d in row.get("documents", []) if d.get("metadata", {}).get("role") == "support"]
            support_docs = support_docs[:2]

            query_tokens = _tokenize(query)
            overlaps = []
            payload: Dict[str, object] = {
                "sample_id": idx,
                "query": query,
                "mapped_support_domains": ",".join(domains),
            }

            for i in range(2):
                if i < len(support_docs):
                    doc = support_docs[i]
                    md = doc.get("metadata", {})
                    text = str(doc.get("text", ""))
                    snippet = text.replace("\n", " ")[:160]
                    overlaps.append(_jaccard(query_tokens, _tokenize(text)))
                    payload[f"support_title_{i+1}"] = md.get("title", "")
                    payload[f"support_domain_{i+1}"] = md.get("domain", "")
                    payload[f"support_snippet_{i+1}"] = snippet
                else:
                    payload[f"support_title_{i+1}"] = ""
                    payload[f"support_domain_{i+1}"] = ""
                    payload[f"support_snippet_{i+1}"] = ""

            overlap_avg = sum(overlaps) / len(overlaps) if overlaps else 0.0
            domain_hit_count = _domain_hits(query)
            score_relevance = _score_relevance(overlap_avg)
            score_cross_domain = _score_cross_domain(domains)
            score_query_domain_signal = 1 if domain_hit_count >= 2 else 0
            score_total = score_relevance + score_cross_domain + score_query_domain_signal

            payload.update(
                {
                    "overlap_avg": f"{overlap_avg:.3f}",
                    "domain_hit_count": domain_hit_count,
                    "score_relevance": score_relevance,
                    "score_cross_domain": score_cross_domain,
                    "score_query_domain_signal": score_query_domain_signal,
                    "score_total": score_total,
                }
            )
            writer.writerow(payload)


if __name__ == "__main__":
    main()
