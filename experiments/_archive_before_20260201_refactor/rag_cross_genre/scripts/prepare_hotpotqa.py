#!/usr/bin/env python3
"""Build a cross-genre RAG dataset from HotpotQA (distractor)."""

from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


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
        "geology",
        "climate",
        "ecology",
        "evolution",
        "species",
        "mathematics",
        "mathematician",
        "theorem",
        "equation",
        "neuron",
        "neural",
        "neuroscience",
        "enzyme",
    ],
    "technology": [
        "technology",
        "technological",
        "computer",
        "software",
        "hardware",
        "internet",
        "network",
        "data",
        "database",
        "algorithm",
        "programming",
        "code",
        "coding",
        "system",
        "engineer",
        "engineering",
        "robot",
        "ai",
        "artificial intelligence",
        "machine learning",
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
        "bank",
        "banking",
        "trade",
        "investment",
        "inflation",
        "price",
        "currency",
        "money",
        "capital",
        "company",
        "corporation",
        "business",
        "industry",
        "enterprise",
        "stock",
        "exchange",
        "profit",
        "revenue",
        "budget",
        "tax",
        "gdp",
        "manufacturing",
        "commerce",
        "export",
        "import",
        "insurance",
    ],
    "history": [
        "history",
        "historical",
        "empire",
        "dynasty",
        "king",
        "queen",
        "emperor",
        "president",
        "prime minister",
        "governor",
        "war",
        "battle",
        "revolution",
        "independence",
        "treaty",
        "colonial",
        "civil war",
        "world war",
        "century",
        "ancient",
        "medieval",
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
        "painting",
        "painter",
        "sculpture",
        "sculptor",
        "music",
        "musician",
        "composer",
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
        "theatre",
        "theater",
        "play",
        "novel",
        "poem",
        "poet",
        "literature",
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
        "psychological",
        "psychologist",
        "psychiatry",
        "psychiatrist",
        "cognitive",
        "cognition",
        "behavior",
        "behavioral",
        "mental",
        "emotion",
        "memory",
        "perception",
        "mind",
        "personality",
        "therapy",
    ],
    "philosophy": [
        "philosophy",
        "philosopher",
        "ethics",
        "ethical",
        "morality",
        "moral",
        "metaphysics",
        "epistemology",
        "ontology",
        "logic",
        "existential",
        "existentialism",
        "rationalism",
        "empiricism",
        "paradox",
        "meaning",
        "truth",
    ],
    "daily_life": [
        "health",
        "medical",
        "medicine",
        "hospital",
        "disease",
        "nutrition",
        "diet",
        "food",
        "cooking",
        "sport",
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
        "education",
        "school",
        "college",
        "university",
        "student",
        "family",
    ],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_hits(text: str, tokens: set[str], keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return keyword in tokens


def _score_domains(text: str, weight: int = 1) -> Dict[str, int]:
    text_lower = text.lower()
    tokens = set(_tokenize(text_lower))
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if _keyword_hits(text_lower, tokens, keyword):
                score += weight
        scores[domain] = score
    return scores


def _score_domains_weighted(title: str, text: str) -> Dict[str, int]:
    scores = _score_domains(title, weight=2)
    text_scores = _score_domains(text, weight=1)
    for domain, score in text_scores.items():
        scores[domain] = scores.get(domain, 0) + score
    return scores


def _pick_domain(title: str, text: str) -> Tuple[str, Dict[str, int], int]:
    scores = _score_domains_weighted(title, text)
    max_score = max(scores.values()) if scores else 0
    if max_score <= 0:
        return "other", scores, 0
    best = sorted([d for d, s in scores.items() if s == max_score])
    return best[0], scores, max_score


class DomainClassifier:
    def __init__(
        self,
        fallback_embedding: bool,
        embedding_model: str,
        embedding_threshold: float,
        fallback_min_score: int,
    ) -> None:
        self.fallback_embedding = fallback_embedding
        self.embedding_model = embedding_model
        self.embedding_threshold = embedding_threshold
        self.fallback_min_score = fallback_min_score
        self.cache: Dict[str, Tuple[str, bool, bool]] = {}
        self.model = None
        self.domain_names: List[str] = []
        self.domain_vectors = None
        if self.fallback_embedding:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "sentence-transformers is required for embedding fallback. "
                    "Run with .venv/bin/python or install dependencies."
                ) from exc
            self.model = SentenceTransformer(self.embedding_model)
            self.domain_names = list(DOMAIN_KEYWORDS.keys())
            domain_texts = [
                f"{name} {' '.join(DOMAIN_KEYWORDS[name])}" for name in self.domain_names
            ]
            self.domain_vectors = self.model.encode(domain_texts, normalize_embeddings=True)

    def classify(self, title: str, text: str, allow_fallback: bool) -> Tuple[str, bool, bool]:
        cache_key = title or text[:120]
        if cache_key in self.cache:
            return self.cache[cache_key]

        domain, _, max_score = _pick_domain(title, text)
        used_fallback = False
        reclassified = False
        if allow_fallback and self.fallback_embedding and (domain == "other" or max_score < self.fallback_min_score):
            used_fallback = True
            if self.model is None or self.domain_vectors is None:
                raise RuntimeError("Embedding fallback requested without model initialization")
            doc_vec = self.model.encode([text], normalize_embeddings=True)[0]
            import numpy as np

            sims = np.dot(self.domain_vectors, doc_vec)
            best_idx = int(sims.argmax()) if len(sims) else 0
            best_score = float(sims[best_idx]) if len(sims) else 0.0
            if best_score >= self.embedding_threshold:
                domain = self.domain_names[best_idx]
                reclassified = True

        result = (domain, used_fallback, reclassified)
        self.cache[cache_key] = result
        return result


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _doc_id(prefix: str, title: str, idx: int) -> str:
    checksum = zlib.adler32(title.encode("utf-8"))
    return f"{prefix}_{idx}_{_slug(title)}_{checksum}"


def _iter_hotpotqa(split: str, input_path: Path | None) -> Iterator[Dict[str, object]]:
    if input_path:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        for item in payload:
            yield item
        return

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required for HotpotQA download. "
            "Install with .venv/bin/python -m pip install datasets"
        ) from exc

    ds = load_dataset("hotpot_qa", "distractor", split=split, streaming=True)
    for item in ds:
        yield item


def _build_hotpotqa_dataset(
    output: Path,
    split: str,
    max_questions: int | None,
    min_domains: int,
    min_non_other: int,
    include_other: bool,
    fallback_embedding: bool,
    embedding_model: str,
    embedding_threshold: float,
    fallback_min_score: int,
    distractors: int,
    seed: int,
    input_path: Path | None,
) -> Dict[str, int]:
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "total": 0,
        "kept": 0,
        "skipped_domains": 0,
        "fallback_attempted": 0,
        "fallback_reclassified": 0,
    }
    domain_counts: Dict[str, int] = {}
    classifier = DomainClassifier(
        fallback_embedding=fallback_embedding,
        embedding_model=embedding_model,
        embedding_threshold=embedding_threshold,
        fallback_min_score=fallback_min_score,
    )

    with output.open("w", encoding="utf-8") as fh:
        for idx, item in enumerate(_iter_hotpotqa(split=split, input_path=input_path)):
            stats["total"] += 1
            question = str(item.get("question", ""))
            supporting_facts = item.get("supporting_facts", {}) or {}
            if isinstance(supporting_facts, dict):
                support_titles = {str(title) for title in supporting_facts.get("title", [])}
            else:
                support_titles = {str(title) for title, _ in supporting_facts}
            context = item.get("context", {}) or {}

            title_to_sents: Dict[str, List[str]] = {}
            if isinstance(context, dict):
                titles = context.get("title", []) or []
                sentences_list = context.get("sentences", []) or []
                for title, sentences in zip(titles, sentences_list):
                    title_to_sents[str(title)] = [str(s) for s in sentences]
            else:
                for title, sentences in context:
                    title_to_sents[str(title)] = [str(s) for s in sentences]

            support_docs = []
            support_domains = []
            for title in sorted(support_titles):
                sentences = title_to_sents.get(title, [])
                text = " ".join(sentences)
                domain, used_fallback, reclassified = classifier.classify(
                    title=title,
                    text=f"{title} {text}",
                    allow_fallback=True,
                )
                if used_fallback:
                    stats["fallback_attempted"] += 1
                if reclassified:
                    stats["fallback_reclassified"] += 1
                support_domains.append(domain)
                support_docs.append(
                    {
                        "id": _doc_id("hotpot", title, idx),
                        "text": text,
                        "metadata": {
                            "title": title,
                            "role": "support",
                            "domain": domain,
                            "support_domain": domain,
                        },
                    }
                )

            unique_domains = set(support_domains)
            non_other_domains = {d for d in unique_domains if d != "other"}
            if len(non_other_domains) < min_non_other:
                stats["skipped_domains"] += 1
                continue
            if not include_other:
                unique_domains = non_other_domains
            unique_domains = sorted(unique_domains)
            if len(unique_domains) < min_domains:
                stats["skipped_domains"] += 1
                continue

            for domain in unique_domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            distractor_docs = []
            available = [t for t in title_to_sents.keys() if t not in support_titles]
            if distractors > 0 and available:
                chosen = rng.sample(available, k=min(distractors, len(available)))
            else:
                chosen = []
            for title in chosen:
                sentences = title_to_sents.get(title, [])
                text = " ".join(sentences)
                domain, _, _ = classifier.classify(
                    title=title,
                    text=f"{title} {text}",
                    allow_fallback=False,
                )
                distractor_docs.append(
                    {
                        "id": _doc_id("hotpot", title, idx),
                        "text": text,
                        "metadata": {
                            "title": title,
                            "role": "distractor",
                            "domain": domain,
                        },
                    }
                )

            ground_truth = " ".join(doc["text"] for doc in support_docs)
            row = {
                "query": question,
                "ground_truth": ground_truth,
                "documents": support_docs + distractor_docs,
                "expected_tags": unique_domains,
                "mapped_support_domains": unique_domains,
                "selection_mode": "hotpotqa",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            if max_questions is not None and stats["kept"] >= max_questions:
                break

    stats.update({f"domain_{k}": v for k, v in sorted(domain_counts.items())})
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare HotpotQA cross-genre dataset")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--min-domains", type=int, default=2)
    ap.add_argument("--min-non-other", type=int, default=1)
    ap.add_argument("--include-other", action="store_true")
    ap.add_argument("--fallback-embedding", action="store_true")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embedding-threshold", type=float, default=0.25)
    ap.add_argument("--fallback-min-score", type=int, default=2)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input", type=Path, default=None, help="Optional local HotpotQA JSON")
    args = ap.parse_args()

    stats = _build_hotpotqa_dataset(
        output=args.output,
        split=args.split,
        max_questions=args.max_questions,
        min_domains=args.min_domains,
        min_non_other=args.min_non_other,
        include_other=args.include_other,
        fallback_embedding=args.fallback_embedding,
        embedding_model=args.embedding_model,
        embedding_threshold=args.embedding_threshold,
        fallback_min_score=args.fallback_min_score,
        distractors=args.distractors,
        seed=args.seed,
        input_path=args.input,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
