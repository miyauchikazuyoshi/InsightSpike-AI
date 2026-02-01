#!/usr/bin/env python3
"""Generate cross-genre RAG datasets (JSONL).

Two modes:
1) Synthetic mode (default): generate cross-genre queries with paired support docs.
2) Import mode: use an external knowledge base + question set with expected_tags.
   Tags are mapped onto the KB tag space to ensure cross-genre support docs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CategoryTopic = Tuple[str, str]

TAG_ALIASES: Dict[str, str] = {
    "architecture": "art",
    "astronomy": "physics",
    "behavior": "psychology",
    "chaos": "science",
    "civilization": "history",
    "communication": "culture",
    "complexity": "science",
    "computer science": "technology",
    "conservation": "physics",
    "creativity": "art",
    "ecology": "biology",
    "education": "daily_life",
    "emergence": "science",
    "geometry": "mathematics",
    "identity": "philosophy",
    "information": "science",
    "innovation": "business",
    "institutions": "history",
    "learning": "psychology",
    "linguistics": "literature",
    "medicine": "health",
    "methodology": "science",
    "nature": "science",
    "paradox": "philosophy",
    "patterns": "mathematics",
    "performance": "music",
    "progress": "history",
    "society": "culture",
    "sustainability": "economics",
    "systems": "science",
}

DOMAIN_MAP: Dict[str, str] = {
    "ai": "technology",
    "art": "arts",
    "architecture": "arts",
    "astronomy": "science",
    "behavior": "psychology",
    "biology": "science",
    "business": "economics",
    "chaos": "science",
    "chemistry": "science",
    "civilization": "history",
    "cognition": "psychology",
    "communication": "psychology",
    "complexity": "science",
    "computer science": "technology",
    "conservation": "science",
    "creativity": "arts",
    "culture": "history",
    "daily_life": "daily_life",
    "ecology": "science",
    "economics": "economics",
    "education": "daily_life",
    "emergence": "science",
    "ethics": "philosophy",
    "evolution": "science",
    "finance": "economics",
    "geometry": "mathematics",
    "health": "daily_life",
    "history": "history",
    "identity": "philosophy",
    "information": "science",
    "innovation": "economics",
    "institutions": "history",
    "learning": "psychology",
    "linguistics": "arts",
    "literature": "arts",
    "mathematics": "mathematics",
    "medicine": "daily_life",
    "methodology": "science",
    "music": "arts",
    "nature": "science",
    "paradox": "philosophy",
    "patterns": "mathematics",
    "performance": "arts",
    "philosophy": "philosophy",
    "physics": "science",
    "progress": "history",
    "psychology": "psychology",
    "science": "science",
    "society": "history",
    "sports": "daily_life",
    "strategy": "economics",
    "sustainability": "economics",
    "systems": "science",
    "technology": "technology",
    "theorems": "mathematics",
}

CATEGORIES: Dict[str, List[CategoryTopic]] = {
    "science": [
        ("photosynthesis", "converts light into chemical energy in plants"),
        ("gravity", "pulls masses together and shapes orbits"),
        ("ecosystems", "cycle energy through producers and consumers"),
        ("water_cycle", "moves water via evaporation and precipitation"),
    ],
    "technology": [
        ("machine_learning", "optimizes parameters to reduce prediction error"),
        ("encryption", "protects data using mathematical keys"),
        ("networks", "route packets using routing tables"),
        ("sensors", "convert physical signals into digital measurements"),
    ],
    "economics": [
        ("inflation", "reduces purchasing power when prices rise"),
        ("supply_demand", "balances quantities through price signals"),
        ("interest_rates", "influence borrowing and investment costs"),
        ("trade", "moves goods across markets to exploit advantage"),
    ],
    "history": [
        ("renaissance", "revived art and science across Europe"),
        ("industrialization", "shifted labor toward mechanized production"),
        ("printing_press", "accelerated information dissemination"),
        ("urbanization", "concentrated populations into cities"),
    ],
    "arts": [
        ("composition", "arranges elements to guide attention"),
        ("rhythm", "organizes time in repeated patterns"),
        ("symbolism", "uses imagery to convey meaning"),
        ("perspective", "creates depth on a flat surface"),
    ],
    "psychology": [
        ("cognitive_dissonance", "creates discomfort from conflicting beliefs"),
        ("memory", "stores and retrieves past experiences"),
        ("motivation", "drives goal-directed behavior"),
        ("attention", "selects information for processing"),
    ],
    "philosophy": [
        ("identity", "questions what makes an object the same over time"),
        ("ethics", "evaluates right and wrong actions"),
        ("knowledge", "studies justified belief and truth"),
        ("causality", "links events through cause and effect"),
    ],
    "daily_life": [
        ("nutrition", "supports health through balanced intake"),
        ("cooking", "transforms ingredients through heat and time"),
        ("exercise", "improves fitness via repeated training"),
        ("sleep", "restores energy and consolidates memory"),
    ],
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _normalize_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.strip().lower())


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2]


def _make_doc(doc_id: str, category: str, topic: str, fact: str, role: str) -> Dict[str, object]:
    text = f"In {category}, {topic} {fact}."
    return {
        "id": doc_id,
        "text": text,
        "metadata": {
            "category": category,
            "topic": topic,
            "role": role,
        },
    }


def _flatten_question_set(payload: Dict[str, object]) -> List[Dict[str, object]]:
    if "question_categories" in payload:
        questions: List[Dict[str, object]] = []
        for block in payload.get("question_categories", {}).values():
            questions.extend(block.get("questions", []))
        return questions
    if "questions" in payload:
        return list(payload.get("questions", []))
    raise ValueError("Unsupported question set format (missing question_categories or questions)")


def _load_kb(path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("knowledge_entries", [])
    kb = {}
    for entry in entries:
        entry_id = entry.get("id")
        if not entry_id:
            continue
        kb[str(entry_id)] = entry
    if not kb:
        raise ValueError("Knowledge base has no entries")
    return kb


def _load_questions(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _flatten_question_set(payload)


def _build_tag_index(kb: Dict[str, Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    tag_index: Dict[str, List[Dict[str, object]]] = {}
    for entry in kb.values():
        for tag in entry.get("tags", []):
            norm = _normalize_tag(str(tag))
            tag_index.setdefault(norm, []).append(entry)
    return tag_index


def _resolve_tags(raw_tags: Iterable[str], known_tags: set[str]) -> List[str]:
    resolved: List[str] = []
    for tag in raw_tags:
        norm = _normalize_tag(tag)
        if not norm:
            continue
        if norm in known_tags:
            resolved.append(norm)
            continue
        alias = TAG_ALIASES.get(norm)
        if alias and alias in known_tags:
            resolved.append(alias)
    return list(dict.fromkeys(resolved))


def _tag_domain(tag: str) -> str:
    return DOMAIN_MAP.get(tag, tag)


def _score_entry(question_tokens: set[str], entry: Dict[str, object]) -> float:
    if not question_tokens:
        return 0.0
    content = str(entry.get("content", ""))
    related = " ".join(str(x) for x in entry.get("related_concepts", []) or [])
    tags = " ".join(str(x) for x in entry.get("tags", []) or [])
    tokens = set(_tokenize(f"{content} {related} {tags}"))
    if not tokens:
        return 0.0
    overlap = question_tokens & tokens
    return len(overlap) / len(question_tokens)


def _entry_text(entry: Dict[str, object]) -> str:
    content = str(entry.get("content", ""))
    related = " ".join(str(x) for x in entry.get("related_concepts", []) or [])
    tags = " ".join(str(x) for x in entry.get("tags", []) or [])
    return " ".join(part for part in (content, related, tags) if part)


def _prepare_embedding_selector(
    kb: Dict[str, Dict[str, object]],
    model_name: str,
) -> tuple[Dict[str, object], object]:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Embedding selection requires sentence-transformers + numpy. "
            "Run with .venv/bin/python or install dependencies."
        ) from exc

    entries = list(kb.items())
    texts = [_entry_text(entry) for _, entry in entries]
    embedder = SentenceTransformer(model_name)
    vectors = embedder.encode(texts, normalize_embeddings=True)
    emb_map = {entry_id: vectors[i] for i, (entry_id, _) in enumerate(entries)}
    return emb_map, embedder


def _build_from_import(
    kb_path: Path,
    question_path: Path,
    output: Path,
    seed: int,
    max_questions: int | None,
    distractors: int,
    min_support_domains: int,
    support_per_query: int,
    selection_mode: str,
    embedding_model: str,
) -> None:
    rng = random.Random(seed)
    kb = _load_kb(kb_path)
    questions = _load_questions(question_path)
    tag_index = _build_tag_index(kb)
    known_tags = set(tag_index.keys())
    emb_map = None
    embedder = None
    if selection_mode == "embedding":
        emb_map, embedder = _prepare_embedding_selector(kb, embedding_model)

    if support_per_query < min_support_domains:
        raise ValueError("support_per_query must be >= min_support_domains")

    if max_questions is not None:
        questions = questions[: max_questions]

    kb_ids = list(kb.keys())

    with output.open("w", encoding="utf-8") as fh:
        for q in questions:
            question_text = str(q.get("question", ""))
            question_tokens = set(_tokenize(question_text))
            expected_tags = q.get("expected_tags") or q.get("tags") or []
            resolved_tags = _resolve_tags(expected_tags, known_tags)
            domain_to_tags: Dict[str, List[str]] = {}
            for tag in resolved_tags:
                domain = _tag_domain(tag)
                domain_to_tags.setdefault(domain, []).append(tag)
            if len(domain_to_tags) < min_support_domains:
                continue

            chosen_domains = rng.sample(
                list(domain_to_tags.keys()),
                k=min(len(domain_to_tags), min_support_domains),
            )
            chosen_tags = [rng.choice(domain_to_tags[d]) for d in chosen_domains]
            if len(chosen_tags) < support_per_query:
                remaining = [t for t in resolved_tags if t not in chosen_tags]
                if remaining:
                    extra = rng.sample(
                        remaining, k=min(len(remaining), support_per_query - len(chosen_tags))
                    )
                    chosen_tags.extend(extra)
            support_docs = []
            support_ids: set[str] = set()
            for tag in chosen_tags:
                candidates = [e for e in tag_index.get(tag, []) if e.get("id") not in support_ids]
                if not candidates:
                    continue
                if selection_mode == "embedding":
                    import numpy as np

                    assert embedder is not None and emb_map is not None
                    q_vec = embedder.encode(question_text, normalize_embeddings=True)
                    cand_ids = [str(e.get("id", "")) for e in candidates]
                    cand_vecs = np.stack([emb_map[cid] for cid in cand_ids], axis=0)
                    scores = cand_vecs @ q_vec
                    best_idx = int(scores.argmax()) if len(scores) else 0
                    entry = candidates[best_idx]
                    support_score = float(scores[best_idx]) if len(scores) else 0.0
                else:
                    scored = [(float(_score_entry(question_tokens, e)), e) for e in candidates]
                    max_score = max(score for score, _ in scored)
                    best = [e for score, e in scored if score == max_score]
                    entry = rng.choice(best) if best else rng.choice(candidates)
                    support_score = _score_entry(question_tokens, entry)
                entry_id = str(entry.get("id", ""))
                support_ids.add(entry_id)
                doc = {
                    "id": entry_id,
                    "text": str(entry.get("content", "")),
                    "metadata": {
                        "tags": ",".join(entry.get("tags", [])),
                        "difficulty": str(entry.get("difficulty", "")),
                        "role": "support",
                        "support_tag": tag,
                        "support_domain": _tag_domain(tag),
                        "support_score": f"{support_score:.3f}",
                    },
                }
                support_docs.append(doc)

            if len(support_docs) < min_support_domains:
                continue

            distractor_docs = []
            available = [kid for kid in kb_ids if kid not in support_ids]
            if distractors > 0 and available:
                pick = min(len(available), distractors)
                chosen = rng.sample(available, k=pick)
            else:
                chosen = []
            for kid in chosen:
                entry = kb[kid]
                distractor_docs.append(
                    {
                        "id": kid,
                        "text": str(entry.get("content", "")),
                        "metadata": {
                            "tags": ",".join(entry.get("tags", [])),
                            "difficulty": str(entry.get("difficulty", "")),
                            "role": "distractor",
                        },
                    }
                )

            ground_truth = " ".join(doc["text"] for doc in support_docs)
            row = {
                "query": question_text,
                "ground_truth": ground_truth,
                "documents": support_docs + distractor_docs,
                "expected_tags": expected_tags,
                "mapped_support_tags": chosen_tags,
                "mapped_support_domains": sorted({_tag_domain(tag) for tag in chosen_tags}),
                "selection_mode": selection_mode,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_synthetic(
    output: Path,
    seed: int,
    num_queries: int,
    docs_per_query: int,
) -> None:
    rng = random.Random(seed)
    category_names = list(CATEGORIES.keys())
    if docs_per_query < 2:
        raise ValueError("docs_per_query must be >= 2")
    distractors = max(0, docs_per_query - 2)

    with output.open("w", encoding="utf-8") as fh:
        for idx in range(num_queries):
            cat_a, cat_b = rng.sample(category_names, 2)
            topic_a, fact_a = rng.choice(CATEGORIES[cat_a])
            topic_b, fact_b = rng.choice(CATEGORIES[cat_b])

            doc_a = _make_doc(
                f"{_slug(cat_a)}_{_slug(topic_a)}_{idx}_a",
                cat_a,
                topic_a,
                fact_a,
                "support",
            )
            doc_b = _make_doc(
                f"{_slug(cat_b)}_{_slug(topic_b)}_{idx}_b",
                cat_b,
                topic_b,
                fact_b,
                "support",
            )

            query = (
                f"How does {topic_a} in {cat_a} connect with "
                f"{topic_b} in {cat_b}?"
            )
            ground_truth = f"{doc_a['text']} {doc_b['text']}"

            docs = [doc_a, doc_b]
            used_pairs = {(cat_a, topic_a), (cat_b, topic_b)}
            for j in range(distractors):
                for _ in range(10):
                    cat_d = rng.choice([c for c in category_names if c not in {cat_a, cat_b}])
                    topic_d, fact_d = rng.choice(CATEGORIES[cat_d])
                    if (cat_d, topic_d) not in used_pairs:
                        used_pairs.add((cat_d, topic_d))
                        break
                docs.append(
                    _make_doc(
                        f"{_slug(cat_d)}_{_slug(topic_d)}_{idx}_d{j}",
                        cat_d,
                        topic_d,
                        fact_d,
                        "distractor",
                    )
                )

            row = {
                "query": query,
                "ground_truth": ground_truth,
                "documents": docs,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate cross-genre RAG datasets (JSONL)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-queries", type=int, default=200)
    ap.add_argument("--docs-per-query", type=int, default=6)
    ap.add_argument("--kb", type=Path, default=None, help="Optional knowledge base JSON")
    ap.add_argument("--question-set", type=Path, default=None, help="Optional question set JSON")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--min-support-domains", type=int, default=2)
    ap.add_argument("--support-per-query", type=int, default=2)
    ap.add_argument("--selection-mode", choices=["token", "embedding"], default="token")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.kb and args.question_set:
        _build_from_import(
            kb_path=args.kb,
            question_path=args.question_set,
            output=args.output,
            seed=args.seed,
            max_questions=args.max_questions,
            distractors=args.distractors,
            min_support_domains=args.min_support_domains,
            support_per_query=args.support_per_query,
            selection_mode=args.selection_mode,
            embedding_model=args.embedding_model,
        )
    else:
        _build_synthetic(
            output=args.output,
            seed=args.seed,
            num_queries=args.num_queries,
            docs_per_query=args.docs_per_query,
        )

    print(f"Wrote dataset to {args.output}")


if __name__ == "__main__":
    main()
