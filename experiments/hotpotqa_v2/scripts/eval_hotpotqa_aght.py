#!/usr/bin/env python3
"""Evaluate AGHT vs Legacy graph scoring on HotpotQA paragraph selection.

HotpotQA distractor setting: 10 paragraphs, find the 2 gold ones.
No BM25 retrieval — pure graph-based ranking comparison.

Metrics:
  - Recall@2: fraction of gold paragraphs in top-2
  - Recall@4: fraction of gold paragraphs in top-4
  - MRR: mean reciprocal rank of first gold paragraph
  - P@2: precision at 2
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_hotpotqa(path: str, limit: int = 100):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data.append(json.loads(line))
    return data


def evaluate_aght(data: list[dict], nlp) -> list[dict]:
    """Run AGHT on each HotpotQA question."""
    from unified_graph import run_aght, AGHTConfig

    config = AGHTConfig(mp_iterations=3, mp_alpha=0.4)
    results = []

    for i, item in enumerate(data):
        query = item["question"]
        ctx = item["context"]
        titles_raw = ctx["title"]
        sents_raw = ctx["sentences"]
        gold_titles = set(item["supporting_facts"]["title"])
        qtype = item["type"]

        # Build input for AGHT
        titles = [f"doc_{j}" for j in range(len(titles_raw))]
        sentences_list = []
        doc_id_map = {}
        for j, (title, sents) in enumerate(zip(titles_raw, sents_raw)):
            # Join sentence fragments and split properly
            full_text = " ".join(sents)
            # Use original sentence list (already split)
            clean_sents = [s.strip() for s in sents if len(s.strip()) > 5]
            if not clean_sents:
                clean_sents = [full_text[:500]]
            sentences_list.append(clean_sents[:30])
            doc_id_map[f"doc_{j}"] = title  # Use original title as doc_id

        t0 = time.time()
        scores, aght_result, aght_graph = run_aght(
            titles, sentences_list, query, nlp, doc_id_map,
            config=config,
        )
        latency = (time.time() - t0) * 1000

        # Rank paragraphs by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        ranked_titles = [title for title, _ in ranked]

        # Metrics (paragraph level)
        top2_titles = set(ranked_titles[:2])
        top4_titles = set(ranked_titles[:4])
        recall_2 = len(gold_titles & top2_titles) / len(gold_titles)
        recall_4 = len(gold_titles & top4_titles) / len(gold_titles)
        p_at_2 = len(gold_titles & top2_titles) / 2

        # MRR: reciprocal rank of first gold
        mrr = 0.0
        for rank, title in enumerate(ranked_titles, 1):
            if title in gold_titles:
                mrr = 1.0 / rank
                break

        # ── Sentence-level supporting fact evaluation ──
        # Extract per-sentence relevance from AGHT graph
        gold_sf = item["supporting_facts"]
        gold_sf_set = set()
        for sf_title, sf_sid in zip(gold_sf["title"], gold_sf["sent_id"]):
            gold_sf_set.add((sf_title, sf_sid))

        # Collect (title, sent_idx, relevance) for all S nodes
        sent_scores = []
        for nid, ndata in aght_graph.nodes(data=True):
            if ndata.get("node_type") != "S":
                continue
            p_idx = ndata.get("para_idx", -1)
            s_idx = ndata.get("sent_idx", -1)
            rel = ndata.get("relevance", 0.0)
            if p_idx >= 0 and p_idx < len(titles_raw):
                orig_title = titles_raw[p_idx]
                sent_scores.append((orig_title, s_idx, rel))

        # Rank sentences by relevance
        sent_scores.sort(key=lambda x: -x[2])

        # Compute supporting fact EM and F1
        n_gold = len(gold_sf_set)
        # Take top-K sentences (K = n_gold for fair comparison, also top-5)
        pred_sf_at_k = set()
        pred_sf_at_5 = set()
        for rank_idx, (st, si, _) in enumerate(sent_scores):
            if rank_idx < n_gold:
                pred_sf_at_k.add((st, si))
            if rank_idx < 5:
                pred_sf_at_5.add((st, si))

        # F1 at K=n_gold
        tp_k = len(pred_sf_at_k & gold_sf_set)
        sf_p_k = tp_k / max(len(pred_sf_at_k), 1)
        sf_r_k = tp_k / max(n_gold, 1)
        sf_f1_k = 2 * sf_p_k * sf_r_k / max(sf_p_k + sf_r_k, 1e-10)
        sf_em_k = 1.0 if pred_sf_at_k == gold_sf_set else 0.0

        # F1 at 5
        tp_5 = len(pred_sf_at_5 & gold_sf_set)
        sf_p_5 = tp_5 / max(len(pred_sf_at_5), 1)
        sf_r_5 = tp_5 / max(n_gold, 1)
        sf_f1_5 = 2 * sf_p_5 * sf_r_5 / max(sf_p_5 + sf_r_5, 1e-10)

        results.append({
            "qid": item["id"],
            "type": qtype,
            "recall_2": recall_2,
            "recall_4": recall_4,
            "p_at_2": p_at_2,
            "mrr": mrr,
            "sf_em": sf_em_k,
            "sf_f1": sf_f1_k,
            "sf_f1_at5": sf_f1_5,
            "latency_ms": latency,
            "n_ag": aght_result.n_ag,
            "n_dg": aght_result.n_dg,
            "n_nodes": aght_result.n_s_nodes + aght_result.n_t_nodes,
        })

        if (i + 1) % 10 == 0:
            avg_r2 = np.mean([r["recall_2"] for r in results])
            print(f"  [{i+1}/{len(data)}] R@2={avg_r2:.3f} ({qtype})")

    return results


def evaluate_legacy(data: list[dict], nlp) -> list[dict]:
    """Run legacy entity graph + PageRank scoring on each HotpotQA question."""
    import networkx as nx
    from entity_graph import build_sentence_graph, extract_entities

    results = []

    for i, item in enumerate(data):
        query = item["question"]
        ctx = item["context"]
        titles_raw = ctx["title"]
        sents_raw = ctx["sentences"]
        gold_titles = set(item["supporting_facts"]["title"])
        qtype = item["type"]

        titles = [f"doc_{j}" for j in range(len(titles_raw))]
        sentences_list = []
        doc_id_map = {}
        for j, (title, sents) in enumerate(zip(titles_raw, sents_raw)):
            clean_sents = [s.strip() for s in sents if len(s.strip()) > 5]
            if not clean_sents:
                clean_sents = [" ".join(sents)[:500]]
            sentences_list.append(clean_sents[:30])
            doc_id_map[f"doc_{j}"] = title

        t0 = time.time()

        # Build legacy sentence graph
        graph = build_sentence_graph(
            titles, sentences_list,
            max_para_freq=5,
            nlp=nlp,
        )

        # Classic scoring: PageRank + query entity overlap
        if graph.number_of_nodes() == 0:
            graph_scores = {doc_id_map[t]: 0.0 for t in titles}
        else:
            try:
                pagerank = nx.pagerank(graph, weight="strength")
            except Exception:
                pagerank = {n: 1.0 / graph.number_of_nodes()
                            for n in graph.nodes()}

            # Query entities for overlap scoring
            query_lower = query.lower()
            query_tokens = set(query_lower.split())

            graph_scores = {}
            for title_idx, title in enumerate(titles):
                doc_id = doc_id_map[title]
                doc_nodes = [
                    n for n in graph.nodes()
                    if graph.nodes[n].get("title") == title
                ]
                if not doc_nodes:
                    graph_scores[doc_id] = 0.0
                    continue

                # PageRank component
                pr_sum = sum(pagerank.get(n, 0) for n in doc_nodes)

                # Query overlap component
                overlap = 0.0
                for n in doc_nodes:
                    text = graph.nodes[n].get("text", "").lower()
                    match = sum(1 for t in query_tokens if t in text)
                    overlap += match / max(len(query_tokens), 1)
                overlap /= len(doc_nodes)

                graph_scores[doc_id] = pr_sum + 0.5 * overlap

            # Normalize
            vals = list(graph_scores.values())
            mn, mx = min(vals), max(vals)
            rng = mx - mn
            if rng > 1e-10:
                graph_scores = {k: (v - mn) / rng for k, v in graph_scores.items()}

        latency = (time.time() - t0) * 1000

        # Rank
        ranked = sorted(graph_scores.items(), key=lambda x: -x[1])
        ranked_titles = [title for title, _ in ranked]

        top2_titles = set(ranked_titles[:2])
        top4_titles = set(ranked_titles[:4])
        recall_2 = len(gold_titles & top2_titles) / len(gold_titles)
        recall_4 = len(gold_titles & top4_titles) / len(gold_titles)
        p_at_2 = len(gold_titles & top2_titles) / 2

        mrr = 0.0
        for rank, title in enumerate(ranked_titles, 1):
            if title in gold_titles:
                mrr = 1.0 / rank
                break

        results.append({
            "qid": item["id"],
            "type": qtype,
            "recall_2": recall_2,
            "recall_4": recall_4,
            "p_at_2": p_at_2,
            "mrr": mrr,
            "latency_ms": latency,
        })

        if (i + 1) % 10 == 0:
            avg_r2 = np.mean([r["recall_2"] for r in results])
            print(f"  [{i+1}/{len(data)}] R@2={avg_r2:.3f} ({qtype})")

    return results


def print_summary(name: str, results: list[dict]):
    """Print summary by type."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for qtype in ["all", "bridge", "comparison"]:
        if qtype == "all":
            subset = results
        else:
            subset = [r for r in results if r["type"] == qtype]
        if not subset:
            continue

        r2 = np.mean([r["recall_2"] for r in subset])
        r4 = np.mean([r["recall_4"] for r in subset])
        p2 = np.mean([r["p_at_2"] for r in subset])
        mrr = np.mean([r["mrr"] for r in subset])
        lat = np.mean([r["latency_ms"] for r in subset])

        # Sentence-level metrics (if available)
        has_sf = "sf_f1" in subset[0]
        if has_sf:
            sf_em = np.mean([r["sf_em"] for r in subset])
            sf_f1 = np.mean([r["sf_f1"] for r in subset])
            sf_f1_5 = np.mean([r["sf_f1_at5"] for r in subset])
            print(f"  {qtype:12s} (n={len(subset):3d}): "
                  f"R@2={r2:.3f}  MRR={mrr:.3f}  "
                  f"SF_EM={sf_em:.3f}  SF_F1={sf_f1:.3f}  SF_F1@5={sf_f1_5:.3f}  "
                  f"lat={lat:.0f}ms")
        else:
            print(f"  {qtype:12s} (n={len(subset):3d}): "
                  f"R@2={r2:.3f}  R@4={r4:.3f}  P@2={p2:.3f}  MRR={mrr:.3f}  "
                  f"lat={lat:.0f}ms")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="experiments/hotpotqa_v2/data/hotpotqa_sample_100.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm")

    data = load_hotpotqa(args.data, limit=args.limit)
    print(f"Loaded {len(data)} HotpotQA questions")

    types = defaultdict(int)
    for d in data:
        types[d["type"]] += 1
    print(f"  Types: {dict(types)}")

    # Run AGHT
    print(f"\n--- AGHT (Spec Z) ---")
    aght_results = evaluate_aght(data, nlp)
    print_summary("AGHT (Spec Z)", aght_results)

    # Run Legacy
    print(f"\n--- Legacy (geDIG refine) ---")
    legacy_results = evaluate_legacy(data, nlp)
    print_summary("Legacy (geDIG refine)", legacy_results)

    # Side-by-side comparison
    print(f"\n{'='*60}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*60}")
    for qtype in ["all", "bridge", "comparison"]:
        if qtype == "all":
            a_sub = aght_results
            l_sub = legacy_results
        else:
            a_sub = [r for r in aght_results if r["type"] == qtype]
            l_sub = [r for r in legacy_results if r["type"] == qtype]
        if not a_sub:
            continue

        a_r2 = np.mean([r["recall_2"] for r in a_sub])
        l_r2 = np.mean([r["recall_2"] for r in l_sub])
        a_mrr = np.mean([r["mrr"] for r in a_sub])
        l_mrr = np.mean([r["mrr"] for r in l_sub])

        # Sentence-level F1 (AGHT only)
        a_sf_f1 = np.mean([r.get("sf_f1", 0) for r in a_sub])

        delta_r2 = a_r2 - l_r2
        delta_mrr = a_mrr - l_mrr
        winner = "AGHT" if delta_r2 > 0.01 else ("Legacy" if delta_r2 < -0.01 else "TIE")

        print(f"  {qtype:12s}: AGHT R@2={a_r2:.3f} SF_F1={a_sf_f1:.3f} vs Legacy R@2={l_r2:.3f}  "
              f"Δ_R@2={delta_r2:+.3f}  [{winner}]")

    # Save results
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with open(f"{args.output}/aght_results.jsonl", "w") as f:
            for r in aght_results:
                f.write(json.dumps(r) + "\n")
        with open(f"{args.output}/legacy_results.jsonl", "w") as f:
            for r in legacy_results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
