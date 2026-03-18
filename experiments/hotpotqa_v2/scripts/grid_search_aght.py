#!/usr/bin/env python3
"""Grid search AGHT parameters on HotpotQA 100q.

Searches over the most impactful parameters:
1. mp_alpha (message passing smoothing): [0.1, 0.2, 0.3, 0.5]
2. mp_iterations: [1, 2, 3, 5]
3. w_q1 (direct match weight): [0.5, 0.7, 1.0]
4. f_lambda: [0.5, 1.0, 1.5]

Total: 4 × 4 × 3 × 3 = 144 configs
Each config takes ~2min → ~5 hours

For speed, we do 2-phase:
  Phase 1: Coarse grid on 30q (fast)
  Phase 2: Fine grid around best on 100q
"""

import json
import sys
import time
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def evaluate_config(data, nlp, config_overrides: dict) -> dict:
    """Run AGHT with specific config on HotpotQA data."""
    from unified_graph import run_aght, AGHTConfig

    config = AGHTConfig(**config_overrides)

    r2_list, mrr_list, sf_f1_list = [], [], []

    for item in data:
        query = item["question"]
        ctx = item["context"]
        titles_raw = ctx["title"]
        sents_raw = ctx["sentences"]
        gold_titles = set(item["supporting_facts"]["title"])
        gold_sf = item["supporting_facts"]
        gold_sf_set = set(zip(gold_sf["title"], gold_sf["sent_id"]))

        titles = [f"doc_{j}" for j in range(len(titles_raw))]
        sentences_list = []
        doc_id_map = {}
        for j, (title, sents) in enumerate(zip(titles_raw, sents_raw)):
            clean_sents = [s.strip() for s in sents if len(s.strip()) > 5]
            if not clean_sents:
                clean_sents = [" ".join(sents)[:500]]
            sentences_list.append(clean_sents[:30])
            doc_id_map[f"doc_{j}"] = title

        scores, _, aght_graph = run_aght(
            titles, sentences_list, query, nlp, doc_id_map,
            config=config,
        )

        # Paragraph R@2
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        ranked_titles = [t for t, _ in ranked]
        top2 = set(ranked_titles[:2])
        r2_list.append(len(gold_titles & top2) / len(gold_titles))

        # MRR
        mrr = 0.0
        for rank, title in enumerate(ranked_titles, 1):
            if title in gold_titles:
                mrr = 1.0 / rank
                break
        mrr_list.append(mrr)

        # Sentence SF F1
        sent_scores = []
        for nid, ndata in aght_graph.nodes(data=True):
            if ndata.get("node_type") != "S":
                continue
            p_idx = ndata.get("para_idx", -1)
            s_idx = ndata.get("sent_idx", -1)
            rel = ndata.get("relevance", 0.0)
            if 0 <= p_idx < len(titles_raw):
                sent_scores.append((titles_raw[p_idx], s_idx, rel))

        sent_scores.sort(key=lambda x: -x[2])
        n_gold = len(gold_sf_set)
        pred = set((t, s) for t, s, _ in sent_scores[:n_gold])
        tp = len(pred & gold_sf_set)
        p = tp / max(len(pred), 1)
        r = tp / max(n_gold, 1)
        f1 = 2 * p * r / max(p + r, 1e-10)
        sf_f1_list.append(f1)

    return {
        "r2": np.mean(r2_list),
        "mrr": np.mean(mrr_list),
        "sf_f1": np.mean(sf_f1_list),
        "combined": np.mean(r2_list) * 0.4 + np.mean(sf_f1_list) * 0.6,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="experiments/hotpotqa_v2/data/hotpotqa_sample_100.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", default="experiments/hotpotqa_v2/results/v35_grid_search")
    args = parser.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm")

    data = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            data.append(json.loads(line))
    print(f"Loaded {len(data)} questions for grid search", flush=True)

    # Parameter grid
    grid = {
        "mp_alpha": [0.1, 0.2, 0.3, 0.5],
        "mp_iterations": [1, 2, 3, 5],
        "w_q1": [0.5, 0.7, 1.0],
        "f_lambda": [0.5, 1.0, 1.5],
    }

    keys = list(grid.keys())
    values = list(grid.values())
    configs = list(itertools.product(*values))
    print(f"Total configs: {len(configs)}", flush=True)

    results = []
    best_combined = -1
    best_config = None

    for ci, combo in enumerate(configs):
        config_dict = dict(zip(keys, combo))
        t0 = time.time()
        metrics = evaluate_config(data, nlp, config_dict)
        elapsed = time.time() - t0

        results.append({**config_dict, **metrics})

        if metrics["combined"] > best_combined:
            best_combined = metrics["combined"]
            best_config = config_dict

        if (ci + 1) % 12 == 0 or ci == len(configs) - 1:
            print(f"  [{ci+1}/{len(configs)}] "
                  f"R@2={metrics['r2']:.3f} SF_F1={metrics['sf_f1']:.3f} "
                  f"combined={metrics['combined']:.3f} "
                  f"({elapsed:.1f}s) "
                  f"| best={best_combined:.3f}", flush=True)

    # Sort by combined score
    results.sort(key=lambda x: -x["combined"])

    print(f"\n{'='*70}")
    print(f"  TOP 10 CONFIGURATIONS")
    print(f"{'='*70}")
    print(f"{'mp_a':>5} {'mp_i':>4} {'w_q1':>5} {'f_λ':>5} | "
          f"{'R@2':>5} {'SF_F1':>6} {'MRR':>5} {'comb':>5}")
    print("-" * 55)
    for r in results[:10]:
        print(f"{r['mp_alpha']:>5.1f} {r['mp_iterations']:>4d} {r['w_q1']:>5.1f} {r['f_lambda']:>5.1f} | "
              f"{r['r2']:>5.3f} {r['sf_f1']:>6.3f} {r['mrr']:>5.3f} {r['combined']:>5.3f}")

    print(f"\nBest config: {best_config}")
    print(f"Best combined: {best_combined:.4f}")

    # Save
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output}/grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{args.output}/best_config.json", "w") as f:
        json.dump({"config": best_config, "metrics": {"combined": best_combined}}, f, indent=2)
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
