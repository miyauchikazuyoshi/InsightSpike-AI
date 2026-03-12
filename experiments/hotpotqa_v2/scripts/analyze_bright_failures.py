#!/usr/bin/env python3
"""Analyze nDCG=0 vs nDCG>0 queries in BRIGHT adaptive retrieval results."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

RESULTS_DIR = Path("experiments/hotpotqa_v2/results/v12_bright_adaptive")
QUERY_DIR = Path("experiments/hotpotqa_v2/data/bright")

RESULT_FILES = {
    "biology": RESULTS_DIR / "biology_results.jsonl",
    "economics": RESULTS_DIR / "economics_results.jsonl",
    "stackoverflow": RESULTS_DIR / "stackoverflow_results.jsonl",
}

QUERY_FILES = {
    "biology": QUERY_DIR / "biology_queries.jsonl",
    "economics": QUERY_DIR / "economics_queries.jsonl",
    "stackoverflow": QUERY_DIR / "stackoverflow_queries.jsonl",
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_queries(path):
    """Load query file and return dict mapping id -> full query text."""
    queries = {}
    for rec in load_jsonl(path):
        queries[str(rec["id"])] = rec["query"]
    return queries


def main():
    all_zero = []
    all_nonzero = []

    for domain, rpath in RESULT_FILES.items():
        results = load_jsonl(rpath)
        # Also load full query texts from query files
        full_queries = load_queries(QUERY_FILES[domain])

        for r in results:
            # Attach full query from query file (result file query is truncated)
            qid = str(r["query_id"])
            r["full_query"] = full_queries.get(qid, r.get("query", ""))
            r["domain"] = domain

            if r["ndcg_10"] == 0.0:
                all_zero.append(r)
            else:
                all_nonzero.append(r)

    # --- Overall summary ---
    print("=" * 80)
    print("BRIGHT ADAPTIVE RETRIEVAL: nDCG=0 FAILURE ANALYSIS")
    print("=" * 80)
    total = len(all_zero) + len(all_nonzero)
    print(f"\nTotal queries: {total}")
    print(f"  nDCG=0:  {len(all_zero)} ({100*len(all_zero)/total:.1f}%)")
    print(f"  nDCG>0:  {len(all_nonzero)} ({100*len(all_nonzero)/total:.1f}%)")

    # --- Per-domain breakdown ---
    print(f"\n{'Domain':<15} {'Total':>6} {'nDCG=0':>7} {'nDCG>0':>7} {'Fail%':>7}")
    print("-" * 45)
    for domain in ["biology", "economics", "stackoverflow"]:
        z = [r for r in all_zero if r["domain"] == domain]
        nz = [r for r in all_nonzero if r["domain"] == domain]
        t = len(z) + len(nz)
        print(f"{domain:<15} {t:>6} {len(z):>7} {len(nz):>7} {100*len(z)/t:>6.1f}%")

    # --- Average query length ---
    print("\n" + "=" * 80)
    print("QUERY LENGTH ANALYSIS")
    print("=" * 80)
    zero_lens = [len(r["full_query"]) for r in all_zero]
    nonzero_lens = [len(r["full_query"]) for r in all_nonzero]
    print(f"\nAvg query length (chars):")
    print(f"  nDCG=0:  {sum(zero_lens)/len(zero_lens):.1f}")
    print(f"  nDCG>0:  {sum(nonzero_lens)/len(nonzero_lens):.1f}")

    zero_word_lens = [len(r["full_query"].split()) for r in all_zero]
    nonzero_word_lens = [len(r["full_query"].split()) for r in all_nonzero]
    print(f"\nAvg query length (words):")
    print(f"  nDCG=0:  {sum(zero_word_lens)/len(zero_word_lens):.1f}")
    print(f"  nDCG>0:  {sum(nonzero_word_lens)/len(nonzero_word_lens):.1f}")

    # --- Tier distribution ---
    print("\n" + "=" * 80)
    print("ROUTING TIER DISTRIBUTION")
    print("=" * 80)
    zero_tiers = Counter(r.get("routing_tier", "?") for r in all_zero)
    nonzero_tiers = Counter(r.get("routing_tier", "?") for r in all_nonzero)

    print(f"\n{'Tier':<6} {'nDCG=0':>8} {'%':>7} {'nDCG>0':>8} {'%':>7}")
    print("-" * 40)
    all_tiers = sorted(set(list(zero_tiers.keys()) + list(nonzero_tiers.keys())))
    for t in all_tiers:
        zc = zero_tiers.get(t, 0)
        nc = nonzero_tiers.get(t, 0)
        zp = 100 * zc / len(all_zero) if all_zero else 0
        np_ = 100 * nc / len(all_nonzero) if all_nonzero else 0
        print(f"{t:<6} {zc:>8} {zp:>6.1f}% {nc:>8} {np_:>6.1f}%")

    # --- Tier distribution per domain ---
    print("\nPer-domain tier distribution for nDCG=0:")
    for domain in ["biology", "economics", "stackoverflow"]:
        z = [r for r in all_zero if r["domain"] == domain]
        tc = Counter(r.get("routing_tier", "?") for r in z)
        parts = [f"T{k}={v}" for k, v in sorted(tc.items())]
        print(f"  {domain:<15} {', '.join(parts)}")

    # --- CoT analysis ---
    print("\n" + "=" * 80)
    print("CHAIN-OF-THOUGHT RE-RETRIEVAL ANALYSIS")
    print("=" * 80)

    # nDCG=0 with n_cot_new_gold > 0
    zero_with_new_gold = [r for r in all_zero if r.get("n_cot_new_gold", 0) > 0]
    print(f"\nnDCG=0 queries with n_cot_new_gold > 0: {len(zero_with_new_gold)} / {len(all_zero)}")
    print("  (These found new gold docs via re-retrieval but STILL scored 0)")

    # cot_skipped analysis
    zero_skipped = [r for r in all_zero if r.get("cot_skipped", False)]
    nonzero_skipped = [r for r in all_nonzero if r.get("cot_skipped", False)]
    print(f"\nCoT skipped (tier 1, no re-retrieval):")
    print(f"  nDCG=0: {len(zero_skipped)} / {len(all_zero)} ({100*len(zero_skipped)/len(all_zero):.1f}%)")
    print(f"  nDCG>0: {len(nonzero_skipped)} / {len(all_nonzero)} ({100*len(nonzero_skipped)/len(all_nonzero) if all_nonzero else 0:.1f}%)")

    # n_cot_retrieved stats
    zero_cot_retrieved = [r.get("n_cot_retrieved", 0) for r in all_zero]
    nonzero_cot_retrieved = [r.get("n_cot_retrieved", 0) for r in all_nonzero]
    print(f"\nAvg n_cot_retrieved:")
    print(f"  nDCG=0: {sum(zero_cot_retrieved)/len(zero_cot_retrieved):.1f}")
    print(f"  nDCG>0: {sum(nonzero_cot_retrieved)/len(nonzero_cot_retrieved):.1f}")

    # n_cot_new_gold stats
    zero_cot_new_gold = [r.get("n_cot_new_gold", 0) for r in all_zero]
    nonzero_cot_new_gold = [r.get("n_cot_new_gold", 0) for r in all_nonzero]
    print(f"\nAvg n_cot_new_gold:")
    print(f"  nDCG=0: {sum(zero_cot_new_gold)/len(zero_cot_new_gold):.2f}")
    print(f"  nDCG>0: {sum(nonzero_cot_new_gold)/len(nonzero_cot_new_gold):.2f}")

    # --- beta_0 (initial gold hits) analysis ---
    print("\n" + "=" * 80)
    print("INITIAL RETRIEVAL (beta_0) ANALYSIS")
    print("=" * 80)
    zero_beta0 = [r.get("beta_0", 0) for r in all_zero]
    nonzero_beta0 = [r.get("beta_0", 0) for r in all_nonzero]
    print(f"\nAvg beta_0 (initial gold in top-50 BM25):")
    print(f"  nDCG=0: {sum(zero_beta0)/len(zero_beta0):.2f}")
    print(f"  nDCG>0: {sum(nonzero_beta0)/len(nonzero_beta0):.2f}")

    zero_beta0_zero = sum(1 for b in zero_beta0 if b <= 2)
    nonzero_beta0_zero = sum(1 for b in nonzero_beta0 if b <= 2)
    print(f"\nbeta_0 <= 2 (very few gold in initial retrieval):")
    print(f"  nDCG=0: {zero_beta0_zero} / {len(all_zero)} ({100*zero_beta0_zero/len(all_zero):.1f}%)")
    print(f"  nDCG>0: {nonzero_beta0_zero} / {len(all_nonzero)} ({100*nonzero_beta0_zero/len(all_nonzero):.1f}%)")

    # --- n_gold distribution ---
    print("\n" + "=" * 80)
    print("NUMBER OF GOLD DOCUMENTS")
    print("=" * 80)
    zero_ngold = [r.get("n_gold", 0) for r in all_zero]
    nonzero_ngold = [r.get("n_gold", 0) for r in all_nonzero]
    print(f"\nAvg n_gold:")
    print(f"  nDCG=0: {sum(zero_ngold)/len(zero_ngold):.2f}")
    print(f"  nDCG>0: {sum(nonzero_ngold)/len(nonzero_ngold):.2f}")

    # --- SAMPLE QUERIES ---
    print("\n" + "=" * 80)
    print("SAMPLE QUERIES: nDCG=0 (5 per domain)")
    print("=" * 80)

    for domain in ["biology", "economics", "stackoverflow"]:
        z = [r for r in all_zero if r["domain"] == domain]
        sample = random.sample(z, min(5, len(z)))
        print(f"\n--- {domain.upper()} (nDCG=0) ---")
        for i, r in enumerate(sample, 1):
            q = r["full_query"]
            # Truncate for display
            q_display = q[:200] + "..." if len(q) > 200 else q
            print(f"\n  [{i}] query_id={r['query_id']}")
            print(f"      tier={r.get('routing_tier','?')}  beta_0={r.get('beta_0',0)}  n_gold={r.get('n_gold',0)}")
            print(f"      n_cot_retrieved={r.get('n_cot_retrieved',0)}  n_cot_new_gold={r.get('n_cot_new_gold',0)}  cot_skipped={r.get('cot_skipped',False)}")
            print(f"      gold_hits_top10={r.get('gold_hits_top10',0)}")
            print(f"      query: {q_display}")

    print("\n" + "=" * 80)
    print("SAMPLE QUERIES: nDCG>0 (5 per domain)")
    print("=" * 80)

    for domain in ["biology", "economics", "stackoverflow"]:
        nz = [r for r in all_nonzero if r["domain"] == domain]
        sample = random.sample(nz, min(5, len(nz)))
        print(f"\n--- {domain.upper()} (nDCG>0) ---")
        for i, r in enumerate(sample, 1):
            q = r["full_query"]
            q_display = q[:200] + "..." if len(q) > 200 else q
            print(f"\n  [{i}] query_id={r['query_id']}  ndcg_10={r['ndcg_10']:.4f}")
            print(f"      tier={r.get('routing_tier','?')}  beta_0={r.get('beta_0',0)}  n_gold={r.get('n_gold',0)}")
            print(f"      n_cot_retrieved={r.get('n_cot_retrieved',0)}  n_cot_new_gold={r.get('n_cot_new_gold',0)}  cot_skipped={r.get('cot_skipped',False)}")
            print(f"      gold_hits_top10={r.get('gold_hits_top10',0)}")
            print(f"      query: {q_display}")

    # --- Cross-tab: tier x cot_skipped x domain for nDCG=0 ---
    print("\n" + "=" * 80)
    print("nDCG=0 CROSS-TAB: domain x tier x cot_skipped")
    print("=" * 80)
    for domain in ["biology", "economics", "stackoverflow"]:
        z = [r for r in all_zero if r["domain"] == domain]
        print(f"\n  {domain}:")
        crosstab = defaultdict(int)
        for r in z:
            key = (r.get("routing_tier", "?"), r.get("cot_skipped", False))
            crosstab[key] += 1
        for (tier, skipped), count in sorted(crosstab.items()):
            print(f"    tier={tier}, cot_skipped={skipped}: {count}")

    # --- Among nDCG=0 + cot NOT skipped: what fraction found ANY new gold? ---
    print("\n" + "=" * 80)
    print("nDCG=0 + CoT NOT SKIPPED: Did re-retrieval help at all?")
    print("=" * 80)
    zero_cot_ran = [r for r in all_zero if not r.get("cot_skipped", False)]
    zero_cot_ran_found = [r for r in zero_cot_ran if r.get("n_cot_new_gold", 0) > 0]
    print(f"\n  CoT ran but nDCG=0: {len(zero_cot_ran)}")
    print(f"  Of those, found new gold (n_cot_new_gold>0): {len(zero_cot_ran_found)} ({100*len(zero_cot_ran_found)/len(zero_cot_ran) if zero_cot_ran else 0:.1f}%)")
    print(f"  Of those, found NO new gold: {len(zero_cot_ran) - len(zero_cot_ran_found)}")

    if zero_cot_ran_found:
        print(f"\n  These {len(zero_cot_ran_found)} queries found gold via CoT but STILL scored nDCG=0:")
        for r in zero_cot_ran_found:
            print(f"    {r['domain']}/q{r['query_id']}: n_cot_new_gold={r['n_cot_new_gold']}, gold_hits_top10={r.get('gold_hits_top10',0)}, n_gold={r.get('n_gold',0)}")


if __name__ == "__main__":
    main()
