#!/usr/bin/env python3
"""All-Context experiments for MuSiQue.

Conditions:
  baseline_a      : Feed all paragraphs to LLM, plain CoT
  v9_guided       : Feed all 20 paragraphs + title cross-reference hints
  v10_guided      : Feed all 20 paragraphs + entity-graph reasoning chain
  v10_reordered   : Chain paragraphs first + entity-graph reasoning chain
  v10_reorder_only: Chain paragraphs first, NO reasoning guide (baseline prompt)
  v11_subgraph    : Pre-computed graph → subgraph extraction → answer from subgraph only
  v11_routing     : Pre-computed graph → F-value → System1 (subgraph) / System2 (full)

Usage:
  python run_allcontext.py --mode baseline_a     --data DATA --output DIR [--limit N]
  python run_allcontext.py --mode v9_guided      --data DATA --output DIR [--limit N]
  python run_allcontext.py --mode v10_guided     --data DATA --output DIR [--limit N]
  python run_allcontext.py --mode v10_reordered  --data DATA --output DIR [--limit N]
  python run_allcontext.py --mode v11_subgraph   --data DATA --output DIR [--limit N]
  python run_allcontext.py --mode v11_routing    --data DATA --output DIR [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

# Add src to path for entity_graph import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from entity_graph import (
    extract_reasoning_chain,
    format_reasoning_guide,
    reorder_paragraphs,
)

# ---------------------------------------------------------------------------
# Evaluation helpers (copied from evaluator.py for standalone use)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    import string
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))

def compute_f1(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt)
    r = n / len(gt)
    return 2 * p * r / (p + r)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

BASELINE_A_PROMPT = """\
Read ALL of the following paragraphs carefully, then answer the question.

{paragraphs}

Question: {question}

Think step by step. Identify which paragraphs contain relevant information, \
trace the reasoning chain across paragraphs, then give your final answer.

IMPORTANT: Your final answer must be a short phrase (a few words). \
Write it on the last line after "Answer: "."""

V9_GUIDED_PROMPT = """\
Read ALL of the following paragraphs carefully, then answer the question.

{paragraphs}

=== REASONING GUIDE ===
Structural analysis of the paragraphs reveals these key connections:
{reasoning_hints}

Use these connections to trace the multi-hop reasoning chain.
=== END GUIDE ===

Question: {question}

Think step by step following the reasoning chain above, then give your final answer.

IMPORTANT: Your final answer must be a short phrase (a few words). \
Write it on the last line after "Answer: "."""

V10_GUIDED_PROMPT = """\
Read ALL of the following paragraphs carefully, then answer the question.

{paragraphs}

=== REASONING GUIDE ===
{reasoning_guide}
=== END GUIDE ===

Question: {question}

Follow the reasoning guide above to trace the multi-hop chain.
Your final answer must be a short phrase. Write it after "Answer: "."""

# ---------------------------------------------------------------------------
# v11 prompts (NO reasoning guide — v10 finding: "guided text hurts GPT-4o")
# ---------------------------------------------------------------------------

V11_SUBGRAPH_PROMPT = """\
Read the following paragraphs carefully, then answer the question.

{paragraphs}

Question: {question}

Think step by step. Identify which paragraphs contain relevant information, \
trace the reasoning chain across paragraphs, then give your final answer.

IMPORTANT: Your final answer must be a short phrase (a few words). \
Write it on the last line after "Answer: "."""

V11_SYSTEM2_PROMPT = """\
Read ALL of the following paragraphs carefully, then answer the question.
The most relevant paragraphs are listed first.

{paragraphs}

Question: {question}

Think step by step. Focus especially on the first few paragraphs which are \
most likely to contain key information. Trace the reasoning chain, then give \
your final answer.

IMPORTANT: Your final answer must be a short phrase (a few words). \
Write it on the last line after "Answer: "."""

# ---------------------------------------------------------------------------
# Graph-guided reasoning hints (v9)
# ---------------------------------------------------------------------------

def build_paragraph_graph(titles: list[str], sentences_list: list[list[str]]) -> dict:
    """Build a paragraph-level cross-reference graph.

    Edge (i -> j) if paragraph i's text mentions title j.
    Returns adjacency dict: {title_i: [title_j, ...]}
    """
    # Combine sentences into full text per paragraph
    texts = {}
    for title, sents in zip(titles, sentences_list):
        texts[title] = " ".join(sents)

    # Build edges: paragraph i references title j
    adj: dict[str, list[str]] = {t: [] for t in titles}
    for src_title in titles:
        src_text = texts[src_title].lower()
        for dst_title in titles:
            if src_title == dst_title:
                continue
            if dst_title.lower() in src_text:
                adj[src_title].append(dst_title)

    return adj

def find_reasoning_chain(
    question: str,
    titles: list[str],
    sentences_list: list[list[str]],
) -> str:
    """Identify key cross-references between paragraphs to guide reasoning."""
    adj = build_paragraph_graph(titles, sentences_list)

    # Find paragraphs mentioned in the question
    q_lower = question.lower()
    q_relevant = [t for t in titles if t.lower() in q_lower]

    hints = []

    # Cross-reference edges (most important signal)
    edges = []
    for src, dsts in adj.items():
        for dst in dsts:
            edges.append((src, dst))

    if edges:
        hints.append("Cross-references between paragraphs:")
        for src, dst in edges[:15]:  # cap at 15 edges
            hints.append(f'  - "{src}" mentions "{dst}"')

    # Question-relevant paragraphs
    if q_relevant:
        hints.append(f"\nParagraphs directly related to the question:")
        for t in q_relevant:
            hints.append(f'  - "{t}"')

    # Hub paragraphs (referenced by many others)
    in_degree = Counter()
    for src, dsts in adj.items():
        for dst in dsts:
            in_degree[dst] += 1
    hubs = in_degree.most_common(3)
    if hubs:
        hints.append(f"\nKey hub paragraphs (referenced by multiple others):")
        for title, deg in hubs:
            hints.append(f'  - "{title}" (referenced {deg} times)')

    # Connected components
    # Simple BFS to find components
    visited = set()
    components = []
    for t in titles:
        if t in visited:
            continue
        component = set()
        stack = [t]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            # Undirected: follow edges in both directions
            for dst in adj.get(node, []):
                if dst not in visited:
                    stack.append(dst)
            for src, dsts in adj.items():
                if node in dsts and src not in visited:
                    stack.append(src)
        components.append(component)

    if len(components) > 1:
        hints.append(f"\n⚠ {len(components)} disconnected groups detected — look for implicit connections:")
        for i, comp in enumerate(sorted(components, key=len, reverse=True)[:3]):
            sample = list(comp)[:3]
            hints.append(f'  Group {i+1}: {", ".join(sample)}...')

    if not hints:
        hints.append("No strong cross-references detected. Read all paragraphs carefully.")

    return "\n".join(hints)

# ---------------------------------------------------------------------------
# Format paragraphs
# ---------------------------------------------------------------------------

def format_paragraphs(titles: list[str], sentences_list: list[list[str]]) -> str:
    parts = []
    for i, (title, sents) in enumerate(zip(titles, sentences_list), 1):
        text = " ".join(sents)
        parts.append(f"[{i}] {title}: {text}")
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, prompt: str, model: str = "gpt-4o") -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt * 5
                print(f"  [rate-limit] retry {attempt+1}/3 in {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("LLM call failed after 3 retries")

def extract_answer(response: str) -> str:
    """Extract the final answer from LLM response."""
    # Look for "Answer: ..." pattern
    match = re.search(r"(?:Answer|ANSWER|Final Answer|final answer)\s*:\s*(.+?)(?:\n|$)",
                      response, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")
    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="All-context MuSiQue experiments")
    parser.add_argument("--mode", required=True,
                        choices=["baseline_a", "v9_guided", "v10_guided",
                                 "v10_reordered", "v10_reorder_only",
                                 "v10_pruned",
                                 "v11_subgraph", "v11_routing"])
    parser.add_argument("--data", required=True, help="Path to JSONL data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    # v11 arguments
    parser.add_argument("--k-hop", type=int, default=2,
                        help="v11: k-hop radius for subgraph extraction (default: 2)")
    parser.add_argument("--max-subgraph-paras", type=int, default=10,
                        help="v11: max paragraphs in extracted subgraph (default: 10)")
    parser.add_argument("--theta-f", type=float, default=0.0,
                        help="v11: F-value threshold for routing (default: 0.0)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.jsonl"

    # Load data
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    if args.limit > 0:
        examples = examples[:args.limit]

    # Resume support
    completed_ids: set[str] = set()
    if args.resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                completed_ids.add(r["example_id"])
        print(f"[run] Resuming: {len(completed_ids)} already completed")

    client = OpenAI()
    total_em, total_f1 = 0.0, 0.0
    n_done = len(completed_ids)

    print(f"[run] Mode: {args.mode}, Data: {len(examples)}q, Model: {args.model}")

    with open(results_file, "a" if args.resume else "w") as fout:
        for i, ex in enumerate(examples):
            eid = ex["id"]
            if eid in completed_ids:
                continue

            question = ex["question"]
            answer = ex["answer"]
            ctx = ex["context"]
            titles = ctx["title"]
            sentences_list = ctx["sentences"]

            # v10: extract reasoning chain (used by all v10 modes)
            chain_info = None
            if args.mode in ("v10_guided", "v10_reordered", "v10_reorder_only",
                             "v10_pruned"):
                chain_info = extract_reasoning_chain(
                    question, titles, sentences_list
                )

            # v11: corpus graph + routing
            routing = None
            if args.mode.startswith("v11"):
                from corpus_graph import CorpusGraphBuilder

                builder = CorpusGraphBuilder(
                    k_hop=args.k_hop,
                    max_subgraph_paras=args.max_subgraph_paras,
                )
                builder.build(titles, sentences_list)

                gold_titles = list(set(ex["supporting_facts"]["title"]))
                routing = builder.route(
                    question, titles, sentences_list,
                    theta_f=args.theta_f, gold_titles=gold_titles,
                )

            # Build prompt
            if args.mode.startswith("v11") and routing is not None:
                # v11 modes: use routing decision to select paragraphs
                ctx_indices = routing.context_paragraphs
                ctx_titles = [titles[i] for i in ctx_indices]
                ctx_sents = [sentences_list[i] for i in ctx_indices]
                paragraphs_text = format_paragraphs(ctx_titles, ctx_sents)

                if args.mode == "v11_subgraph":
                    prompt = V11_SUBGRAPH_PROMPT.format(
                        paragraphs=paragraphs_text,
                        question=question,
                    )
                elif args.mode == "v11_routing":
                    if routing.system == "system1":
                        prompt = V11_SUBGRAPH_PROMPT.format(
                            paragraphs=paragraphs_text,
                            question=question,
                        )
                    else:  # system2
                        prompt = V11_SYSTEM2_PROMPT.format(
                            paragraphs=paragraphs_text,
                            question=question,
                        )
            elif args.mode == "v10_pruned" and chain_info and chain_info["chain_found"]:
                # Pruning: feed ONLY chain paragraphs (no distractors)
                chain_indices = chain_info["chain"]
                pruned_titles = [titles[i] for i in chain_indices]
                pruned_sents = [sentences_list[i] for i in chain_indices]
                paragraphs_text = format_paragraphs(pruned_titles, pruned_sents)
                prompt = BASELINE_A_PROMPT.format(
                    paragraphs=paragraphs_text,
                    question=question,
                )
            elif args.mode in ("v10_reordered", "v10_reorder_only") and chain_info and chain_info["chain_found"]:
                # Reorder paragraphs: chain first
                ro_titles, ro_sents = reorder_paragraphs(
                    titles, sentences_list, chain_info["chain"]
                )
                paragraphs_text = format_paragraphs(ro_titles, ro_sents)
                if args.mode == "v10_reorder_only":
                    prompt = BASELINE_A_PROMPT.format(
                        paragraphs=paragraphs_text,
                        question=question,
                    )
                else:  # v10_reordered
                    guide = format_reasoning_guide(chain_info)
                    prompt = V10_GUIDED_PROMPT.format(
                        paragraphs=paragraphs_text,
                        question=question,
                        reasoning_guide=guide,
                    )
            else:
                paragraphs_text = format_paragraphs(titles, sentences_list)
                if args.mode in ("baseline_a",):
                    prompt = BASELINE_A_PROMPT.format(
                        paragraphs=paragraphs_text,
                        question=question,
                    )
                elif args.mode == "v9_guided":
                    hints = find_reasoning_chain(question, titles, sentences_list)
                    prompt = V9_GUIDED_PROMPT.format(
                        paragraphs=paragraphs_text,
                        question=question,
                        reasoning_hints=hints,
                    )
                else:  # v10_guided or v10_reordered (chain not found)
                    if chain_info:
                        guide = format_reasoning_guide(chain_info)
                        prompt = V10_GUIDED_PROMPT.format(
                            paragraphs=paragraphs_text,
                            question=question,
                            reasoning_guide=guide,
                        )
                    else:
                        prompt = BASELINE_A_PROMPT.format(
                            paragraphs=paragraphs_text,
                            question=question,
                        )

            t0 = time.time()
            try:
                response = call_llm(client, prompt, args.model)
                prediction = extract_answer(response)
                latency_ms = (time.time() - t0) * 1000
            except Exception as e:
                n_done += 1
                print(f"  [{n_done}/{len(examples)}] ERROR on {eid}: {e}")
                record = {
                    "example_id": eid, "question": question,
                    "ground_truth": answer, "prediction": "",
                    "em": 0, "f1": 0.0, "mode": args.mode,
                    "error": str(e), "latency_ms": 0,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                continue

            em = compute_em(prediction, answer)
            f1 = compute_f1(prediction, answer)
            total_em += em
            total_f1 += f1
            n_done += 1

            # Determine hop count
            hop = eid.split("__")[0] if "__" in eid else "?"

            # Progress display
            v11_info = ""
            if routing is not None:
                v11_info = (f" F={routing.f_value:.2f} {routing.system} "
                            f"sub={len(routing.subgraph.paragraph_indices)}p "
                            f"gR={routing.subgraph.gold_recall:.2f}")
            print(f"  [{n_done}/{len(examples)}] EM={int(em)} F1={f1:.3f} "
                  f"hop={hop}{v11_info} ({latency_ms:.0f}ms) "
                  f"[running EM={total_em/n_done:.1%} F1={total_f1/n_done:.3f}]")

            record = {
                "example_id": eid,
                "question": question,
                "ground_truth": answer,
                "prediction": prediction,
                "em": em,
                "f1": f1,
                "mode": args.mode,
                "hop": hop,
                "latency_ms": latency_ms,
            }

            # v10: add topology features
            if chain_info is not None:
                topo = chain_info["topology"]
                record["beta_0"] = topo["beta_0"]
                record["beta_1"] = topo["beta_1"]
                record["n_bridges"] = topo["n_bridges"]
                record["chain_length"] = len(chain_info["chain"])
                record["chain_found"] = chain_info["chain_found"]
                # v10c: sentence-level and tier diagnostics
                record["n_sent_nodes"] = topo.get("n_sent_nodes", topo["n_nodes"])
                record["n_tier1"] = topo.get("n_tier1_edges", 0)
                record["n_tier2"] = topo.get("n_tier2_edges", 0)
                record["n_tier3"] = topo.get("n_tier3_edges", 0)

            # v11: add routing diagnostics
            if routing is not None:
                record["f_value"] = round(routing.f_value, 4)
                record["system_used"] = routing.system
                record["subgraph_n_paras"] = len(routing.subgraph.paragraph_indices)
                record["subgraph_beta_0"] = routing.subgraph.beta_0
                record["subgraph_beta_1"] = routing.subgraph.beta_1
                record["subgraph_gold_precision"] = round(routing.subgraph.gold_precision, 4)
                record["subgraph_gold_recall"] = round(routing.subgraph.gold_recall, 4)
                record["context_tokens_est"] = routing.context_tokens_est
                record["total_paras"] = len(titles)
                record["query_matched_nodes"] = routing.subgraph.query_matched_nodes

            fout.write(json.dumps(record) + "\n")
            fout.flush()

    # Summary
    if n_done > 0:
        print(f"\n{'='*50}")
        print(f"  {args.mode}: {n_done}q  EM={total_em/n_done:.1%}  F1={total_f1/n_done:.3f}")
        print(f"{'='*50}")

        summary = {
            "mode": args.mode,
            "model": args.model,
            "n": n_done,
            "em": round(total_em / n_done, 4),
            "f1": round(total_f1 / n_done, 4),
        }

        # v11: aggregate routing diagnostics from results file
        if args.mode.startswith("v11"):
            v11_records = []
            with open(results_file) as rf:
                for line in rf:
                    r = json.loads(line)
                    if "f_value" in r:
                        v11_records.append(r)
            if v11_records:
                summary["v11_params"] = {
                    "k_hop": args.k_hop,
                    "max_subgraph_paras": args.max_subgraph_paras,
                    "theta_f": args.theta_f,
                }
                n_s1 = sum(1 for r in v11_records if r.get("system_used") == "system1")
                n_s2 = len(v11_records) - n_s1
                summary["v11_routing"] = {
                    "system1_count": n_s1,
                    "system2_count": n_s2,
                    "system1_pct": round(n_s1 / len(v11_records), 4),
                }
                summary["v11_subgraph"] = {
                    "avg_gold_recall": round(
                        sum(r["subgraph_gold_recall"] for r in v11_records) / len(v11_records), 4),
                    "avg_gold_precision": round(
                        sum(r["subgraph_gold_precision"] for r in v11_records) / len(v11_records), 4),
                    "avg_subgraph_paras": round(
                        sum(r["subgraph_n_paras"] for r in v11_records) / len(v11_records), 2),
                    "avg_f_value": round(
                        sum(r["f_value"] for r in v11_records) / len(v11_records), 4),
                }

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
