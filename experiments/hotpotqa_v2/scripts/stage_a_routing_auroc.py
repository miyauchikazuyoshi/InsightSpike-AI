#!/usr/bin/env python3
"""Phase 2 / Stage A: retrospective routing-signal AUROC (cost $0).

Pre-registered in docs/prereg/phase2_router_duel.md BEFORE this run:
  Target (primary): did Hybrid-E1 answer correctly? (em == 1)
  P1: AUROC(extended_f) > 0.5 (orientation reported; flips noted honestly)
  P2: AUROC(extended_f) > AUROC(question-type router)
  P3: deferred to Stage B (logprob/margin not logged retrospectively)

Caveat (also pre-registered): extended_f was logged AFTER system selection,
so Stage A carries selection bias; results are preliminary.
"""
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "results"
RUNS = {"gpt-4o": "500q_hybrid_e1_4o", "gpt-4o-mini": "500q_hybrid_e1_mini"}
N_BOOT = 2000
random.seed(42)


def load(run_dir: str) -> list:
    rows = []
    with open(BASE / run_dir / "results.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auroc(scores, labels) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties handled by midranks."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        mid = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    u = rank_sum_pos - pos * (pos + 1) / 2
    return u / (pos * neg)


def boot_ci(fn, idx_n, alpha=0.05):
    vals = sorted(fn([random.randrange(idx_n) for _ in range(idx_n)]) for _ in range(N_BOOT))
    lo = vals[int(alpha / 2 * N_BOOT)]
    hi = vals[int((1 - alpha / 2) * N_BOOT) - 1]
    return lo, hi


def analyze(model: str, run_dir: str) -> dict:
    rows = [r for r in load(run_dir) if r.get("extended_f") is not None]
    n = len(rows)
    y = [1 if float(r.get("em", 0.0)) >= 1.0 else 0 for r in rows]
    f_sig = [float(r["extended_f"]) for r in rows]
    # question-type router score: comparison -> route to Hybrid (score 1)
    t_sig = [1.0 if r.get("question_type") == "comparison" else 0.0 for r in rows]

    a_f_raw = auroc(f_sig, y)
    # orientation: theory reads LOW F = good structure = System 1 suffices,
    # so the theory-oriented score is -F. Report both.
    a_f = auroc([-s for s in f_sig], y)
    a_t = auroc(t_sig, y)

    def f_stat(idx):
        return auroc([-f_sig[i] for i in idx], [y[i] for i in idx])

    def t_stat(idx):
        return auroc([t_sig[i] for i in idx], [y[i] for i in idx])

    def diff_stat(idx):
        return f_stat(idx) - t_stat(idx)

    f_lo, f_hi = boot_ci(f_stat, n)
    t_lo, t_hi = boot_ci(t_stat, n)
    d_lo, d_hi = boot_ci(diff_stat, n)

    # exploratory: other logged signals
    extras = {}
    for key in ("dg_fired", "ag_fired", "cot_steps"):
        if all(key in r for r in rows):
            sig = [float(bool(r[key])) if isinstance(r[key], bool) else float(r[key]) for r in rows]
            extras[key] = round(auroc(sig, y), 4)

    return {
        "model": model,
        "n": n,
        "base_rate_hybrid_correct": round(sum(y) / n, 4),
        "auroc": {
            "extended_f_raw_orientation": round(a_f_raw, 4),
            "extended_f_theory_oriented(-F)": round(a_f, 4),
            "extended_f_CI95": [round(f_lo, 4), round(f_hi, 4)],
            "type_router": round(a_t, 4),
            "type_router_CI95": [round(t_lo, 4), round(t_hi, 4)],
            "diff_F_minus_type": round(a_f - a_t, 4),
            "diff_CI95": [round(d_lo, 4), round(d_hi, 4)],
        },
        "exploratory_auroc": extras,
        "P1_auroc_above_0.5": bool(f_lo > 0.5),
        "P2_F_beats_type_router": bool(d_lo > 0.0),
    }


def main():
    out = {}
    for model, run_dir in RUNS.items():
        res = analyze(model, run_dir)
        out[model] = res
        a = res["auroc"]
        print(f"\n=== {model} (n={res['n']}, base rate {res['base_rate_hybrid_correct']:.1%}) ===")
        print(f"  AUROC(-F)        : {a['extended_f_theory_oriented(-F)']:.3f}  CI95 {a['extended_f_CI95']}")
        print(f"  AUROC(F raw)     : {a['extended_f_raw_orientation']:.3f}")
        print(f"  AUROC(type)      : {a['type_router']:.3f}  CI95 {a['type_router_CI95']}")
        print(f"  diff (-F - type) : {a['diff_F_minus_type']:+.3f}  CI95 {a['diff_CI95']}")
        print(f"  P1 (AUROC>0.5)   : {'PASS' if res['P1_auroc_above_0.5'] else 'FAIL'}")
        print(f"  P2 (F > type)    : {'PASS' if res['P2_F_beats_type_router'] else 'FAIL'}")
        print(f"  exploratory      : {res['exploratory_auroc']}")

    out_path = BASE / "stage_a_routing_auroc.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
