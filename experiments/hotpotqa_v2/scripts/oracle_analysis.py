#!/usr/bin/env python3
"""Phase 1: Perfect-router oracle analysis.

For each question, route to whichever arm (Hybrid-E1 vs IRCoT) answered
correctly, and compute the resulting EM ceiling. This bounds what ANY
routing signal (F, logprob, classifier, ...) could achieve by choosing
between the two pipelines per question.

Decision gate (pre-registered in the 2026-06-10 external review):
  ceiling - best_single < +2pt  -> retreat from the routing line
  ceiling - best_single >= +8pt -> invest in Phase 2 (router duel)
  otherwise                     -> re-measure on MuSiQue before deciding
"""
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "results"

PAIRS = {
    "gpt-4o": ("500q_hybrid_e1_4o", "500q_ircot_4o"),
    "gpt-4o-mini": ("500q_hybrid_e1_mini", "500q_ircot_mini"),
}


def load(run_dir: str) -> dict:
    rows = {}
    with open(BASE / run_dir / "results.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["example_id"]] = r
    return rows


def analyze(model: str, hyb_dir: str, irc_dir: str) -> dict:
    hyb, irc = load(hyb_dir), load(irc_dir)
    ids = sorted(set(hyb) & set(irc))
    n = len(ids)

    def em(r):
        return 1 if float(r.get("em", 0.0)) >= 1.0 else 0

    hyb_em = sum(em(hyb[i]) for i in ids)
    irc_em = sum(em(irc[i]) for i in ids)
    oracle = sum(max(em(hyb[i]), em(irc[i])) for i in ids)
    floor = sum(min(em(hyb[i]), em(irc[i])) for i in ids)

    both = sum(1 for i in ids if em(hyb[i]) and em(irc[i]))
    hyb_only = sum(1 for i in ids if em(hyb[i]) and not em(irc[i]))
    irc_only = sum(1 for i in ids if not em(hyb[i]) and em(irc[i]))
    neither = n - both - hyb_only - irc_only

    by_type = defaultdict(lambda: {"n": 0, "hyb": 0, "irc": 0, "oracle": 0})
    for i in ids:
        qt = hyb[i].get("question_type", "unknown")
        s = by_type[qt]
        s["n"] += 1
        s["hyb"] += em(hyb[i])
        s["irc"] += em(irc[i])
        s["oracle"] += max(em(hyb[i]), em(irc[i]))

    best_single = max(hyb_em, irc_em)
    headroom_pt = (oracle - best_single) / n * 100

    if headroom_pt < 2.0:
        gate = "RETREAT (<+2pt): routing between these two arms cannot pay off"
    elif headroom_pt >= 8.0:
        gate = "INVEST (>=+8pt): proceed to Phase 2 router duel"
    else:
        gate = "INTERMEDIATE: re-measure on MuSiQue before deciding"

    return {
        "model": model,
        "arms": {"hybrid": hyb_dir, "ircot": irc_dir},
        "n": n,
        "em": {
            "hybrid": round(hyb_em / n, 4),
            "ircot": round(irc_em / n, 4),
            "oracle_ceiling": round(oracle / n, 4),
            "intersection_floor": round(floor / n, 4),
        },
        "headroom_vs_best_single_pt": round(headroom_pt, 2),
        "contingency": {
            "both_correct": both,
            "hybrid_only": hyb_only,
            "ircot_only": irc_only,
            "neither": neither,
        },
        "by_question_type": {
            qt: {
                "n": s["n"],
                "hybrid_em": round(s["hyb"] / s["n"], 4),
                "ircot_em": round(s["irc"] / s["n"], 4),
                "oracle_em": round(s["oracle"] / s["n"], 4),
            }
            for qt, s in sorted(by_type.items())
        },
        "decision_gate": gate,
    }


def main():
    out = {}
    for model, (hyb_dir, irc_dir) in PAIRS.items():
        res = analyze(model, hyb_dir, irc_dir)
        out[model] = res
        e = res["em"]
        c = res["contingency"]
        print(f"\n=== {model} (n={res['n']}) ===")
        print(f"  Hybrid-E1 EM : {e['hybrid']:.1%}")
        print(f"  IRCoT EM     : {e['ircot']:.1%}")
        print(f"  Oracle EM    : {e['oracle_ceiling']:.1%}  (floor {e['intersection_floor']:.1%})")
        print(f"  Headroom     : +{res['headroom_vs_best_single_pt']:.1f}pt vs best single arm")
        print(f"  Contingency  : both={c['both_correct']} hyb_only={c['hybrid_only']} "
              f"irc_only={c['ircot_only']} neither={c['neither']}")
        print(f"  GATE         : {res['decision_gate']}")
        for qt, s in res["by_question_type"].items():
            print(f"    [{qt}] n={s['n']} hyb={s['hybrid_em']:.1%} "
                  f"irc={s['ircot_em']:.1%} oracle={s['oracle_em']:.1%}")

    out_path = BASE.parent / "results" / "oracle_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
