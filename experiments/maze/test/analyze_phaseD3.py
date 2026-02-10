#!/usr/bin/env python3
"""Analyze Phase D-3 prefer mode ablation results."""
import json, os, sys
import statistics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "phaseD3_prefer")

CONDITIONS = {
    "A": "override(baseline)",
    "B": "prefer+Q4",
    "C": "prefer+Q4+p0.5",
    "D": "prefer+Q4+p1.0",
    "E": "prefer+p1.0",
}
SEEDS = [0, 1, 2]


def load_summary(cond, seed):
    path = os.path.join(RESULTS_DIR, f"cond{cond}_s{seed}.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: {path}: {e}", file=sys.stderr)
        return None


def load_steps(cond, seed):
    path = os.path.join(RESULTS_DIR, f"cond{cond}_s{seed}_steps.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(summary):
    """Extract key metrics from summary JSON."""
    if summary is None:
        return None
    runs = summary.get("runs", [])
    if not runs:
        return None
    run = runs[0]  # single seed per file
    return {
        "steps": run.get("total_steps", run.get("n_steps", 500)),
        "unique": run.get("unique_cells_visited", 0),
        "goal": run.get("reached_goal", run.get("goal_reached", False)),
        "revisit_ratio": run.get("revisit_ratio", 0),
    }


def main():
    print("=" * 70)
    print("Phase D-3: --sleep-guide prefer Ablation Results")
    print("=" * 70)

    # Per-condition aggregation
    print(f"\n{'Cond':<6} {'Description':<22} {'Steps':>8} {'Unique':>8} {'Goal':>6} {'Revisit%':>9}")
    print("-" * 65)

    cond_stats = {}
    for cond, desc in CONDITIONS.items():
        steps_list, unique_list, goal_list = [], [], []
        for seed in SEEDS:
            summary = load_summary(cond, seed)
            m = extract_metrics(summary)
            if m:
                steps_list.append(m["steps"])
                unique_list.append(m["unique"])
                goal_list.append(m["goal"])

        if steps_list:
            avg_steps = statistics.mean(steps_list)
            avg_unique = statistics.mean(unique_list)
            goal_rate = sum(1 for g in goal_list if g) / len(goal_list)
            revisit = 1.0 - avg_unique / avg_steps if avg_steps > 0 else 0

            cond_stats[cond] = {
                "steps": avg_steps,
                "unique": avg_unique,
                "goal_rate": goal_rate,
                "revisit": revisit,
                "steps_list": steps_list,
                "goal_list": goal_list,
            }
            print(f"{cond:<6} {desc:<22} {avg_steps:>8.1f} {avg_unique:>8.1f} "
                  f"{goal_rate:>5.0%} {revisit:>8.1%}")
        else:
            print(f"{cond:<6} {desc:<22} {'N/A':>8} {'N/A':>8} {'N/A':>6} {'N/A':>9}")

    # Per-seed detail
    print(f"\n{'Cond':<6} {'Seed':>4} {'Steps':>8} {'Unique':>8} {'Goal':>6}")
    print("-" * 40)
    for cond in CONDITIONS:
        for seed in SEEDS:
            summary = load_summary(cond, seed)
            m = extract_metrics(summary)
            if m:
                print(f"{cond:<6} {seed:>4} {m['steps']:>8} {m['unique']:>8} "
                      f"{'YES' if m['goal'] else 'no':>6}")
        print()

    # Pairwise comparisons
    print("=" * 50)
    print("Pairwise Analysis")
    print("=" * 50)
    comparisons = [
        ("A", "B", "override → prefer の効果"),
        ("B", "C", "+ propagated α=0.5 の効果"),
        ("B", "D", "+ propagated α=1.0 の効果"),
        ("D", "E", "Q-bias 有無の差"),
        ("A", "D", "ベースライン vs 最終候補"),
    ]
    for c1, c2, label in comparisons:
        if c1 in cond_stats and c2 in cond_stats:
            s1, s2 = cond_stats[c1], cond_stats[c2]
            delta_steps = s2["steps"] - s1["steps"]
            delta_unique = s2["unique"] - s1["unique"]
            delta_goal = s2["goal_rate"] - s1["goal_rate"]
            print(f"\n{c1}→{c2}: {label}")
            print(f"  steps: {s1['steps']:.1f} → {s2['steps']:.1f} ({delta_steps:+.1f})")
            print(f"  unique: {s1['unique']:.1f} → {s2['unique']:.1f} ({delta_unique:+.1f})")
            print(f"  goal: {s1['goal_rate']:.0%} → {s2['goal_rate']:.0%} ({delta_goal:+.0%})")

    # Step-level comparison for seed 0
    print("\n" + "=" * 50)
    print("Step-level divergence (seed 0)")
    print("=" * 50)
    ref_steps = load_steps("A", 0)
    if ref_steps:
        for cond in ["B", "C", "D", "E"]:
            cond_steps = load_steps(cond, 0)
            if cond_steps:
                n = min(len(ref_steps), len(cond_steps))
                diffs = 0
                first_diff = None
                for i in range(n):
                    a1 = ref_steps[i].get("action", ref_steps[i].get("action_label"))
                    a2 = cond_steps[i].get("action", cond_steps[i].get("action_label"))
                    if a1 != a2:
                        diffs += 1
                        if first_diff is None:
                            first_diff = i
                print(f"  A vs {cond}: {diffs}/{n} steps differ, first at step {first_diff}")


if __name__ == "__main__":
    main()
