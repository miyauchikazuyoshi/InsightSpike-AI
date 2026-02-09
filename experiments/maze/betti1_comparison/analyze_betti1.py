"""Analyze β₁ dynamics and correlation with geDIG F-value from maze experiments."""
import json
import glob
import os
import sys
import numpy as np
from scipy import stats


def load_runs(results_dir: str) -> list[dict]:
    """Load all seed result files."""
    files = sorted(glob.glob(os.path.join(results_dir, "seed*.json")))
    runs = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        run = data.get("runs", [{}])[0]
        run["_file"] = os.path.basename(path)
        runs.append(run)
    return runs


def analyze_single_run(run: dict) -> dict:
    """Analyze a single seed's β₁ dynamics and F-value correlation."""
    b1 = np.array(run.get("betti1_series", []), dtype=float)
    nodes = np.array(run.get("node_count_series", []), dtype=float)
    edges = np.array(run.get("edge_count_series", []), dtype=float)
    g0 = np.array(run.get("g0_series", []), dtype=float)
    gmin = np.array(run.get("gmin_series", []), dtype=float)
    accepted = run.get("accepted_series", [])
    success = run.get("success", False)
    steps = run.get("steps", len(b1))

    result = {
        "file": run.get("_file", "?"),
        "success": success,
        "steps": steps,
        "final_V": int(nodes[-1]) if len(nodes) > 0 else 0,
        "final_E": int(edges[-1]) if len(edges) > 0 else 0,
        "final_b1": int(b1[-1]) if len(b1) > 0 else 0,
        "max_b1": int(b1.max()) if len(b1) > 0 else 0,
    }

    # Δβ₁ dynamics
    if len(b1) > 1:
        delta_b1 = np.diff(b1)
        result["delta_b1_nonzero"] = int(np.count_nonzero(delta_b1))
        result["delta_b1_pos"] = int(np.sum(delta_b1 > 0))   # cycle created
        result["delta_b1_neg"] = int(np.sum(delta_b1 < 0))   # cycle destroyed
    else:
        result["delta_b1_nonzero"] = 0
        result["delta_b1_pos"] = 0
        result["delta_b1_neg"] = 0

    # β₁ vs g0 (F-value at hop 0) correlation
    result["rho_b1_g0"] = None
    result["rho_b1_gmin"] = None
    if len(b1) > 1:
        if len(g0) >= len(b1) and np.std(b1) > 0 and np.std(g0[:len(b1)]) > 0:
            rho, _ = stats.spearmanr(b1, g0[:len(b1)])
            result["rho_b1_g0"] = round(float(rho), 4)
        if len(gmin) >= len(b1) and np.std(b1) > 0 and np.std(gmin[:len(b1)]) > 0:
            rho, _ = stats.spearmanr(b1, gmin[:len(b1)])
            result["rho_b1_gmin"] = round(float(rho), 4)

    # β₁ vs edge density (E/V)
    result["rho_b1_density"] = None
    if len(b1) > 1 and len(edges) >= len(b1) and len(nodes) >= len(b1):
        density = np.where(nodes[:len(b1)] > 0, edges[:len(b1)] / nodes[:len(b1)], 0)
        if np.std(b1) > 0 and np.std(density) > 0:
            rho, _ = stats.spearmanr(b1, density)
            result["rho_b1_density"] = round(float(rho), 4)

    # β₁ growth pattern: at which step fraction does β₁ first become > 0?
    if len(b1) > 0 and b1.max() > 0:
        first_nonzero = int(np.argmax(b1 > 0))
        result["b1_onset_pct"] = round(first_nonzero / len(b1) * 100, 1)
    else:
        result["b1_onset_pct"] = None

    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    runs = load_runs(results_dir)
    if not runs:
        print("ERROR: no result files found in", results_dir)
        sys.exit(1)

    print(f"=== β₁ Dynamics Analysis ({len(runs)} seeds, 25x25 maze) ===\n")

    results = []
    for run in runs:
        r = analyze_single_run(run)
        results.append(r)

    # Per-seed table
    hdr = f"{'seed':<16} {'ok':>2} {'stp':>4} {'V':>4} {'E':>4} {'β₁':>3} {'mx':>3} {'Δ≠0':>4} {'+':>2} {'-':>2} {'onset%':>6} {'ρ(g0)':>7} {'ρ(gm)':>7} {'ρ(d)':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ok = "Y" if r["success"] else "N"
        rg0 = f"{r['rho_b1_g0']:.3f}" if r["rho_b1_g0"] is not None else "  n/a"
        rgm = f"{r['rho_b1_gmin']:.3f}" if r["rho_b1_gmin"] is not None else "  n/a"
        rd = f"{r['rho_b1_density']:.3f}" if r["rho_b1_density"] is not None else "  n/a"
        onset = f"{r['b1_onset_pct']:.0f}%" if r["b1_onset_pct"] is not None else "  n/a"
        print(f"{r['file']:<16} {ok:>2} {r['steps']:>4} {r['final_V']:>4} {r['final_E']:>4} {r['final_b1']:>3} {r['max_b1']:>3} {r['delta_b1_nonzero']:>4} {r['delta_b1_pos']:>2} {r['delta_b1_neg']:>2} {onset:>6} {rg0:>7} {rgm:>7} {rd:>7}")

    # Aggregate
    print(f"\n=== Aggregate ===")
    print(f"Seeds: {len(results)}, Success: {sum(1 for r in results if r['success'])}/{len(results)}")

    final_b1s = [r["final_b1"] for r in results]
    print(f"Final β₁: mean={np.mean(final_b1s):.1f}, range=[{min(final_b1s)}, {max(final_b1s)}]")

    for label, key in [("ρ(β₁, g0)", "rho_b1_g0"), ("ρ(β₁, gmin)", "rho_b1_gmin"), ("ρ(β₁, density)", "rho_b1_density")]:
        vals = [r[key] for r in results if r[key] is not None]
        if vals:
            arr = np.array(vals)
            print(f"{label}: mean={arr.mean():.3f}, median={np.median(arr):.3f}, range=[{arr.min():.3f}, {arr.max():.3f}]")
        else:
            print(f"{label}: no valid data")

    # β₁ event summary
    total_pos = sum(r["delta_b1_pos"] for r in results)
    total_neg = sum(r["delta_b1_neg"] for r in results)
    total_steps = sum(r["steps"] for r in results)
    print(f"\nβ₁ events: +{total_pos} cycles created, -{total_neg} destroyed across {total_steps} total steps")
    print(f"  cycle creation rate: {total_pos/total_steps*100:.1f}% of steps")

    print("\n=== Interpretation ===")
    print("β₁ = independent cycles in the knowledge graph.")
    print("β₁ > 0 means the agent discovered alternative paths (redundancy).")
    print("ρ(β₁, g0) > 0: cycles correlate with higher F-value (structural surprise).")
    print("ρ(β₁, g0) < 0: cycles correlate with lower F-value (structural predictability).")


if __name__ == "__main__":
    main()
