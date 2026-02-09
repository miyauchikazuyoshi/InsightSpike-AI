"""Validate β₁ data in maze experiment output JSON."""
import json
import glob
import sys
import os

def validate_file(path: str) -> list[str]:
    """Validate a single JSON output file. Returns list of errors."""
    errors = []
    with open(path) as f:
        data = json.load(f)

    # Data lives in runs[0] (per-seed run result)
    runs = data.get("runs", [])
    if not runs:
        errors.append("no 'runs' array in output")
        return errors
    summary = runs[0]

    # Check betti1_series exists
    b1 = summary.get("betti1_series")
    if b1 is None:
        errors.append("betti1_series missing from output")
        return errors

    if len(b1) == 0:
        errors.append("betti1_series is empty")
        return errors

    # Check node/edge count series exist
    nodes = summary.get("node_count_series")
    edges = summary.get("edge_count_series")
    if nodes is None:
        errors.append("node_count_series missing")
    if edges is None:
        errors.append("edge_count_series missing")

    if nodes is None or edges is None:
        return errors

    # Check lengths match
    if len(b1) != len(nodes) or len(b1) != len(edges):
        errors.append(
            f"series length mismatch: betti1={len(b1)}, nodes={len(nodes)}, edges={len(edges)}"
        )

    # Check β₁ = E - V + 1 consistency (for steps where V > 0)
    inconsistent = 0
    for i, (v, e, beta) in enumerate(zip(nodes, edges, b1)):
        if v > 0:
            expected = e - v + 1
            if beta != expected:
                inconsistent += 1
                if inconsistent <= 3:
                    errors.append(
                        f"step {i}: β₁={beta} != E-V+1={expected} (V={v}, E={e})"
                    )
    if inconsistent > 3:
        errors.append(f"... and {inconsistent - 3} more inconsistencies")

    # Check β₁ >= 0
    negatives = [i for i, b in enumerate(b1) if b < 0]
    if negatives:
        errors.append(f"negative β₁ at steps: {negatives[:5]}")

    # Check that β₁ is not all zero (some graph growth expected)
    non_zero = [b for b in b1 if b != 0]
    all_zero_nodes = all(v == 0 for v in nodes)

    return errors


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    files = sorted(glob.glob(os.path.join(results_dir, "betti1_smoke_seed*.json")))

    if not files:
        print("ERROR: no result files found in", results_dir)
        sys.exit(1)

    print(f"Validating {len(files)} result file(s)...\n")

    all_ok = True
    all_b1_series = []

    for path in files:
        fname = os.path.basename(path)
        errors = validate_file(path)

        # Also collect stats for summary
        with open(path) as f:
            data = json.load(f)
        run = data.get("runs", [{}])[0]

        b1 = run.get("betti1_series", [])
        nodes = run.get("node_count_series", [])
        edges = run.get("edge_count_series", [])
        success = run.get("success", "?")
        steps = run.get("steps", len(b1))

        if errors:
            print(f"FAIL  {fname}")
            for e in errors:
                print(f"      {e}")
            all_ok = False
        else:
            print(f"OK    {fname}")

        # Print stats
        non_zero_b1 = [b for b in b1 if b != 0]
        max_b1 = max(b1) if b1 else 0
        final_v = nodes[-1] if nodes else 0
        final_e = edges[-1] if edges else 0
        print(f"      success={success}, steps={steps}")
        print(f"      final: V={final_v}, E={final_e}, β₁={b1[-1] if b1 else '?'}")
        print(f"      β₁ range: [0, {max_b1}], non-zero steps: {len(non_zero_b1)}/{len(b1)}")
        print()

        all_b1_series.append(b1)

    # Cross-seed summary
    if len(all_b1_series) > 1:
        print("--- Cross-seed summary ---")
        for i, b1 in enumerate(all_b1_series):
            max_b = max(b1) if b1 else 0
            final_b = b1[-1] if b1 else 0
            print(f"  seed {i}: final β₁={final_b}, max β₁={max_b}, steps={len(b1)}")
        print()

    if all_ok:
        print("ALL PASSED")
        sys.exit(0)
    else:
        print("SOME FAILURES")
        sys.exit(1)


if __name__ == "__main__":
    main()
