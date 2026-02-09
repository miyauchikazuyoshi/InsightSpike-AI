"""Validate backward compatibility: baseline vs after-change legacy mode."""
import json
import sys


def compare(baseline_path, after_path):
    with open(baseline_path) as f:
        base = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    checks = []

    for seed_idx in range(len(base["runs"])):
        b = base["runs"][seed_idx]
        a = after["runs"][seed_idx]
        tag = f"seed{seed_idx}"

        checks.append((f"{tag}/success", b["success"] == a["success"]))
        checks.append((f"{tag}/steps", b["steps"] == a["steps"]))

        for key in [
            "g0_series",
            "gmin_series",
            "node_count_series",
            "edge_count_series",
            "betti1_series",
        ]:
            b_len = len(b.get(key, []))
            a_len = len(a.get(key, []))
            checks.append((f"{tag}/{key}_len", b_len == a_len))

        # g0 values match (float tolerance)
        g0_ok = True
        for i, (bg, ag) in enumerate(
            zip(b.get("g0_series", []), a.get("g0_series", []))
        ):
            if abs(bg - ag) > 1e-9:
                g0_ok = False
                break
        checks.append((f"{tag}/g0_values", g0_ok))

    passed = sum(1 for _, ok in checks if ok)
    failed = [(name, ok) for name, ok in checks if not ok]
    print(f"Compatibility: {passed}/{len(checks)} passed")
    if failed:
        print("FAILURES:")
        for name, _ in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("ALL PASSED")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
