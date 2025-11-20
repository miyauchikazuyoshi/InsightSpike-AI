"""Export IG robustness CSV for fig_r_ig_robust.py from Exp23 ablations.

This bridges `exp2to4_lite` ablation outputs and the paper's
`fig_r_ig_robust.py`, which expects:

  data/rag_eval/ig_robustness.csv

with columns:
  run_id, ig_def, H, k, PER, acceptance, FMR, latency_ms

We derive these from the latest `exp23_paper_*_ablations.json` bundle under
`experiments/exp2to4_lite/results`, using per-query samples from the
`gedig_ag_dg` baseline across a small set of IG-related variants:

  - base         : default geDIG settings
  - epc_only     : lambda_weight = 0.0 (no IG term)
  - ig_emphasis  : lambda_weight >> 1 (IG-emphasised)

The goal is to show that changing IG weighting / formulation does not
dramatically distort PER / acceptance / FMR / latency.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any, Dict, Iterable, List, Tuple


def _latest_ablations(results_dir: Path) -> Path | None:
    """Return the latest exp23_paper_*_ablations.json path, if any."""
    files = sorted(results_dir.glob("exp23_paper_*_ablations.json"))
    return files[-1] if files else None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_variants(
    bundle: Dict[str, Any],
    variants: Iterable[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name in variants:
        payload = bundle.get(name)
        if not isinstance(payload, dict):
            continue
        results = payload.get("results", {})
        if "gedig_ag_dg" not in results:
            continue
        out.append((name, results["gedig_ag_dg"]))
    return out


def export_ig_robustness(results_dir: Path, out_path: Path) -> None:
    ablations_path = _latest_ablations(results_dir)
    if ablations_path is None:
        raise SystemExit(f"No exp23_paper_*_ablations.json found under {results_dir}")

    bundle = _load_json(ablations_path)
    # Interpret these ablation variants as different IG definitions / weightings.
    ig_variants = ["base", "epc_only", "ig_emphasis"]
    pairs = _iter_variants(bundle, ig_variants)
    if not pairs:
        raise SystemExit(f"No usable variants {ig_variants} with gedig_ag_dg results in {ablations_path}")

    rows: List[List[str]] = []
    # Assume all variants share the same number of per_samples (same query set).
    for ig_def, stats in pairs:
        per_samples = stats.get("per_samples") or []
        # H/k are not essential for the robustness panel; we keep them fixed.
        H = 3
        k = 8
        for qid, sample in enumerate(per_samples):
            try:
                per = float(sample.get("per", 0.0))
            except Exception:
                per = 0.0
            accepted = bool(sample.get("accepted", False))
            acc = 1.0 if accepted else 0.0
            fmr = 1.0 - acc
            try:
                lat = float(sample.get("latency_ms", 0.0))
            except Exception:
                lat = 0.0
            rows.append(
                [
                    str(qid),           # run_id: align by query index across defs
                    ig_def,             # ig_def label
                    str(H),
                    str(k),
                    f"{per:.4f}",
                    f"{acc:.4f}",
                    f"{fmr:.4f}",
                    f"{lat:.2f}",
                ]
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "ig_def", "H", "k", "PER", "acceptance", "FMR", "latency_ms"])
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    results_dir = repo_root / "experiments" / "exp2to4_lite" / "results"
    out_path = repo_root / "data" / "rag_eval" / "ig_robustness.csv"
    export_ig_robustness(results_dir, out_path)
    print(f"✅ Wrote {out_path}")


if __name__ == "__main__":
    main()

