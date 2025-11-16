"""Export per-query PSZ scatter points from Exp23 lite results.

This bridges `exp2to4_lite` outputs and the paper's
`fig_rag_psz_scatter.py`, which expects a CSV at:

  data/rag_eval/psz_points.csv

with columns:
  run_id, query_id, H, k, PER, acceptance, FMR, latency_ms

We derive these from the latest `exp23_paper_*.json` result under
`experiments/exp2to4_lite/results`, using per-query samples for each
baseline (static/frequency/cosine/geDIG-lite).
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any, Dict, List, Tuple


def _latest_exp23_paper(results_dir: Path) -> Path | None:
    """Return the latest exp23_paper_*.json path with `results`, if any."""
    candidates = sorted(results_dir.glob("exp23_paper_*.json"))
    for path in reversed(candidates):
        try:
            js = _load_json(path)
        except Exception:
            continue
        if isinstance(js, dict) and "results" in js:
            return path
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _friendly_run_id(cfg: Dict[str, Any], baseline_name: str) -> str:
    name = str(cfg.get("name", "exp23"))
    mapping = {
        "static_rag": "Static",
        "frequency": "Frequency",
        "cosine_topk": "Cosine",
        "gedig_ag_dg": "geDIG-lite",
    }
    method = mapping.get(baseline_name, baseline_name)
    return f"{name}_{method}"


def _hk_for_baseline(cfg: Dict[str, Any], baseline_name: str) -> Tuple[int, int]:
    """Return (H, k) for a given baseline under exp2to4_lite config."""
    # Default: use retrieval.top_k from config; H depends on whether geDIG is active.
    k = int(cfg.get("retrieval", {}).get("top_k", 4))
    if baseline_name == "gedig_ag_dg":
        H = int(cfg.get("gedig", {}).get("max_hops", 3))
    else:
        # Non-graph baselines effectively operate at H=0 (flat)
        H = 0
    return H, k


def export_psz_points(results_dir: Path, out_path: Path) -> None:
    result_path = _latest_exp23_paper(results_dir)
    if result_path is None:
        raise SystemExit(f"No exp23_paper_*.json found under {results_dir}")

    data = _load_json(result_path)
    cfg = data.get("config", {})
    results = data.get("results", {})

    rows: List[List[str]] = []
    # Iterate over baselines that have per_samples
    for baseline_name, payload in results.items():
        per_samples = payload.get("per_samples") or []
        if not per_samples:
            continue
        run_id = _friendly_run_id(cfg, baseline_name)
        H, k = _hk_for_baseline(cfg, baseline_name)
        for qid, sample in enumerate(per_samples):
            try:
                per = float(sample.get("per", 0.0))
            except Exception:
                per = 0.0
            accepted = bool(sample.get("accepted", False))
            # Instance-level acceptance/FMR: 1 for accepted/correct, 0 otherwise
            acc = 1.0 if accepted else 0.0
            fmr = 1.0 - acc  # treat non-accepted as "failure" for coloring
            try:
                latency = float(sample.get("latency_ms", 0.0))
            except Exception:
                latency = 0.0
            rows.append(
                [
                    run_id,
                    str(qid),
                    str(H),
                    str(k),
                    f"{per:.4f}",
                    f"{acc:.4f}",
                    f"{fmr:.4f}",
                    f"{latency:.2f}",
                ]
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "query_id", "H", "k", "PER", "acceptance", "FMR", "latency_ms"])
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    results_dir = repo_root / "experiments" / "exp2to4_lite" / "results"
    out_path = repo_root / "data" / "rag_eval" / "psz_points.csv"
    export_psz_points(results_dir, out_path)
    print(f"✅ Wrote {out_path}")


if __name__ == "__main__":
    main()
