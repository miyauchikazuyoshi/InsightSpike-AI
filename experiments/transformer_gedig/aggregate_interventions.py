#!/usr/bin/env python3
"""
Summarize intervene_sketch.json: per-intervention F_mean, SST-2 accuracy, and correlation stats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def summarize(path: Path) -> Dict:
    data = json.loads(path.read_text())
    out: Dict[str, float] = {}
    interventions = data.get("interventions", [])
    if interventions:
        out.update({f"F_mean_{r['intervention']}": r["F_mean_last_layer"] for r in interventions})
    sst = data.get("sst2_eval", {})
    if sst:
        out["sst2_accuracy_baseline"] = sst.get("accuracy")
        out["sst2_conf_vs_F_corr_baseline"] = sst.get("conf_vs_F_corr")
    sst_int = data.get("sst2_interventions", {})
    for name, res in sst_int.items():
        out[f"sst2_accuracy_{name}"] = res.get("accuracy")
        out[f"sst2_conf_vs_F_corr_{name}"] = res.get("conf_vs_F_corr")
        f_vals = [s.get("F_mean_last_layer") for s in res.get("samples", []) if "F_mean_last_layer" in s]
        if f_vals:
            out[f"F_mean_{name}"] = float(np.mean(f_vals))
    return out


def main() -> None:
    src = Path("results/transformer_gedig/intervene_sketch.json")
    summary = summarize(src)
    out_path = Path("results/transformer_gedig/intervene_summary.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
