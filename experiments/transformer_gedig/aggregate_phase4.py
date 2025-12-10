#!/usr/bin/env python3
"""
Aggregate geDIG F vs checkpoint accuracy for Phase4 PoC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def main() -> None:
    # load geDIG scores
    scores = json.loads(Path("results/transformer_gedig/score_smoke.json").read_text())
    df = pd.DataFrame(scores)
    f_means = df.groupby("model")["F"].mean().to_dict()

    # load checkpoint summary
    summary_path = Path("results/transformer_gedig/checkpoints/distilbert_sst2_poc/run_summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    ckpt_acc = summary.get("checkpoints", {})

    out: Dict[str, float] = {}
    out["F_mean_overall"] = float(np.mean(df["F"]))
    for m, v in f_means.items():
        out[f"F_mean_{m}"] = float(v)
    for ckpt, acc in ckpt_acc.items():
        out[f"accuracy_{ckpt}"] = acc

    Path("results/transformer_gedig/phase4_poc_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
