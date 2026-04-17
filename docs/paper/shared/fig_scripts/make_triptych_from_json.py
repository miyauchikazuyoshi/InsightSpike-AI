#!/usr/bin/env python3
"""
Generate a triptych figure (s0, s, Δs) from an existing alignment JSON.

Usage:
  python make_triptych_from_json.py input.json output.pdf

The input JSON is expected to contain records with keys:
  - "s"   : float (cos(z_ins, z_ans))
  - "s0"  : float (cos(z0,   z_ans))
  - "delta_s": float (s - s0)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 3:
        print("Usage: python make_triptych_from_json.py input.json output.pdf")
        return 2

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    recs = data.get("records", [])
    if not recs:
        print("No records found in JSON")
        return 1

    s = np.array([float(r.get("s", 0.0)) for r in recs], dtype=float)
    s0 = np.array([float(r.get("s0", 0.0)) for r in recs], dtype=float)
    ds = np.array([float(r.get("delta_s", s[i]-s0[i])) for i, r in enumerate(recs)], dtype=float)

    rng = np.random.default_rng(0)
    nbs = 8000 if len(s) < 200 else 3000

    def ci95(x):
        bs = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(nbs)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    ci0 = ci95(s0)
    ci1 = ci95(s)
    ci2 = ci95(ds)
    pos_ratio = float((ds > 0).mean())

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    b0 = max(10, min(40, len(s0)//2))
    b1 = max(10, min(40, len(s)//2))
    b2 = max(10, min(40, len(ds)//2))

    # Panel 1: s0
    axes[0].hist(s0, bins=b0, color="#9ecae1", alpha=0.85)
    axes[0].axvline(s0.mean(), color="#08519c", linestyle=":", lw=1.2)
    axes[0].set_title(f"s0 (0-hop)\nmean={s0.mean():+.3f} 95%CI=[{ci0[0]:+.3f},{ci0[1]:+.3f}]")
    axes[0].set_xlabel("cos(z0, z_ans)")
    axes[0].set_ylabel("count")

    # Panel 2: s
    axes[1].hist(s, bins=b1, color="#3182bd", alpha=0.85)
    axes[1].axvline(s.mean(), color="#08306b", linestyle=":", lw=1.2)
    axes[1].set_title(f"s (H-hop)\nmean={s.mean():+.3f} 95%CI=[{ci1[0]:+.3f},{ci1[1]:+.3f}]")
    axes[1].set_xlabel("cos(z_ins, z_ans)")
    axes[1].set_ylabel("")

    # Panel 3: Δs
    axes[2].hist(ds, bins=b2, color="#4C78A8", alpha=0.85)
    axes[2].axvline(ds.mean(), color="red", linestyle="--", lw=1.2)
    axes[2].set_title(f"Δs (H-hop − 0-hop)\nmean={ds.mean():+.3f} 95%CI=[{ci2[0]:+.3f},{ci2[1]:+.3f}] pos={pos_ratio*100:.0f}%")
    axes[2].set_xlabel("Δs")
    axes[2].set_ylabel("")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"✅ Wrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())

