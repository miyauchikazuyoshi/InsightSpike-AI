#!/usr/bin/env python3
"""Split a JSONL dataset into train/val/test by ratios or counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import List


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-train", type=Path, required=True)
    ap.add_argument("--out-val", type=Path, required=True)
    ap.add_argument("--out-test", type=Path, required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(sum(args.ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    random.seed(args.seed)

    rows: List[str] = [ln for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * args.ratios[0])
    n_val = int(n * args.ratios[1])
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    args.out_train.write_text("\n".join(train) + "\n", encoding="utf-8")
    args.out_val.write_text("\n".join(val) + "\n", encoding="utf-8")
    args.out_test.write_text("\n".join(test) + "\n", encoding="utf-8")

    print(f"Split {n} -> train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()

