#!/usr/bin/env python3
"""
Lightweight self-test for GeDIG A/B logger.

Cloud-safe: no network, no heavy deps, writes only under results/.
Used by Makefile target `selftest-ab` and optional in scripts/codex_smoke.sh.
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    # Ensure repo-relative output dirs (cloud-safe)
    out_dir = Path(os.getenv("INSIGHTSPIKE_LOG_DIR", "results/logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gedig_ab_logger_selftest.csv"

    # Import lazily to keep imports minimal
    from insightspike.algorithms.gedig_ab_logger import GeDIGABLogger

    logger = GeDIGABLogger(window=10, threshold=0.5, min_pairs=5, flush_every=3)
    logger.set_auto_csv_path(str(csv_path))

    # Record a few synthetic pairs; keep values simple and deterministic
    pairs = [
        ("q1", {"gedig": 0.10, "ged": 0.20, "ig": 0.15}, {"gedig": 0.12, "ged": 0.22, "ig": 0.16}),
        ("q2", {"gedig": 0.05, "ged": 0.18, "ig": 0.14}, {"gedig": 0.06, "ged": 0.19, "ig": 0.13}),
        ("q3", {"gedig": -0.02, "ged": 0.12, "ig": 0.20}, {"gedig": -0.01, "ged": 0.13, "ig": 0.19}),
    ]

    for qid, pure, full in pairs:
        logger.record(qid, pure, full)

    # Export explicitly to ensure a file is written even if auto-flush hasn't fired
    _ = logger.export_csv(str(csv_path))

    # Minimal success line for scripts/tests to detect
    print("[selftest-ab] OK", csv_path)


if __name__ == "__main__":
    main()

