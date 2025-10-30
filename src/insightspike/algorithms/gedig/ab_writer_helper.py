from __future__ import annotations

import os
import time
from typing import IO, Tuple

import csv


def create_csv_writer(path: str | None = None) -> Tuple[csv.writer, IO[str]]:
    """Create a CSV writer for A/B logs.

    If `path` is None, use `INSIGHTSPIKE_AB_LOG_DIR` (default 'data/metrics') and
    generate a timestamped filename (gedig_ab_log_YYYYMMDD_HHMMSS.csv).

    Returns (writer, file_handle). Caller is responsible for closing the handle.
    """
    if path is None:
        base_dir = os.getenv("INSIGHTSPIKE_AB_LOG_DIR", "data/metrics")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(base_dir, f"gedig_ab_log_{ts}.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    return w, f


__all__ = ["create_csv_writer"]

