"""
Analyze A/B geDIG k-estimates aggregated over one or more CSV logs.

Expected CSV header fields (minimal subset used by tests):
- query_id, pure_gedig, full_gedig, pure_ged, full_ged, pure_ig, full_ig,
  k_estimate, k_missing_reason, window_corr_at_record, timestamp

Public function:
    compute_k_stats(paths: list[str]) -> dict
Returns:
    {
        'total_rows': int,
        'rows_with_k': int,
        'missing_rate': float,
        'k_min': float|None,
        'k_max': float|None,
        'window_corr_last': float|None,
    }
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional


def _safe_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() == 'none':
            return None
        return float(s)
    except Exception:
        return None


def compute_k_stats(paths: Iterable[str]) -> dict:
    total = 0
    with_k = 0
    k_min: Optional[float] = None
    k_max: Optional[float] = None
    last_corr: Optional[float] = None

    for p in paths:
        pth = Path(p)
        if not pth.exists():
            continue
        with pth.open(newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                k_val = _safe_float(row.get('k_estimate'))
                if k_val is not None:
                    with_k += 1
                    k_min = k_val if k_min is None else min(k_min, k_val)
                    k_max = k_val if k_max is None else max(k_max, k_val)
                last_corr = _safe_float(row.get('window_corr_at_record')) or last_corr

    missing_rate = (total - with_k) / total if total > 0 else 0.0
    return {
        'total_rows': total,
        'rows_with_k': with_k,
        'missing_rate': missing_rate,
        'k_min': k_min,
        'k_max': k_max,
        'window_corr_last': last_corr,
    }


__all__ = ['compute_k_stats']

