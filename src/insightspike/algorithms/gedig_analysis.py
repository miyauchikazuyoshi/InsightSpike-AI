"""Utilities for analyzing GeDIG dual evaluation divergence logs (Day3).

CSV format expected: ts,legacy,ref,delta (header from dual_evaluate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class DivergenceStats:
    count: int
    mean_delta: float
    max_delta: float
    mean_legacy: float
    mean_ref: float
    pct_over_threshold: float
    threshold: float

    def to_dict(self) -> Dict[str, float]:  # convenience
        return {
            'count': self.count,
            'mean_delta': self.mean_delta,
            'max_delta': self.max_delta,
            'mean_legacy': self.mean_legacy,
            'mean_ref': self.mean_ref,
            'pct_over_threshold': self.pct_over_threshold,
            'threshold': self.threshold,
        }


def analyze_divergence(csv_path: str, threshold: float = 0.3) -> DivergenceStats:
    """Analyze a divergence CSV produced by dual_evaluate.

    Parameters
    ----------
    csv_path: str
        Path to divergence log.
    threshold: float
        Threshold for counting exceedances (consistency with dual_delta_threshold).
    """
    import csv
    rows: List[Dict[str, str]] = []
    with open(csv_path) as fh:
        r = csv.DictReader(fh)
        required = {'legacy','ref','delta'}
        if not required.issubset(r.fieldnames or []):  # type: ignore[arg-type]
            raise ValueError(f"Missing required columns in divergence csv: {r.fieldnames}")
        for row in r:
            rows.append(row)
    if not rows:
        return DivergenceStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, threshold)
    deltas = [float(r['delta']) for r in rows]
    legacy_vals = [float(r['legacy']) for r in rows]
    ref_vals = [float(r['ref']) for r in rows]
    count = len(deltas)
    mean_delta = sum(deltas)/count
    max_delta = max(deltas)
    mean_legacy = sum(legacy_vals)/count
    mean_ref = sum(ref_vals)/count
    over = sum(1 for d in deltas if d > threshold)
    pct_over = over / count if count else 0.0
    return DivergenceStats(count, mean_delta, max_delta, mean_legacy, mean_ref, pct_over, threshold)


__all__ = ["analyze_divergence", "DivergenceStats"]