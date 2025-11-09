"""Metrics and PSZ computation (self-contained)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


def _token_overlap(a: str, b: str) -> float:
    import re

    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def compute_per(answer: str, ground_truth: str) -> float:
    return _token_overlap(answer, ground_truth)


def compute_acceptance(answer: str, ground_truth: str, threshold: float) -> bool:
    return compute_per(answer, ground_truth) >= threshold


def simulate_latency_ms(steps: int) -> float:
    base = 120.0
    return base + steps * 40.0


@dataclass
class MetricsSummary:
    per_scores: List[float] = field(default_factory=list)
    acceptances: List[bool] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)

    def add(self, per: float, accepted: bool, latency: float) -> None:
        self.per_scores.append(per)
        self.acceptances.append(accepted)
        self.latencies.append(latency)

    def to_dict(self) -> Dict[str, float]:
        if not self.per_scores:
            return {
                "per_mean": 0.0,
                "per_std": 0.0,
                "acceptance_rate": 0.0,
                "fmr": 1.0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
            }
        per_arr = np.asarray(self.per_scores)
        acc_arr = np.asarray(self.acceptances, dtype=float)
        lat_arr = np.asarray(self.latencies)
        fmr = 1.0 - np.mean(acc_arr)
        return {
            "per_mean": float(np.mean(per_arr)),
            "per_std": float(np.std(per_arr)),
            "acceptance_rate": float(np.mean(acc_arr)),
            "fmr": float(fmr),
            "latency_p50": float(np.percentile(lat_arr, 50)),
            "latency_p95": float(np.percentile(lat_arr, 95)),
        }

    def inside_psz(self, acceptance_threshold: float, fmr_threshold: float, latency_p50_threshold: float) -> bool:
        stats = self.to_dict()
        return (
            stats["acceptance_rate"] >= acceptance_threshold
            and stats["fmr"] <= fmr_threshold
            and stats["latency_p50"] <= latency_p50_threshold
        )

