"""Spike detection for geDIG.

This module provides spike detection logic extracted from GeDIGCore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import SpikeDetectionMode, GeDIGResult

if TYPE_CHECKING:
    pass


def detect_spike(
    result: GeDIGResult,
    mode: str | SpikeDetectionMode,
    spike_threshold: float,
    tau_s: float,
    tau_i: float,
    ig_variance: float = 0.0,
) -> bool:
    """Detect if a geDIG result represents an insight spike.

    Args:
        result: The GeDIGResult to evaluate.
        mode: Detection mode ('threshold', 'and', 'or').
        spike_threshold: Threshold for threshold mode.
        tau_s: Structural improvement threshold.
        tau_i: Information gain z-score threshold.
        ig_variance: Current IG variance for fallback logic.

    Returns:
        True if a spike is detected, False otherwise.
    """
    # Normalize mode to enum
    if isinstance(mode, str):
        mode = SpikeDetectionMode(mode.lower())

    structural_signal = float(
        getattr(result, 'structural_improvement', -result.delta_ged_norm)
    )

    # Threshold mode: simple comparison
    if mode == SpikeDetectionMode.THRESHOLD:
        return bool(result.gedig_value < spike_threshold)

    # AND mode: both conditions must be met
    if mode == SpikeDetectionMode.AND:
        if (structural_signal > tau_s) and (result.ig_z_score > tau_i):
            return True
        # Fallback: if IG variance is negligible, use structural signal only
        if ig_variance < 1e-9 and structural_signal > (tau_s * 2):
            return True
        return False

    # OR mode: either condition is sufficient
    if mode == SpikeDetectionMode.OR:
        if (structural_signal > tau_s) or (result.ig_z_score > tau_i):
            return True
        # Backward compatibility: any positive signal is a spike
        if (structural_signal > 0) or (result.ig_z_score > 0):
            return True
        return False

    # Fallback: natural spike induction when thresholds are at floor
    if tau_s <= 1e-4 and tau_i <= 1e-4 and ig_variance < 1e-9 and structural_signal > 0.0:
        return True

    return bool(result.gedig_value < spike_threshold)


def compute_rewards(
    result: GeDIGResult,
    lambda_weight: float,
    mu: float,
    decay_factor: float,
    warmup_steps: int,
    ig_count: int,
) -> None:
    """Compute reward values and update the result in-place.

    Args:
        result: GeDIGResult to update.
        lambda_weight: Weight for IG component.
        mu: Weight for structural component.
        decay_factor: Decay factor for multi-hop aggregation.
        warmup_steps: Number of warmup steps before using lambda.
        ig_count: Current IG sample count.
    """
    # During warmup, don't use lambda
    effective_lambda = 0.0 if ig_count <= warmup_steps else lambda_weight

    structural_signal = -result.delta_ged_norm
    result.hop0_reward = effective_lambda * result.ig_z_score + mu * structural_signal

    if result.hop_results:
        total_si = 0.0
        total_w = 0.0
        for hop, hr in result.hop_results.items():
            w = decay_factor ** hop
            total_si += w * (-hr.ged)
            total_w += w
        avg_si = (total_si / total_w) if total_w > 0 else structural_signal
        result.aggregate_reward = effective_lambda * result.ig_z_score + mu * avg_si
    else:
        result.aggregate_reward = result.hop0_reward


__all__ = [
    "detect_spike",
    "compute_rewards",
]
