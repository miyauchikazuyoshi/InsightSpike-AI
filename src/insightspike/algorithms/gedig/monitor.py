"""Runtime monitoring for geDIG spike predictions.

This module provides monitoring and auto-tuning capabilities for spike detection.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .types import GeDIGResult


class GeDIGMonitor:
    """Runtime monitoring for spike predictions.

    Features (extended):
      - Rolling spike rate
      - False positive rate tracking (when ground-truth provided)
      - Simple auto-threshold adjustment to keep FP rate under target
      - Ground-truth spike auto-derivation (structural_improvement & ig_z_score)
      - Exportable metrics snapshot (JSON / CSV)
      - Tau (tau_s, tau_i) adjustment history
    """

    def __init__(
        self,
        window_size: int = 200,
        target_fp_rate: float = 0.1,
        adjust_factor: float = 1.1,
        gt_si_threshold: Optional[float] = None,
        gt_igz_threshold: Optional[float] = None,
        gt_mode: str = 'and',
    ) -> None:
        self.pred_buffer: deque = deque(maxlen=window_size)
        self.fp_buffer: deque = deque(maxlen=window_size)
        self.actual_buffer: deque = deque(maxlen=window_size)
        self.target_fp_rate = target_fp_rate
        self.adjust_factor = adjust_factor
        self.gt_si_threshold = gt_si_threshold
        self.gt_igz_threshold = gt_igz_threshold
        self.gt_mode = gt_mode.lower()
        self.tau_history: List[Dict[str, float]] = []
        # Spike が全く検出されない期間が続く場合に tau を積極的に緩和するためのカウンタ
        self.zero_spike_backoff_count: int = 0

    def record_prediction(self, predicted_spike: bool) -> None:
        """Record a spike prediction."""
        self.pred_buffer.append(1 if predicted_spike else 0)

    def record_outcome(self, actual_spike: bool) -> None:
        """Record actual spike outcome for FP rate calculation."""
        if not self.pred_buffer:
            return
        predicted = bool(self.pred_buffer[-1])
        is_fp = 1 if (predicted and not actual_spike) else 0
        self.fp_buffer.append(is_fp)
        self.actual_buffer.append(1 if actual_spike else 0)

    def derive_ground_truth(self, result: 'GeDIGResult', core: Any) -> bool:
        """Derive ground truth spike label from result."""
        if self.gt_mode == 'threshold':
            return bool(result.has_spike)
        si_thr = self.gt_si_threshold if self.gt_si_threshold is not None else getattr(core, 'tau_s', 0.0)
        ig_thr = self.gt_igz_threshold if self.gt_igz_threshold is not None else getattr(core, 'tau_i', 0.0)
        cond_si = result.structural_improvement > si_thr
        cond_ig = result.ig_z_score > ig_thr
        if self.gt_mode == 'or':
            return cond_si or cond_ig
        return cond_si and cond_ig

    def record_auto_outcome(self, result: 'GeDIGResult', core: Any) -> bool:
        """Auto-derive and record outcome."""
        label = self.derive_ground_truth(result, core)
        self.record_outcome(label)
        return label

    def spike_rate(self) -> float:
        """Calculate rolling spike rate."""
        if not self.pred_buffer:
            return 0.0
        return sum(self.pred_buffer) / len(self.pred_buffer)

    def false_positive_rate(self) -> float:
        """Calculate rolling false positive rate."""
        if not self.fp_buffer:
            return 0.0
        return sum(self.fp_buffer) / len(self.fp_buffer)

    def auto_adjust_thresholds(self, core: Any) -> None:
        """Auto-adjust tau_s and tau_i based on FP rate."""
        if len(self.fp_buffer) < 10:
            return
        fp = self.false_positive_rate()
        sp_rate = self.spike_rate()

        # 誤検出多い → 閾値強化
        if fp > self.target_fp_rate * 1.1:
            core.tau_s *= self.adjust_factor
            core.tau_i *= self.adjust_factor
        # 誤検出少ない & spike もほぼ出ていない → 閾値緩和
        elif fp < self.target_fp_rate * 0.5 and sp_rate < 0.05:
            core.tau_s /= self.adjust_factor
            core.tau_i /= self.adjust_factor

        # 全く spike が無い期間がウィンドウ満杯で継続 → 一段強い緩和 (二乗)
        if sp_rate == 0.0 and len(self.pred_buffer) >= self.pred_buffer.maxlen:
            core.tau_s /= (self.adjust_factor ** 2)
            core.tau_i /= (self.adjust_factor ** 2)
            self.zero_spike_backoff_count += 1
            if self.zero_spike_backoff_count >= 2:
                try:
                    core.spike_detection_mode = 'or'
                except Exception:
                    pass

        core.tau_s = float(np.clip(core.tau_s, 1e-4, 10.0))
        core.tau_i = float(np.clip(core.tau_i, 1e-4, 10.0))
        self.tau_history.append({
            'n_samples': float(len(self.fp_buffer)),
            'tau_s': core.tau_s,
            'tau_i': core.tau_i,
        })

    def get_metrics(self) -> Dict[str, Union[float, int]]:
        """Get current metrics snapshot."""
        return {
            'spike_rate': self.spike_rate(),
            'false_positive_rate': self.false_positive_rate(),
            'n_predictions': len(self.pred_buffer),
            'n_actual': len(self.actual_buffer),
            'zero_spike_backoff_count': self.zero_spike_backoff_count,
        }

    def export_metrics(
        self,
        path: str,
        core: Any,
        include_history: bool = True,
    ) -> None:
        """Export metrics to JSON or CSV file."""
        metrics = self.get_metrics()
        metrics.update({
            'tau_s': core.tau_s,
            'tau_i': core.tau_i,
            'lambda_weight': getattr(core, 'lambda_weight', 0.0),
            'mu': getattr(core, 'mu', 0.0),
        })

        if path.endswith('.json'):
            out: Dict[str, Any] = {'metrics': metrics}
            if include_history:
                out['tau_history'] = self.tau_history
            with open(path, 'w') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        else:
            fieldnames = sorted(metrics.keys())
            first = not os.path.exists(path)
            with open(path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if first:
                    w.writeheader()
                w.writerow(metrics)
            if include_history and self.tau_history:
                hist_path = path + '.tau_history.json'
                with open(hist_path, 'w') as fh:
                    json.dump(self.tau_history, fh, ensure_ascii=False, indent=2)

    def summarize_hop_results(self, result: 'GeDIGResult') -> Dict[str, float]:
        """Lightweight stats for logging hop results."""
        if not getattr(result, 'hop_results', None):
            return {}
        vals = [hr.gedig for hr in result.hop_results.values()]
        if not vals:
            return {}
        mean_v = statistics.fmean(vals)
        p95 = sorted(vals)[int(len(vals) * 0.95) - 1] if len(vals) >= 2 else vals[0]
        return {
            'hop_gedig_mean': float(mean_v),
            'hop_gedig_p95': float(p95),
            'hop_gedig_max': float(max(vals)),
            'hop_count': float(len(vals)),
        }


__all__ = ["GeDIGMonitor"]
