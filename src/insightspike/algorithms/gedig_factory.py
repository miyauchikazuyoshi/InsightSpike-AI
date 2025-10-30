"""GeDIG feature flag factory and dual evaluation utilities (Phase A6).

Currently legacy and refactored both point to `GeDIGCore` but hooks are provided
for future divergence. The `dual_evaluate` helper runs both implementations
and reports divergence magnitude.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import logging

from .gedig_core import GeDIGCore

logger = logging.getLogger(__name__)

class GeDIGFactory:
    LEGACY_PRESET = {
        # Emulate older behavior: product-like reward, threshold spike mode
        'use_refactored_reward': False,
        'use_legacy_formula': True,
        'spike_detection_mode': 'threshold',
        'tau_s': 0.2,  # not used in threshold mode but keep defaults
        'tau_i': 0.3,
    }

    @staticmethod
    def create(config: Optional[Dict[str, Any]] = None, **overrides) -> GeDIGCore:
        cfg = config or {}
        use_ref = cfg.get('use_refactored_gedig', True)
        params = {**overrides}
        if use_ref:
            return GeDIGCore(**params)
        legacy_params = {k: v for k, v in GeDIGFactory.LEGACY_PRESET.items() if k not in params}
        legacy_params.update(params)
        return GeDIGCore(**legacy_params)


def dual_evaluate(legacy_core: GeDIGCore,
                  ref_core: GeDIGCore,
                  *,
                  g_prev: Any,
                  g_now: Any,
                  delta_threshold: float = 0.3,
                  divergence_logger: Optional[str] = None) -> Tuple[Any, float]:
    """Run both cores and emit warning if divergence exceeds threshold.

    If divergence_logger path provided, append CSV rows: step(auto),legacy,ref,delta,timestamp.
    Returns: (ref_result, delta)
    """
    import time, os, csv
    old_res = legacy_core.calculate(g_prev=g_prev, g_now=g_now)
    new_res = ref_core.calculate(g_prev=g_prev, g_now=g_now)
    delta = abs(old_res.gedig_value - new_res.gedig_value)
    if delta > delta_threshold:
        logger.warning(
            "GeDIG divergence %.4f (legacy=%.4f, ref=%.4f)",
            delta, old_res.gedig_value, new_res.gedig_value
        )
    if divergence_logger:
        # Minimal rotating append (no size mgmt here)
        header = ['ts','legacy','ref','delta']
        exists = os.path.exists(divergence_logger)
        with open(divergence_logger, 'a', newline='') as fh:
            w = csv.writer(fh)
            if not exists:
                w.writerow(header)
            w.writerow([f"{time.time():.3f}", f"{old_res.gedig_value:.6f}", f"{new_res.gedig_value:.6f}", f"{delta:.6f}"])
    return new_res, delta
