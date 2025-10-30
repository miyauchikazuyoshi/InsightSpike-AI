"""P0: Small maze stability test for GeDIG refactor.

Validates coefficient of variation (CV) of pseudo reward proxy after warmup.
NOTE: Uses pseudo reward until navigator exposes direct GeDIG reward.
"""
import pytest

from insightspike.metrics.validation_helpers import run_small_maze_stability


@pytest.mark.performance
def test_small_maze_stability():
    res = run_small_maze_stability()
    if not (res.stable_via_primary or res.stable_via_sparse_fallback):
        raise AssertionError(
            f"StabilityFail: cv={res.cv:.3f} mean={res.mean:.4g} std={res.std:.4g} "
            f"n={len(res.rewards)} nonzero_frac={res.nonzero_fraction:.2%} "
            f"outlier_rate={res.outlier_rate:.2%}"
        )
    assert res.outlier_rate < 0.1, f"Outlier rate high: {res.outlier_rate:.2%}"
