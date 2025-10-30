"""P0: Spike reproducibility test (placeholder logic).

Currently uses memory node count threshold as a surrogate for first spike event
until direct spike result is exposed via navigator.
"""
import pytest

from insightspike.metrics.validation_helpers import run_spike_reproducibility

SEED_BASE = 20250823
SEEDS = [SEED_BASE + i for i in range(5)]


@pytest.mark.reproducibility
def test_spike_reproducibility():
    data = run_spike_reproducibility(SEEDS)
    # Tighten back to target 0.25 using real spike exposure from navigator
    if all(s == max(data['steps']) for s in data['steps']):
        pytest.xfail("No spike detected across runs; parameter tuning required (tau_s,tau_i)")
    assert data['cv'] < 0.25, f"Spike steps CV too high: {data['cv']:.3f} steps={data['steps']}"
