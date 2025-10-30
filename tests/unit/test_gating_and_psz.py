from __future__ import annotations

from insightspike.algorithms.gating import decide_gates
from insightspike.metrics.psz import summarize_accept_latency


def test_decide_gates_basic():
    g0, gmin = 0.6, 0.2
    r = decide_gates(g0=g0, gmin=gmin, theta_ag=0.5, theta_dg=0.1)
    assert r.ag is True  # g0 > theta_ag
    assert r.dg is False  # min(g0,gmin)=0.2 > theta_dg
    assert abs(r.b_value - 0.2) < 1e-9


def test_psz_summary():
    s = summarize_accept_latency([
        {"accepted": True, "latency_ms": 120},
        {"accepted": True, "latency_ms": 180},
        {"accepted": False, "latency_ms": 210},
    ])
    assert 0.6 <= s.acceptance_rate <= 0.67
    assert s.fmr > 0
    assert 150 <= s.latency_p50_ms <= 200
    assert s.inside_psz is False

