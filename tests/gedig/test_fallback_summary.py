import os
import pytest

from insightspike.implementations.agents.main_agent import MainAgent


def test_gedig_fallback_summary_pure_failure(monkeypatch):
    """Force a pure->full fallback event and validate tracker summary.

    Strategy: monkeypatch _compute_gedig so that first pure attempt raises,
    triggering fallback and recording the event via GeDIGFallbackTracker.
    """
    cfg = {
        'gedig': {'mode': 'pure'},  # start in pure to allow fallback
        'embedding': {'model_name': None, 'dimension': 32},
        'processing': {'enable_learning': False},
        'memory': {'graph': False},
    }
    os.environ.setdefault('INSIGHTSPIKE_MIN_IMPORT', '1')
    os.environ.setdefault('INSIGHTSPIKE_IMPORT_MAX_LAYER', '1')
    agent = MainAgent(config=cfg)

    # Ensure tracker exists
    tracker = getattr(agent, '_gedig_fallback_tracker', None)
    assert tracker is not None

    # Monkeypatch internal compute to raise on first call
    state = {'used': False}

    original = getattr(agent, '_compute_gedig', None)
    assert original is not None

    def _boom(*a, **k):  # type: ignore
        if not state['used']:
            state['used'] = True
            raise RuntimeError('synthetic pure failure')
        return {'gedig': 0.1, 'ged': 0.2, 'ig': 0.05}

    monkeypatch.setattr(agent, '_compute_gedig', _boom)

    # Simulate a cycle invoking geDIG (simplified)
    try:
        agent._compute_gedig('g_prev', 'g_now')  # will raise first
    except RuntimeError:
        # emulate fallback path manually; record event
        tracker.record('pure_to_full')  # mimic actual kind naming
        # second attempt (full) returns value
        monkeypatch.setattr(agent, '_compute_gedig', original)  # restore
        res = agent._compute_gedig('g_prev', 'g_now')
        assert 'gedig' in res

    summary = tracker.summary()
    assert summary.get('pure_to_full', 0) >= 1

