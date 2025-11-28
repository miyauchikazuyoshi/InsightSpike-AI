import os, sys, time
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP_PATH = os.path.abspath(os.path.join(ROOT, 'experiments', 'maze-navigation-enhanced', 'src'))
if EXP_PATH not in sys.path:
    sys.path.insert(0, EXP_PATH)

if not os.path.exists(EXP_PATH):
    pytest.skip("maze-navigation-enhanced sources are not available", allow_module_level=True)

from core.eviction_policy import get_policy  # type: ignore

class DummyEp:
    def __init__(self, eid, ts, visits, pos):
        self.episode_id = eid
        self.timestamp = ts
        self.visit_count = visits
        self.position = pos


def _make_episodes(n=10):
    now = time.time()
    eps = []
    for i in range(n):
        eps.append(DummyEp(i, now - (n-i), visits=i%3, pos=(i,0)))
    return eps


def test_heuristic_policy_size_and_determinism():
    episodes = _make_episodes(12)
    policy = get_policy('heuristic')
    sel1 = policy.select(episodes, 5, context={'current_pos': (0,0)})
    sel2 = policy.select(episodes, 5, context={'current_pos': (0,0)})
    assert len(sel1) == 5
    assert sel1 == sel2  # deterministic given same input
    assert len(set(sel1)) == 5


def test_lru_policy_orders_oldest():
    episodes = _make_episodes(8)
    policy = get_policy('lru')
    selected = policy.select(episodes, 3, context={'current_pos': (0,0)})
    # oldest timestamps correspond to smallest timestamp (now - (n-i)) => episode 0 oldest
    assert selected[0] == 0
    assert len(selected) == 3


def test_get_policy_default_and_alias():
    episodes = _make_episodes(5)
    default_policy = get_policy(None)
    alias_policy = get_policy('lru_visit')
    assert default_policy.name == 'heuristic'
    assert alias_policy.name == 'lru_visit'
    assert alias_policy.select(episodes, 2, context={'current_pos': (0,0)})
