"""SPEC Phase-1 verification: sleep propagation semantics vs SPEC.md section 3.3.

Background. The staged plan in graph_persistent_dg/SPEC.md section 6 called
for verifying, BEFORE any effect ablation, that propagated values have the
designed sign structure (goal-side positive, dead-end-side negative). That
intermediate check was skipped when Phases 0-2 landed in one commit
(e47829a). Writing it as a test (2026-07-02) exposed a real divergence:

  DESIGN (SPEC section 3.3, one-directional intuition):
      goal=+1.0, E=+1.25, D=+1.49, C=-0.65, B=-0.32, dead_end=-1.0
      -> at branch A, the dead-end side is NEGATIVE and gets avoided.

  IMPLEMENTATION (undirected max-propagation, gamma=0.95, fixed point
  ~ reward/(1-gamma^2) via mutual reinforcement, e.g. goal<->E):
      every node inflates positive: goal~13, E~12.7, ..., dead_end~+9.9
      -> the dead-end NEVER goes negative; tanh(propagated) (dim9)
         saturates to ~1.0 on every node of a goal-reaching episode.

  What survives: the ORDERING (goal-side > dead-end-side) and hence a
  gradient signal; the "avoid negative" story does not.

This file therefore pins BOTH: the implemented semantics (regression
guard for the pre-registered ablation, docs/prereg/maze_sleep_ablation.md)
and the SPEC design intent (strict xfail — if someone fixes propagation
to match SPEC, the xpass will flag it loudly).
"""

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_persistent_dg.sleep_propagate import (  # noqa: E402
    propagate_rewards,
    sleep_optimize,
)

GAMMA = 0.95
N_ITERS = 50  # production default (qhlib/cli.py --sleep-propagate-iters)


def build_spec_example() -> nx.Graph:
    """The two-path graph of SPEC section 3.3.

    path 1 (fail):    start - A - B - C - dead_end
    path 2 (success): start - A - D - E - goal
    Rewards use the SPEC draft values; the propagation operator is
    reward-agnostic so the semantics conclusions carry over to the
    implemented reward table (+0.2/-0.4).
    """
    g = nx.Graph()
    rewards = {
        "start": 0.0, "A": 0.3, "B": 0.3, "C": 0.3, "dead_end": -1.0,
        "D": 0.3, "E": 0.3, "goal": 1.0,
    }
    for node, r in rewards.items():
        g.add_node(node, reward=r)
    g.add_edges_from([
        ("start", "A"),
        ("A", "B"), ("B", "C"), ("C", "dead_end"),
        ("A", "D"), ("D", "E"), ("E", "goal"),
    ])
    return g


def propagated(g: nx.Graph) -> dict:
    return {n: d["propagated"] for n, d in g.nodes(data=True)}


class TestImplementedSemantics:
    """Regression guard: what the shipped propagation actually does.

    These tests document (and pin) the behavior that the pre-registered
    sleep ablation will be run against. If they break, the ablation arms
    are no longer comparable to v6_perseed.
    """

    def setup_method(self):
        self.g = build_spec_example()
        propagate_rewards(self.g, gamma=GAMMA, n_iters=N_ITERS)
        self.p = propagated(self.g)

    def test_goal_episode_inflates_all_nodes_positive(self):
        # Undirected mutual reinforcement: even the dead end turns positive.
        assert all(v > 0 for v in self.p.values()), self.p
        assert self.p["dead_end"] > 5.0  # grossly positive, not mildly

    def test_ordering_toward_goal_survives(self):
        # The usable signal is the ordering, not the sign:
        # goal side dominates dead-end side at the branch.
        assert self.p["D"] > self.p["B"]
        assert self.p["goal"] > self.p["E"] > self.p["D"]
        assert self.p["A"] > self.p["B"] > self.p["C"] > self.p["dead_end"]

    def test_dim9_saturates_on_goal_episode(self):
        # tanh sync (sleep_optimize step 4) saturates: dim9 carries almost
        # no discriminative signal between goal-side and dead-end-side.
        spread = abs(math.tanh(self.p["D"]) - math.tanh(self.p["dead_end"]))
        assert math.tanh(self.p["dead_end"]) > 0.999
        assert spread < 1e-3

    def test_failed_episode_stays_below_goal_scale(self):
        # SPEC section 4.4 case: goal never reached (no +1.0 anywhere).
        g = build_spec_example()
        g.nodes["goal"]["reward"] = 0.3  # just another novel cell
        propagate_rewards(g, gamma=GAMMA, n_iters=N_ITERS)
        p = propagated(g)
        # chain-wide mutual reinforcement drives values toward r/(1-gamma)
        # = 0.3/0.05 = 6.0 (observed max ~5.97) — still saturating tanh,
        # but strictly below the goal-episode scale (~13).
        assert max(p.values()) < 0.3 / (1.0 - GAMMA) + 0.1
        assert max(p.values()) > 4.0  # documents the inflation, not mild values
        # dead end is the minimum, but note: it is STILL not negative
        assert min(p, key=p.get) == "dead_end"
        assert p["dead_end"] > 0.0


class TestSpecDesignIntent:
    """SPEC section 3.3/3.4 design intent — NOT satisfied by the implementation.

    strict xfail: if propagation is ever changed to match the SPEC story
    (e.g. goal as absorbing state, directed propagation, or single-sweep),
    these will XPASS and fail the suite, forcing a conscious update here
    and in the prereg's interpretation section.
    """

    def setup_method(self):
        self.g = build_spec_example()
        propagate_rewards(self.g, gamma=GAMMA, n_iters=N_ITERS)
        self.p = propagated(self.g)

    @pytest.mark.xfail(strict=True, reason="undirected max-propagation never yields negative dead-end side (SPEC 3.3 story unimplemented)")
    def test_dead_end_side_chain_is_negative(self):
        assert self.p["dead_end"] < 0.0
        assert self.p["C"] < 0.0

    @pytest.mark.xfail(strict=True, reason="branch decision is ordering-based, not avoid-negative (SPEC 3.4 story unimplemented)")
    def test_branch_sees_negative_vs_positive_contrast(self):
        assert self.p["B"] < 0.0 < self.p["D"]

    @pytest.mark.xfail(strict=True, reason="SPEC 3.3 magnitudes assume one-directional sweep; implementation fixed point is ~reward/(1-gamma^2)")
    def test_spec_magnitudes(self):
        assert self.p["E"] == pytest.approx(1.25, abs=0.05)
        assert self.p["D"] == pytest.approx(1.49, abs=0.05)


class TestSleepOptimizePackage:
    """sleep_optimize = propagation + isolate removal + dim9/dim8 sync."""

    def test_isolate_removed_and_vectors_synced(self):
        g = build_spec_example()
        g.add_node("orphan", reward=0.1)
        for n in g.nodes:
            g.nodes[n]["abs_vector"] = np.zeros(10)
        out = sleep_optimize(g, gamma=GAMMA, n_iters=N_ITERS)
        assert "orphan" not in out.nodes
        for _n, d in out.nodes(data=True):
            assert d["abs_vector"][8] == pytest.approx(d.get("reward", 0.0))
            assert d["abs_vector"][9] == pytest.approx(math.tanh(d["propagated"]))

    def test_input_graph_untouched(self):
        g = build_spec_example()
        sleep_optimize(g, gamma=GAMMA, n_iters=N_ITERS)
        assert all("propagated" not in d for _, d in g.nodes(data=True))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
