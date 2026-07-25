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
    annotate_dg_size,
    propagate_rewards,
    sleep_optimize,
    sleep_replay_optimize,
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


class TestReplayVariant:
    """sleep_replay_optimize: Q-table transcription (redesign variant #1).

    The Q table itself comes from qhlib.sleep.build_sleep_q_table (directed
    episodic backup, long-standing dictionary-sleep code). These tests pin
    the transcription contract: bounded values, negatives SURVIVE (unlike
    max-propagation), correct node-kind routing, dim9 non-saturation.
    """

    def build_graph(self):
        g = nx.Graph()
        # direction nodes (r, c, action 0-3), query nodes (r, c, -1)
        for node, reward in [
            ((1, 1, -1), 0.0), ((1, 1, 0), 0.2), ((1, 1, 2), -1.0),
            ((1, 2, -1), 0.0), ((1, 2, 0), 1.0),
        ]:
            g.add_node(node, reward=reward, abs_vector=np.zeros(10))
        g.add_edges_from([
            ((1, 1, -1), (1, 1, 0)), ((1, 1, -1), (1, 1, 2)),
            ((1, 1, 0), (1, 2, -1)), ((1, 2, -1), (1, 2, 0)),
        ])
        g.add_node("orphan", reward=0.0, abs_vector=np.zeros(10))
        return g

    def setup_method(self):
        from graph_persistent_dg.sleep_propagate import sleep_replay_optimize
        self.q = {
            (1, 1): {0: 0.85, 2: -0.35},   # toward goal positive, dead-end direction NEGATIVE
            (1, 2): {0: 0.99},
        }
        self.out = sleep_replay_optimize(self.build_graph(), self.q)

    def test_direction_nodes_get_q_values(self):
        assert self.out.nodes[(1, 1, 0)]["propagated"] == pytest.approx(0.85)
        assert self.out.nodes[(1, 1, 2)]["propagated"] == pytest.approx(-0.35)

    def test_negative_examples_survive(self):
        # The core fix over max-propagation: a negative Q stays negative
        # even though positive neighbors exist.
        assert self.out.nodes[(1, 1, 2)]["propagated"] < 0.0

    def test_query_nodes_get_state_value(self):
        assert self.out.nodes[(1, 1, -1)]["propagated"] == pytest.approx(0.85)  # max_a Q
        assert self.out.nodes[(1, 2, -1)]["propagated"] == pytest.approx(0.99)

    def test_dim9_not_saturated_and_signed(self):
        d9 = {n: d["abs_vector"][9] for n, d in self.out.nodes(data=True)}
        assert d9[(1, 1, 0)] == pytest.approx(math.tanh(0.85))
        assert d9[(1, 1, 2)] == pytest.approx(math.tanh(-0.35))
        assert abs(d9[(1, 1, 0)]) < 0.999  # no saturation at Q scale
        # discriminative spread exists (unlike the saturated on-mode)
        assert abs(d9[(1, 1, 0)] - d9[(1, 1, 2)]) > 0.5

    def test_unknown_state_defaults_zero_and_isolates_removed(self):
        assert "orphan" not in self.out.nodes


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


class TestDgSizeAnnotation:
    """v7 contract: edge β₁ proxy -> source query-state/action projection."""

    @staticmethod
    def build_projection_graph() -> nx.Graph:
        graph = nx.Graph()
        query = (0, 0, -1)
        mid = (0, 1, -1)
        direction_small = (0, 2, 3)
        direction_large = (1, 2, 3)
        unrelated_direction = (0, 1, 1)
        disconnected_query = (10, 10, -1)
        disconnected_direction = (12, 12, 2)
        graph.add_edges_from(
            [
                # Corridor path: query--mid--small--large.
                (query, mid),
                (mid, direction_small),
                (direction_small, direction_large),
                # Two shortcuts suggest action 3 at the same source query.
                (query, direction_small),  # corridor distance 2 + close = 3
                (query, direction_large),  # corridor distance 3 + close = 4
                # Same-cell wiring is never a shortcut.
                (mid, unrelated_direction),
                # No corridor path between these endpoints.
                (disconnected_query, disconnected_direction),
            ]
        )
        graph.nodes[query]["dg_action_sizes"] = {3: 99.0}
        graph.nodes[direction_small]["dg_size"] = 99.0
        graph.edges[query, mid]["dg_size"] = 99.0
        return graph

    def test_projects_edge_cycle_size_to_source_query_action(self):
        graph = self.build_projection_graph()
        query = (0, 0, -1)
        direction_small = (0, 2, 3)
        direction_large = (1, 2, 3)

        annotate_dg_size(graph)

        assert graph.edges[query, direction_small]["dg_size"] == pytest.approx(3.0)
        assert graph.edges[query, direction_large]["dg_size"] == pytest.approx(4.0)
        assert graph.nodes[query]["dg_action_sizes"] == {3: pytest.approx(4.0)}
        assert graph.nodes[query]["dg_size"] == 0.0
        assert graph.nodes[direction_small]["dg_remote_endpoint_max"] == pytest.approx(3.0)
        assert graph.nodes[direction_large]["dg_remote_endpoint_max"] == pytest.approx(4.0)

    def test_resets_stale_values_and_ignores_non_cycles(self):
        graph = self.build_projection_graph()
        annotate_dg_size(graph)

        assert graph.edges[(0, 0, -1), (0, 1, -1)]["dg_size"] == 0.0
        assert graph.nodes[(0, 1, 1)]["dg_size"] == 0.0
        assert graph.edges[(10, 10, -1), (12, 12, 2)]["dg_size"] == 0.0
        assert graph.nodes[(10, 10, -1)]["dg_action_sizes"].get(2, 0.0) == 0.0

    @pytest.mark.parametrize(
        "optimizer",
        [
            lambda graph: sleep_optimize(graph, gamma=GAMMA, n_iters=1),
            lambda graph: sleep_replay_optimize(graph, sleep_q={}),
        ],
    )
    def test_all_sleep_optimizers_materialise_dg(self, optimizer):
        graph = self.build_projection_graph()
        out = optimizer(graph)
        assert out.edges[(0, 0, -1), (1, 2, 3)]["dg_size"] == pytest.approx(4.0)
        assert out.nodes[(0, 0, -1)]["dg_action_sizes"][3] == pytest.approx(4.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
