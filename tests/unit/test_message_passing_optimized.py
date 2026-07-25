"""Regression tests for optimized message-passing traversal."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

from insightspike.graph.message_passing_optimized import OptimizedMessagePassing


def test_k_hop_neighbors_reaches_the_requested_depth() -> None:
    adjacency = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2],
    }
    message_passing = OptimizedMessagePassing(max_hops=2)

    assert message_passing._get_k_hop_neighbors(
        adjacency,
        {0},
        2,
    ) == {0, 1, 2}


def test_k_hop_neighbors_does_not_exceed_the_requested_depth() -> None:
    adjacency = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2],
    }
    message_passing = OptimizedMessagePassing(max_hops=1)

    assert message_passing._get_k_hop_neighbors(
        adjacency,
        {0},
        1,
    ) == {0, 1}


def test_attention_name_is_a_warned_mean_alias() -> None:
    with pytest.warns(FutureWarning):
        message_passing = OptimizedMessagePassing(
            aggregation="attention"
        )

    assert message_passing.aggregation == "mean"


def test_unknown_aggregation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown.*aggregation"):
        OptimizedMessagePassing(aggregation="typo")
