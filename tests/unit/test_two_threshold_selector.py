import math

import pytest

from insightspike.algorithms.gedig.selector import TwoThresholdCandidateSelector


def test_two_threshold_selector_filters_and_caps():
    docs = [
        {"index": 0, "similarity": 0.92},
        {"index": 1, "similarity": 0.71},
        {"index": 2, "similarity": 0.58},
        {"index": 3, "similarity": 0.41},
        {"index": 4, "similarity": 0.35},
    ]

    selector = TwoThresholdCandidateSelector(theta_cand=0.6, theta_link=0.4, k_cap=3)
    selection = selector.select(docs)

    assert selection.k_star == 2
    assert [item["index"] for item in selection.candidates] == [0, 1]
    assert [item["index"] for item in selection.links] == [0, 1]
    assert selection.forced_links == []

    summary = selection.to_summary()
    assert summary["k_star"] == 2
    assert summary["candidate_indices"][:2] == [0, 1]
    assert all(idx in summary["link_indices"] for idx in [0, 1])
    assert summary["forced_link_indices"] == []


def test_two_threshold_selector_top_m_limits_candidates():
    docs = [
        {"index": i, "similarity": 0.9 - i * 0.05} for i in range(6)
    ]
    selector = TwoThresholdCandidateSelector(theta_cand=0.4, theta_link=0.3, k_cap=5, top_m=3)
    selection = selector.select(docs)

    assert len(selection.candidates) == 3
    assert selection.k_star == 3
    assert [item["index"] for item in selection.candidates] == [0, 1, 2]
    assert selection.forced_links == []


def test_two_threshold_selector_swaps_thresholds_when_needed():
    docs = [
        {"index": 10, "similarity": 0.8},
        {"index": 11, "similarity": 0.6},
    ]
    selector = TwoThresholdCandidateSelector(theta_cand=0.3, theta_link=0.7, k_cap=4)
    selection = selector.select(docs)

    assert selection.k_star == 1
    assert [item["index"] for item in selection.candidates] == [10]
    assert [item["index"] for item in selection.links] == [10]
    assert selection.forced_links == []


def test_two_threshold_selector_applies_radius_filters():
    docs = [
        {"index": 0, "similarity": 0.9, "radius_cand": 0.5, "radius_link": 0.5},
        {"index": 1, "similarity": 0.8, "radius_cand": 5.0, "radius_link": 0.4},
        {"index": 2, "similarity": 0.75, "radius_cand": 0.8, "radius_link": 2.0},
    ]

    selector = TwoThresholdCandidateSelector(
        theta_cand=0.7,
        theta_link=0.7,
        k_cap=5,
        radius_cand=1.0,
        radius_link=1.0,
    )
    selection = selector.select(docs)

    assert [item["index"] for item in selection.candidates] == [0, 2]
    assert [item["index"] for item in selection.links] == [0]
    assert selection.forced_links == []


def test_two_threshold_selector_forces_link_when_no_candidates():
    docs = [
        {"index": 0, "similarity": 0.65, "radius_cand": 0.8, "radius_link": 1.4, "origin": "obs"},
        {"index": 1, "similarity": 0.55, "radius_cand": 1.2, "radius_link": 1.6, "origin": "mem"},
    ]
    selector = TwoThresholdCandidateSelector(
        theta_cand=0.7,
        theta_link=0.6,
        k_cap=4,
        radius_cand=1.5,
        radius_link=0.7,
    )
    selection = selector.select(docs)

    assert selection.k_star == 0
    assert selection.candidates == []
    assert selection.links == []
    assert len(selection.forced_links) == 1
    forced = selection.forced_links[0]
    assert forced["index"] == 0
    assert forced.get("forced") is True

    summary = selection.to_summary()
    assert summary["forced_link_indices"] == [0]


def test_two_threshold_selector_does_not_force_when_linkables_exist():
    docs = [
        {"index": 0, "similarity": 0.9, "radius_cand": 0.1, "radius_link": 0.1},
    ]
    selector = TwoThresholdCandidateSelector(
        theta_cand=0.5,
        theta_link=0.5,
        k_cap=4,
        radius_cand=1.0,
        radius_link=1.0,
    )
    selection = selector.select(docs)

    assert selection.k_star == 1
    assert [item["index"] for item in selection.candidates] == [0]
    assert [item["index"] for item in selection.links] == [0]
    assert selection.forced_links == []
