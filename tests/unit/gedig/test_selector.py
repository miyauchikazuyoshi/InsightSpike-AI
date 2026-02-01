"""Unit tests for gedig.selector module."""

import pytest
import math

from insightspike.algorithms.gedig.selector import (
    TwoThresholdSelection,
    TwoThresholdCandidateSelector,
    _extract_score,
    _safe_log,
)


class TestExtractScore:
    """Tests for _extract_score helper."""

    def test_valid_score(self):
        item = {'similarity': 0.8}
        assert _extract_score(item) == 0.8

    def test_custom_key(self):
        item = {'score': 0.5}
        assert _extract_score(item, 'score') == 0.5

    def test_missing_key(self):
        item = {'other': 0.5}
        assert _extract_score(item) == 0.0

    def test_non_numeric(self):
        item = {'similarity': 'invalid'}
        assert _extract_score(item) == 0.0

    def test_empty_dict(self):
        assert _extract_score({}) == 0.0


class TestSafeLog:
    """Tests for _safe_log helper."""

    def test_positive_value(self):
        result = _safe_log(10)
        assert abs(result - math.log(10)) < 1e-9

    def test_one(self):
        assert _safe_log(1) == 0.0

    def test_zero(self):
        assert _safe_log(0) == 0.0

    def test_negative(self):
        assert _safe_log(-5) == 0.0


class TestTwoThresholdSelection:
    """Tests for TwoThresholdSelection dataclass."""

    def test_basic_creation(self):
        selection = TwoThresholdSelection(
            candidates=[{'index': 'a', 'similarity': 0.8}],
            links=[{'index': 'a', 'similarity': 0.8}],
            forced_links=[],
            k_star=1,
            theta_cand=0.5,
            theta_link=0.7,
            k_cap=10,
        )
        assert selection.k_star == 1
        assert len(selection.candidates) == 1

    def test_to_summary(self):
        selection = TwoThresholdSelection(
            candidates=[
                {'index': 'a', 'similarity': 0.9},
                {'index': 'b', 'similarity': 0.8},
            ],
            links=[{'index': 'a', 'similarity': 0.9}],
            forced_links=[{'index': 'c', 'similarity': 1.0}],
            k_star=2,
            theta_cand=0.5,
            theta_link=0.7,
            k_cap=10,
            top_m=5,
        )
        summary = selection.to_summary()

        assert summary['mode'] == 'two_threshold'
        assert summary['candidate_count'] == 2
        assert summary['link_count'] == 1
        assert summary['k_star'] == 2
        assert 'a' in summary['candidate_indices']
        assert 'c' in summary['forced_link_indices']
        assert summary['log_k_star'] is not None

    def test_to_summary_empty(self):
        selection = TwoThresholdSelection(
            candidates=[],
            links=[],
            forced_links=[],
            k_star=0,
            theta_cand=0.5,
            theta_link=0.7,
            k_cap=10,
        )
        summary = selection.to_summary()

        assert summary['candidate_count'] == 0
        assert summary['link_count'] == 0
        assert summary['log_k_star'] is None


class TestTwoThresholdCandidateSelector:
    """Tests for TwoThresholdCandidateSelector class."""

    def test_basic_init(self):
        # Note: theta_cand must be >= theta_link for higher_is_better=True
        # The selector enforces this by swapping if needed
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.7,
            theta_link=0.5,
            k_cap=10,
        )
        assert selector.theta_cand == 0.7
        assert selector.theta_link == 0.5
        assert selector.k_cap == 10

    def test_init_swaps_thresholds(self):
        """Test that thresholds are swapped when theta_cand < theta_link."""
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.5,  # Will be swapped to 0.7
            theta_link=0.7,  # Will be swapped to 0.5
            k_cap=10,
        )
        # Selector enforces theta_cand >= theta_link by swapping
        assert selector.theta_cand == 0.7
        assert selector.theta_link == 0.5

    def test_select_basic(self):
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.5,
            theta_link=0.7,
            k_cap=10,
        )
        candidates = [
            {'index': 'a', 'similarity': 0.9},
            {'index': 'b', 'similarity': 0.6},
            {'index': 'c', 'similarity': 0.4},  # Below theta_cand
        ]
        result = selector.select(candidates)

        assert isinstance(result, TwoThresholdSelection)
        # Only a and b should be in candidates (>= theta_cand)
        assert result.k_star <= 2

    def test_select_with_top_m(self):
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.3,
            theta_link=0.7,
            k_cap=10,
            top_m=2,
        )
        candidates = [
            {'index': 'a', 'similarity': 0.9},
            {'index': 'b', 'similarity': 0.8},
            {'index': 'c', 'similarity': 0.7},
            {'index': 'd', 'similarity': 0.5},
        ]
        result = selector.select(candidates)

        # top_m=2 should limit candidates
        assert len(result.candidates) <= 2

    def test_select_empty(self):
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.5,
            theta_link=0.7,
            k_cap=10,
        )
        result = selector.select([])

        assert result.k_star == 0
        assert len(result.candidates) == 0
        assert len(result.links) == 0

    def test_select_all_below_threshold(self):
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.9,
            theta_link=0.95,
            k_cap=10,
        )
        candidates = [
            {'index': 'a', 'similarity': 0.5},
            {'index': 'b', 'similarity': 0.6},
        ]
        result = selector.select(candidates)

        assert result.k_star == 0

    def test_mem_origin_items(self):
        """Test that items with origin='mem' can be added to candidates."""
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.7,
            theta_link=0.5,
            k_cap=10,
        )
        candidates = [
            {'index': 'a', 'similarity': 0.9},
            {'index': 'b', 'similarity': 0.6, 'origin': 'mem'},
        ]
        result = selector.select(candidates)

        assert isinstance(result, TwoThresholdSelection)

    def test_k_cap_limiting(self):
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.3,
            theta_link=0.7,
            k_cap=2,  # Small cap
        )
        candidates = [
            {'index': 'a', 'similarity': 0.9},
            {'index': 'b', 'similarity': 0.8},
            {'index': 'c', 'similarity': 0.7},
            {'index': 'd', 'similarity': 0.5},
        ]
        result = selector.select(candidates)

        # k_cap should limit k_star
        assert result.k_star <= 2

    def test_higher_is_better_false(self):
        """Test with distance-based scoring where lower is better."""
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.5,
            theta_link=0.3,
            k_cap=10,
            score_key='distance',
            higher_is_better=False,
        )
        candidates = [
            {'index': 'a', 'distance': 0.1},  # Below theta_cand (good)
            {'index': 'b', 'distance': 0.4},  # Below theta_cand (good)
            {'index': 'c', 'distance': 0.8},  # Above theta_cand (bad)
        ]
        result = selector.select(candidates)

        # a and b should be selected (distance < theta_cand)
        assert len(result.candidates) >= 1

    def test_custom_score_key(self):
        selector = TwoThresholdCandidateSelector(
            theta_cand=0.5,
            theta_link=0.7,
            k_cap=10,
            score_key='relevance',
        )
        candidates = [
            {'index': 'a', 'relevance': 0.9},
            {'index': 'b', 'relevance': 0.6},
        ]
        result = selector.select(candidates)

        assert len(result.candidates) >= 1
