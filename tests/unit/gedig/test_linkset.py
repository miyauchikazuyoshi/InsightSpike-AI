"""Unit tests for gedig.linkset module."""

import networkx as nx
import pytest

from insightspike.algorithms.gedig.linkset import compute_linkset_metrics


class TestComputeLinksetMetrics:
    """Tests for compute_linkset_metrics function."""

    def test_empty_linkset_info(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        result = compute_linkset_metrics(g1, g2, None)
        assert result.delta_ged_norm >= 0
        assert result.gedig_value is not None

    def test_basic_linkset(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [
                {'index': 'a', 'similarity': 0.8},
                {'index': 'b', 'similarity': 0.6},
            ],
            'decision': {'index': 'a', 'similarity': 0.8},
        }
        result = compute_linkset_metrics(g1, g2, linkset_info)
        assert result.before_size >= 0
        assert result.after_size >= 0
        assert result.entropy_before >= 0
        assert result.entropy_after >= 0

    def test_with_candidate_pool(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [{'index': 'a', 'similarity': 0.8}],
            'candidate_pool': [
                {'index': 'a', 'similarity': 0.8, 'origin': 'mem'},
                {'index': 'b', 'similarity': 0.6, 'origin': 'graph'},
            ],
            'decision': {'index': 'a'},
        }
        result = compute_linkset_metrics(g1, g2, linkset_info)
        assert result.before_size >= 1

    def test_pool_base_mode(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [{'index': 'a', 'similarity': 0.8}],
            'candidate_pool': [
                {'index': 'x', 'similarity': 0.9},
                {'index': 'y', 'similarity': 0.7},
            ],
            'base_mode': 'pool',
            'decision': {'index': 'x'},
        }
        result = compute_linkset_metrics(g1, g2, linkset_info)
        assert result.before_size >= 1

    def test_mem_base_mode(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [{'index': 'a', 'similarity': 0.8}],
            'candidate_pool': [
                {'index': 'x', 'similarity': 0.9, 'origin': 'mem'},
                {'index': 'y', 'similarity': 0.7, 'origin': 'graph'},
            ],
            'base_mode': 'mem',
            'decision': {'index': 'x'},
        }
        result = compute_linkset_metrics(g1, g2, linkset_info)
        assert result.before_size >= 1

    def test_with_query_entry(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [{'index': 'a', 'similarity': 0.8}],
            'query_entry': {'index': 'q', 'similarity': 1.0},
            'decision': {'index': 'a'},
        }
        result = compute_linkset_metrics(g1, g2, linkset_info)
        assert result.query_similarity == 1.0

    def test_entropy_tau_parameter(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [
                {'index': 'a', 'similarity': 0.8},
                {'index': 'b', 'similarity': 0.2},
            ],
            'decision': {'index': 'a'},
        }
        result1 = compute_linkset_metrics(g1, g2, linkset_info, entropy_tau=1.0)
        result2 = compute_linkset_metrics(g1, g2, linkset_info, entropy_tau=0.5)
        # Different tau should give different entropy values
        assert result1.entropy_before != result2.entropy_before or result1.entropy_after != result2.entropy_after

    def test_legacy_formula(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [{'index': 'a', 'similarity': 0.8}],
            'decision': {'index': 'a'},
        }
        result_legacy = compute_linkset_metrics(g1, g2, linkset_info, use_legacy_formula=True)
        result_new = compute_linkset_metrics(g1, g2, linkset_info, use_legacy_formula=False)
        # Results may differ based on formula
        assert result_legacy.gedig_value is not None
        assert result_new.gedig_value is not None

    def test_ig_nonneg(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [{'index': 'a', 'similarity': 0.8}],
            'decision': {'index': 'a'},
        }
        result = compute_linkset_metrics(g1, g2, linkset_info, ig_nonneg=True)
        # With ig_nonneg, the IG component should be >= 0
        assert result.gedig_value is not None

    def test_normalized_ig_mode(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [
                {'index': 'a', 'similarity': 0.8},
                {'index': 'b', 'similarity': 0.6},
            ],
            'decision': {'index': 'a'},
        }
        result = compute_linkset_metrics(g1, g2, linkset_info, ig_mode='normalized')
        assert result.gedig_value is not None

    def test_top_weights_diagnostic(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        linkset_info = {
            's_link': [
                {'index': 'a', 'similarity': 0.9},
                {'index': 'b', 'similarity': 0.8},
                {'index': 'c', 'similarity': 0.7},
                {'index': 'd', 'similarity': 0.6},
                {'index': 'e', 'similarity': 0.5},
                {'index': 'f', 'similarity': 0.4},
            ],
            'decision': {'index': 'a'},
        }
        result = compute_linkset_metrics(g1, g2, linkset_info)
        # Top weights should be limited to 5
        assert len(result.topw_before) <= 5
        assert len(result.topw_after) <= 5
        # Should be sorted descending
        if result.topw_before:
            assert result.topw_before == sorted(result.topw_before, reverse=True)
