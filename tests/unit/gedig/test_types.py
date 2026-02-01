"""Unit tests for gedig.types module."""

import pytest
from dataclasses import asdict

from insightspike.algorithms.gedig.types import (
    ProcessingMode,
    SpikeDetectionMode,
    HopResult,
    GeDIGResult,
    LinksetMetrics,
)


class TestProcessingMode:
    """Tests for ProcessingMode enum."""

    def test_mode_values(self):
        assert ProcessingMode.WAKE.value == "wake"
        assert ProcessingMode.SLEEP.value == "sleep"

    def test_mode_from_string(self):
        assert ProcessingMode("wake") == ProcessingMode.WAKE
        assert ProcessingMode("sleep") == ProcessingMode.SLEEP

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            ProcessingMode("invalid")


class TestSpikeDetectionMode:
    """Tests for SpikeDetectionMode enum."""

    def test_mode_values(self):
        assert SpikeDetectionMode.THRESHOLD.value == "threshold"
        assert SpikeDetectionMode.AND.value == "and"
        assert SpikeDetectionMode.OR.value == "or"

    def test_mode_from_string(self):
        assert SpikeDetectionMode("threshold") == SpikeDetectionMode.THRESHOLD
        assert SpikeDetectionMode("and") == SpikeDetectionMode.AND
        assert SpikeDetectionMode("or") == SpikeDetectionMode.OR


class TestHopResult:
    """Tests for HopResult dataclass."""

    def test_required_fields(self):
        hr = HopResult(
            hop=0,
            ged=0.1,
            ig=0.2,
            gedig=-0.1,
            struct_cost=0.1,
            node_count=5,
            edge_count=4,
        )
        assert hr.hop == 0
        assert hr.ged == 0.1
        assert hr.ig == 0.2
        assert hr.gedig == -0.1
        assert hr.struct_cost == 0.1
        assert hr.node_count == 5
        assert hr.edge_count == 4
        assert hr.sp == 0.0  # default

    def test_all_fields(self):
        hr = HopResult(
            hop=2,
            ged=0.5,
            ig=0.3,
            gedig=-0.2,
            struct_cost=0.4,
            node_count=10,
            edge_count=15,
            sp=0.1,
            h_component=0.25,
            ged_raw=2.0,
            ged_den=4.0,
            entropy_before=1.5,
            entropy_after=1.2,
            ig_delta=-0.3,
            ig_den=2.0,
            variance_reduction=0.05,
        )
        assert hr.hop == 2
        assert hr.node_count == 10
        assert hr.edge_count == 15
        assert hr.entropy_before == 1.5

    def test_to_dict(self):
        hr = HopResult(
            hop=1,
            ged=0.1,
            ig=0.2,
            gedig=-0.1,
            struct_cost=0.05,
            node_count=3,
            edge_count=2,
        )
        d = asdict(hr)
        assert isinstance(d, dict)
        assert d["hop"] == 1
        assert d["ged"] == 0.1


class TestGeDIGResult:
    """Tests for GeDIGResult dataclass."""

    def test_default_values(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
        )
        assert result.gedig_value == -0.5
        assert result.ged_value == 0.3
        assert result.ig_value == 0.8
        assert result.structural_improvement == 0.0
        assert result.spike is False
        assert result.hop_results is None  # Default is None, not {}

    def test_has_spike_property(self):
        result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8, spike=True)
        assert result.has_spike is True

        result2 = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8, spike=False)
        assert result2.has_spike is False

    def test_reward_field(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            reward=0.7,
        )
        assert result.reward == 0.7

    def test_to_dict(self):
        result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["gedig_value"] == -0.5


class TestLinksetMetrics:
    """Tests for LinksetMetrics dataclass."""

    def test_required_fields(self):
        metrics = LinksetMetrics(
            delta_ged_norm=0.1,
            delta_h_norm=-0.2,
            delta_sp_rel=0.05,
            gedig_value=-0.1,
            raw_ged=1.0,
            ged_norm_den=10.0,
            ig_norm_den=2.0,
            entropy_before=1.5,
            entropy_after=1.3,
            ig_delta=-0.2,
            before_size=5,
            after_size=6,
            query_similarity=0.95,
        )
        assert metrics.delta_ged_norm == 0.1
        assert metrics.delta_h_norm == -0.2
        assert metrics.before_size == 5
        assert metrics.after_size == 6
        assert metrics.pos_w_before == 0  # default
        assert metrics.pos_w_after == 0  # default

    def test_all_fields(self):
        metrics = LinksetMetrics(
            delta_ged_norm=0.1,
            delta_h_norm=-0.2,
            delta_sp_rel=0.05,
            gedig_value=-0.1,
            raw_ged=1.0,
            ged_norm_den=10.0,
            ig_norm_den=2.0,
            entropy_before=1.5,
            entropy_after=1.3,
            ig_delta=-0.2,
            before_size=5,
            after_size=6,
            query_similarity=0.95,
            pos_w_before=5,
            pos_w_after=6,
            topw_before=[0.9, 0.8, 0.7],
            topw_after=[0.95, 0.85, 0.75],
        )
        assert metrics.before_size == 5
        assert metrics.after_size == 6
        assert len(metrics.topw_before) == 3
