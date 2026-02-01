"""Unit tests for gedig.spike module."""

import pytest

from insightspike.algorithms.gedig.types import GeDIGResult, SpikeDetectionMode
from insightspike.algorithms.gedig.spike import detect_spike, compute_rewards


class TestDetectSpike:
    """Tests for detect_spike function."""

    def test_threshold_mode_spike_detected(self):
        result = GeDIGResult(gedig_value=-0.7, ged_value=0.3, ig_value=0.8)
        assert detect_spike(
            result,
            mode="threshold",
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
        ) is True

    def test_threshold_mode_no_spike(self):
        result = GeDIGResult(gedig_value=-0.3, ged_value=0.3, ig_value=0.8)
        assert detect_spike(
            result,
            mode="threshold",
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
        ) is False

    def test_and_mode_both_conditions_met(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.3,
            ig_z_score=0.5,
        )
        assert detect_spike(
            result,
            mode="and",
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
        ) is True

    def test_and_mode_only_structural_met(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.3,
            ig_z_score=0.1,  # below tau_i
        )
        assert detect_spike(
            result,
            mode="and",
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
        ) is False

    def test_and_mode_fallback_low_variance(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.5,  # > tau_s * 2
            ig_z_score=0.1,
        )
        # With negligible variance, structural improvement alone triggers spike
        assert detect_spike(
            result,
            mode="and",
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
            ig_variance=0.0,
        ) is True

    def test_or_mode_structural_met(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.3,
            ig_z_score=0.1,
        )
        assert detect_spike(
            result,
            mode="or",
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
        ) is True

    def test_or_mode_ig_met(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.05,
            ig_z_score=0.5,
        )
        assert detect_spike(
            result,
            mode="or",
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
        ) is True

    def test_or_mode_backward_compat(self):
        # OR mode with positive signals triggers spike for backward compat
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.01,  # positive but below tau_s
            ig_z_score=0.01,  # positive but below tau_i
        )
        assert detect_spike(
            result,
            mode="or",
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
        ) is True

    def test_mode_as_enum(self):
        result = GeDIGResult(gedig_value=-0.7, ged_value=0.3, ig_value=0.8)
        assert detect_spike(
            result,
            mode=SpikeDetectionMode.THRESHOLD,
            spike_threshold=-0.5,
            tau_s=0.15,
            tau_i=0.25,
        ) is True


class TestComputeRewards:
    """Tests for compute_rewards function."""

    def test_basic_reward_computation(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            delta_ged_norm=0.3,
            ig_z_score=1.5,
        )
        compute_rewards(
            result,
            lambda_weight=1.0,
            mu=0.5,
            decay_factor=0.7,
            warmup_steps=10,
            ig_count=20,
        )
        # After warmup, lambda is used
        assert result.hop0_reward != 0.0
        assert result.aggregate_reward == result.hop0_reward  # no hop_results

    def test_warmup_period(self):
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            delta_ged_norm=0.3,
            ig_z_score=1.5,
        )
        compute_rewards(
            result,
            lambda_weight=1.0,
            mu=0.5,
            decay_factor=0.7,
            warmup_steps=10,
            ig_count=5,  # Still in warmup
        )
        # During warmup, lambda is 0, so reward is only structural
        structural_signal = -result.delta_ged_norm
        expected = 0.0 * result.ig_z_score + 0.5 * structural_signal
        assert abs(result.hop0_reward - expected) < 1e-6

    def test_with_hop_results(self):
        from insightspike.algorithms.gedig.types import HopResult

        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            delta_ged_norm=0.3,
            ig_z_score=1.5,
            hop_results={
                0: HopResult(hop=0, ged=0.3, ig=0.4, gedig=-0.1, struct_cost=0.3, node_count=5, edge_count=4),
                1: HopResult(hop=1, ged=0.2, ig=0.5, gedig=-0.2, struct_cost=0.2, node_count=8, edge_count=7),
                2: HopResult(hop=2, ged=0.1, ig=0.6, gedig=-0.3, struct_cost=0.1, node_count=12, edge_count=11),
            },
        )
        compute_rewards(
            result,
            lambda_weight=1.0,
            mu=0.5,
            decay_factor=0.7,
            warmup_steps=10,
            ig_count=20,
        )
        # With hop_results, aggregate_reward uses weighted average
        assert result.aggregate_reward != result.hop0_reward
