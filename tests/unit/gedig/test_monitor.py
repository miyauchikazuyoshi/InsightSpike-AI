"""Unit tests for gedig.monitor module."""

import json
import os
import tempfile

import pytest

from insightspike.algorithms.gedig.monitor import GeDIGMonitor
from insightspike.algorithms.gedig.types import GeDIGResult, HopResult


class MockCore:
    """Mock core for testing."""

    def __init__(self, tau_s=0.15, tau_i=0.25, lambda_weight=1.0, mu=0.5):
        self.tau_s = tau_s
        self.tau_i = tau_i
        self.lambda_weight = lambda_weight
        self.mu = mu
        self.spike_detection_mode = "and"


class TestGeDIGMonitor:
    """Tests for GeDIGMonitor class."""

    def test_init_defaults(self):
        monitor = GeDIGMonitor()
        assert monitor.target_fp_rate == 0.1
        assert monitor.adjust_factor == 1.1
        assert monitor.gt_mode == "and"
        assert len(monitor.pred_buffer) == 0

    def test_init_custom(self):
        monitor = GeDIGMonitor(
            window_size=100,
            target_fp_rate=0.2,
            adjust_factor=1.2,
            gt_mode="or",
        )
        assert monitor.target_fp_rate == 0.2
        assert monitor.adjust_factor == 1.2
        assert monitor.gt_mode == "or"

    def test_record_prediction(self):
        monitor = GeDIGMonitor(window_size=10)
        monitor.record_prediction(True)
        monitor.record_prediction(False)
        monitor.record_prediction(True)
        assert len(monitor.pred_buffer) == 3
        assert list(monitor.pred_buffer) == [1, 0, 1]

    def test_record_prediction_overflow(self):
        monitor = GeDIGMonitor(window_size=3)
        for i in range(5):
            monitor.record_prediction(True)
        assert len(monitor.pred_buffer) == 3

    def test_record_outcome(self):
        monitor = GeDIGMonitor()
        monitor.record_prediction(True)
        monitor.record_outcome(False)  # FP
        assert len(monitor.fp_buffer) == 1
        assert monitor.fp_buffer[0] == 1

    def test_record_outcome_true_positive(self):
        monitor = GeDIGMonitor()
        monitor.record_prediction(True)
        monitor.record_outcome(True)  # TP
        assert len(monitor.fp_buffer) == 1
        assert monitor.fp_buffer[0] == 0

    def test_record_outcome_no_prediction(self):
        monitor = GeDIGMonitor()
        monitor.record_outcome(True)  # No pred, should be ignored
        assert len(monitor.fp_buffer) == 0

    def test_spike_rate_empty(self):
        monitor = GeDIGMonitor()
        assert monitor.spike_rate() == 0.0

    def test_spike_rate_calculation(self):
        monitor = GeDIGMonitor()
        monitor.record_prediction(True)
        monitor.record_prediction(False)
        monitor.record_prediction(True)
        monitor.record_prediction(False)
        assert monitor.spike_rate() == 0.5

    def test_false_positive_rate_empty(self):
        monitor = GeDIGMonitor()
        assert monitor.false_positive_rate() == 0.0

    def test_false_positive_rate_calculation(self):
        monitor = GeDIGMonitor()
        # 4 predictions, 2 FPs
        for pred, actual in [(True, False), (True, True), (True, False), (False, False)]:
            monitor.record_prediction(pred)
            monitor.record_outcome(actual)
        assert monitor.false_positive_rate() == 0.5

    def test_derive_ground_truth_threshold_mode(self):
        monitor = GeDIGMonitor(gt_mode="threshold")
        core = MockCore()
        result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8, spike=True)
        assert monitor.derive_ground_truth(result, core) is True

    def test_derive_ground_truth_and_mode(self):
        monitor = GeDIGMonitor(gt_mode="and", gt_si_threshold=0.1, gt_igz_threshold=0.2)
        core = MockCore()
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.2,
            ig_z_score=0.3,
        )
        assert monitor.derive_ground_truth(result, core) is True

    def test_derive_ground_truth_and_mode_fails(self):
        monitor = GeDIGMonitor(gt_mode="and", gt_si_threshold=0.1, gt_igz_threshold=0.5)
        core = MockCore()
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.2,
            ig_z_score=0.3,  # below threshold
        )
        assert monitor.derive_ground_truth(result, core) is False

    def test_derive_ground_truth_or_mode(self):
        monitor = GeDIGMonitor(gt_mode="or", gt_si_threshold=0.1, gt_igz_threshold=0.5)
        core = MockCore()
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            structural_improvement=0.2,
            ig_z_score=0.3,  # below threshold but SI passes
        )
        assert monitor.derive_ground_truth(result, core) is True

    def test_record_auto_outcome(self):
        monitor = GeDIGMonitor(gt_mode="threshold")
        core = MockCore()
        result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8, spike=True)
        monitor.record_prediction(True)
        label = monitor.record_auto_outcome(result, core)
        assert label is True
        assert len(monitor.actual_buffer) == 1

    def test_get_metrics(self):
        monitor = GeDIGMonitor()
        monitor.record_prediction(True)
        monitor.record_prediction(False)
        metrics = monitor.get_metrics()
        assert "spike_rate" in metrics
        assert "false_positive_rate" in metrics
        assert "n_predictions" in metrics
        assert metrics["n_predictions"] == 2

    def test_auto_adjust_thresholds_not_enough_data(self):
        monitor = GeDIGMonitor()
        core = MockCore(tau_s=0.15, tau_i=0.25)
        original_tau_s = core.tau_s
        monitor.auto_adjust_thresholds(core)
        assert core.tau_s == original_tau_s  # No change

    def test_auto_adjust_thresholds_high_fp(self):
        monitor = GeDIGMonitor(target_fp_rate=0.1)
        core = MockCore(tau_s=0.15, tau_i=0.25)
        # Simulate high FP rate
        for _ in range(20):
            monitor.record_prediction(True)
            monitor.record_outcome(False)  # All FPs
        original_tau_s = core.tau_s
        monitor.auto_adjust_thresholds(core)
        assert core.tau_s > original_tau_s  # Thresholds increased

    def test_auto_adjust_thresholds_low_fp_low_spike(self):
        monitor = GeDIGMonitor(target_fp_rate=0.1)
        core = MockCore(tau_s=0.15, tau_i=0.25)
        # Simulate low FP rate and low spike rate
        for _ in range(20):
            monitor.record_prediction(False)
            monitor.record_outcome(False)
        original_tau_s = core.tau_s
        monitor.auto_adjust_thresholds(core)
        assert core.tau_s < original_tau_s  # Thresholds decreased

    def test_auto_adjust_thresholds_clipping(self):
        monitor = GeDIGMonitor()
        core = MockCore(tau_s=100.0, tau_i=100.0)
        for _ in range(20):
            monitor.record_prediction(True)
            monitor.record_outcome(False)
        monitor.auto_adjust_thresholds(core)
        assert core.tau_s <= 10.0  # Clipped to max

    def test_export_metrics_json(self):
        monitor = GeDIGMonitor()
        core = MockCore()
        monitor.record_prediction(True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.json")
            monitor.export_metrics(path, core)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "metrics" in data
            assert data["metrics"]["tau_s"] == core.tau_s

    def test_export_metrics_csv(self):
        monitor = GeDIGMonitor()
        core = MockCore()
        monitor.record_prediction(True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.csv")
            monitor.export_metrics(path, core)
            assert os.path.exists(path)

    def test_summarize_hop_results_empty(self):
        monitor = GeDIGMonitor()
        result = GeDIGResult(gedig_value=-0.5, ged_value=0.3, ig_value=0.8)
        summary = monitor.summarize_hop_results(result)
        assert summary == {}

    def test_summarize_hop_results_with_data(self):
        monitor = GeDIGMonitor()
        result = GeDIGResult(
            gedig_value=-0.5,
            ged_value=0.3,
            ig_value=0.8,
            hop_results={
                0: HopResult(hop=0, ged=0.3, ig=0.4, gedig=-0.1, struct_cost=0.3, node_count=5, edge_count=4),
                1: HopResult(hop=1, ged=0.2, ig=0.5, gedig=-0.2, struct_cost=0.2, node_count=8, edge_count=7),
                2: HopResult(hop=2, ged=0.1, ig=0.6, gedig=-0.3, struct_cost=0.1, node_count=12, edge_count=11),
            },
        )
        summary = monitor.summarize_hop_results(result)
        assert "hop_gedig_mean" in summary
        assert "hop_gedig_p95" in summary
        assert "hop_gedig_max" in summary
        assert summary["hop_count"] == 3.0

    def test_zero_spike_backoff(self):
        monitor = GeDIGMonitor(window_size=10)
        core = MockCore(tau_s=0.15, tau_i=0.25)
        # Fill buffer with no spikes
        for _ in range(15):
            monitor.record_prediction(False)
            monitor.record_outcome(False)
        original_tau_s = core.tau_s
        monitor.auto_adjust_thresholds(core)
        # Strong relaxation applied
        assert core.tau_s < original_tau_s
        assert monitor.zero_spike_backoff_count >= 1
