"""Unit tests for gedig.config module."""

import os
import pytest

from insightspike.algorithms.gedig.config import GeDIGConfig, GeDIGPresets


class TestGeDIGConfig:
    """Tests for GeDIGConfig dataclass."""

    def test_default_values(self):
        config = GeDIGConfig()
        assert config.node_cost == 1.0
        assert config.edge_cost == 1.0
        assert config.lambda_weight == 1.0
        assert config.spike_threshold == -0.5
        assert config.enable_multihop is False
        assert config.max_hops == 3

    def test_custom_values(self):
        config = GeDIGConfig(
            node_cost=2.0,
            edge_cost=0.5,
            lambda_weight=0.7,
            enable_multihop=True,
            max_hops=5,
        )
        assert config.node_cost == 2.0
        assert config.edge_cost == 0.5
        assert config.lambda_weight == 0.7
        assert config.enable_multihop is True
        assert config.max_hops == 5

    def test_from_env_defaults(self, monkeypatch):
        # Clear relevant env vars
        for key in [
            "MAZE_GEDIG_LAMBDA",
            "MAZE_GEDIG_NODE_COST",
            "MAZE_GEDIG_EDGE_COST",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = GeDIGConfig.from_env()
        assert config.lambda_weight == 1.0
        assert config.node_cost == 1.0

    def test_from_env_with_overrides(self, monkeypatch):
        monkeypatch.setenv("MAZE_GEDIG_LAMBDA", "0.7")
        monkeypatch.setenv("MAZE_GEDIG_NODE_COST", "2.0")

        config = GeDIGConfig.from_env()
        assert config.lambda_weight == 0.7
        assert config.node_cost == 2.0

    def test_from_env_with_kwargs_override(self, monkeypatch):
        monkeypatch.setenv("MAZE_GEDIG_LAMBDA", "0.7")

        # kwargs should override env
        config = GeDIGConfig.from_env(lambda_weight=0.9)
        assert config.lambda_weight == 0.9

    def test_preset_balanced(self):
        config = GeDIGConfig.preset("balanced")
        assert config.spike_detection_mode == "and"
        assert config.tau_s == 0.15
        assert config.tau_i == 0.25

    def test_preset_conservative(self):
        config = GeDIGConfig.preset("conservative")
        assert config.spike_detection_mode == "and"
        assert config.tau_s == 0.2
        assert config.tau_i == 0.3

    def test_preset_aggressive(self):
        config = GeDIGConfig.preset("aggressive")
        assert config.spike_detection_mode == "or"
        assert config.tau_s == 0.08
        assert config.tau_i == 0.15

    def test_preset_unknown_raises_error(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            GeDIGConfig.preset("unknown_preset")

    def test_to_dict(self):
        config = GeDIGConfig(lambda_weight=0.8, max_hops=4)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["lambda_weight"] == 0.8
        assert d["max_hops"] == 4


class TestGeDIGPresets:
    """Tests for GeDIGPresets class (legacy compatibility)."""

    def test_conservative_preset(self):
        preset = GeDIGPresets.CONSERVATIVE
        assert isinstance(preset, dict)
        assert preset["spike_detection_mode"] == "and"
        assert preset["tau_s"] == 0.2
        assert preset["tau_i"] == 0.3

    def test_balanced_preset(self):
        preset = GeDIGPresets.BALANCED
        assert isinstance(preset, dict)
        assert preset["spike_detection_mode"] == "and"
        assert preset["tau_s"] == 0.15

    def test_aggressive_preset(self):
        preset = GeDIGPresets.AGGRESSIVE
        assert isinstance(preset, dict)
        assert preset["spike_detection_mode"] == "or"
        assert preset["tau_s"] == 0.08
