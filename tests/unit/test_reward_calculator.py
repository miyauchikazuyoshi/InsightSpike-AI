"""
Unit tests for RewardCalculator
===============================

Tests reward calculation functionality including:
- Base reward calculation with weights
- Structure reward based on graph size
- Novelty reward calculation
- Configuration handling
- Edge cases
"""

from unittest.mock import Mock, patch

import pytest

from insightspike.config.constants import Defaults
from insightspike.config.models import InsightSpikeConfig
from insightspike.features.graph_reasoning.reward_calculator import RewardCalculator


class TestRewardCalculatorInitialization:
    """Test RewardCalculator initialization with different configurations."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        calculator = RewardCalculator()
        
        assert calculator.weights["ged"] == Defaults.REWARD_WEIGHT_GED
        assert calculator.weights["ig"] == Defaults.REWARD_WEIGHT_IG
        assert calculator.optimal_graph_size == Defaults.OPTIMAL_GRAPH_SIZE

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = Mock()
        config.graph = Mock()
        config.graph.weight_ged = 0.7
        config.graph.weight_ig = 0.3
        config.graph.optimal_graph_size = 200
        
        calculator = RewardCalculator(config)
        
        assert calculator.weights["ged"] == 0.7
        assert calculator.weights["ig"] == 0.3
        assert calculator.optimal_graph_size == 200

    def test_init_with_partial_config(self):
        """Test initialization with partial configuration (missing some values)."""
        config = Mock()
        config.graph = Mock()
        config.graph.weight_ged = 0.8
        # Missing weight_ig and optimal_graph_size
        config.graph.weight_ig = None
        config.graph.optimal_graph_size = None
        
        # Use getattr to simulate missing attributes
        def custom_getattr(obj, name, default):
            value = getattr(obj, name, None)
            return value if value is not None else default
        
        with patch("insightspike.features.graph_reasoning.reward_calculator.getattr", custom_getattr):
            calculator = RewardCalculator(config)
        
        assert calculator.weights["ged"] == 0.8
        # Should fall back to defaults for missing values
        assert calculator.weights["ig"] == Defaults.REWARD_WEIGHT_IG

    def test_init_without_graph_config(self):
        """Test initialization when config has no graph attribute."""
        config = Mock()
        del config.graph  # Simulate missing graph attribute
        
        calculator = RewardCalculator(config)
        
        # Should use all defaults
        assert calculator.weights["ged"] == Defaults.REWARD_WEIGHT_GED
        assert calculator.weights["ig"] == Defaults.REWARD_WEIGHT_IG
        assert calculator.optimal_graph_size == Defaults.OPTIMAL_GRAPH_SIZE


class TestBaseRewardCalculation:
    """Test base reward calculation functionality."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with known weights."""
        config = Mock()
        config.graph = Mock()
        config.graph.weight_ged = 0.6
        config.graph.weight_ig = 0.4
        config.graph.optimal_graph_size = 150
        
        return RewardCalculator(config)

    def test_calculate_reward_basic(self, calculator):
        """Test basic reward calculation."""
        metrics = {"delta_ged": -0.5, "delta_ig": 0.3}  # GED is negative for improvement
        conflicts = {"total": 0.2}
        
        reward = calculator.calculate_reward(metrics, conflicts)
        
        # Base reward = 0.6 * (-0.5) + 0.4 * 0.3 = -0.3 + 0.12 = -0.18
        expected_base = 0.6 * (-0.5) + 0.4 * 0.3
        assert abs(reward["base"] - expected_base) < 0.001
        
        assert "structure" in reward
        assert "novelty" in reward
        assert "total" in reward

    def test_calculate_reward_positive(self, calculator):
        """Test reward calculation with positive improvements."""
        metrics = {"delta_ged": -0.8, "delta_ig": 0.6}
        conflicts = {"total": 0.1}
        
        reward = calculator.calculate_reward(metrics, conflicts)
        
        # Base reward = 0.6 * (-0.8) + 0.4 * 0.6 = -0.48 + 0.24 = -0.24
        expected_base = 0.6 * (-0.8) + 0.4 * 0.6
        assert abs(reward["base"] - expected_base) < 0.001

    def test_calculate_reward_missing_metrics(self, calculator):
        """Test reward calculation with missing metrics."""
        # Missing delta_ged
        metrics1 = {"delta_ig": 0.5}
        reward1 = calculator.calculate_reward(metrics1, {})
        assert reward1["base"] == 0.4 * 0.5  # Only IG component
        
        # Missing delta_ig
        metrics2 = {"delta_ged": -0.3}
        reward2 = calculator.calculate_reward(metrics2, {})
        assert reward2["base"] == 0.6 * (-0.3)  # Only GED component
        
        # Missing both
        metrics3 = {}
        reward3 = calculator.calculate_reward(metrics3, {})
        assert reward3["base"] == 0.0

    def test_calculate_reward_zero_metrics(self, calculator):
        """Test reward calculation with zero metrics."""
        metrics = {"delta_ged": 0.0, "delta_ig": 0.0}
        conflicts = {"total": 0.0}
        
        reward = calculator.calculate_reward(metrics, conflicts)
        
        assert reward["base"] == 0.0
        assert reward["novelty"] == 0.0


class TestStructureReward:
    """Test structure reward calculation."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with specific optimal size."""
        config = Mock()
        config.graph = Mock()
        config.graph.optimal_graph_size = 100
        config.graph.weight_ged = 0.5
        config.graph.weight_ig = 0.5
        
        return RewardCalculator(config)

    def test_structure_reward_optimal_size(self, calculator):
        """Test structure reward at optimal size."""
        metrics = {"graph_size_current": 100}  # Exactly optimal
        
        structure_reward = calculator._calculate_structure_reward(metrics)
        
        assert structure_reward == 1.0  # Maximum reward

    def test_structure_reward_near_optimal(self, calculator):
        """Test structure reward near optimal size."""
        # 10% deviation
        metrics1 = {"graph_size_current": 110}
        reward1 = calculator._calculate_structure_reward(metrics1)
        assert 0.9 <= reward1 < 1.0
        
        metrics2 = {"graph_size_current": 90}
        reward2 = calculator._calculate_structure_reward(metrics2)
        assert 0.9 <= reward2 < 1.0

    def test_structure_reward_far_from_optimal(self, calculator):
        """Test structure reward far from optimal size."""
        # 50% deviation
        metrics1 = {"graph_size_current": 150}
        reward1 = calculator._calculate_structure_reward(metrics1)
        assert reward1 == 0.5
        
        # 100% deviation
        metrics2 = {"graph_size_current": 200}
        reward2 = calculator._calculate_structure_reward(metrics2)
        assert reward2 == 0.0
        
        # More than 100% deviation
        metrics3 = {"graph_size_current": 250}
        reward3 = calculator._calculate_structure_reward(metrics3)
        assert reward3 == 0.0  # Clamped to 0

    def test_structure_reward_zero_size(self, calculator):
        """Test structure reward with zero graph size."""
        metrics = {"graph_size_current": 0}
        
        reward = calculator._calculate_structure_reward(metrics)
        
        assert reward == 0.0

    def test_structure_reward_missing_size(self, calculator):
        """Test structure reward with missing size metric."""
        metrics = {"other_metric": 42}
        
        reward = calculator._calculate_structure_reward(metrics)
        
        assert reward == 0.0


class TestNoveltyReward:
    """Test novelty reward calculation."""

    @pytest.fixture
    def calculator(self):
        """Create a basic calculator."""
        return RewardCalculator()

    def test_novelty_reward_positive(self, calculator):
        """Test novelty reward with positive information gain."""
        metrics = {"delta_ig": 0.7}
        conflicts = {"total": 0.2}
        
        novelty = calculator._calculate_novelty_reward(metrics, conflicts)
        
        assert novelty == 0.7

    def test_novelty_reward_zero(self, calculator):
        """Test novelty reward with zero information gain."""
        metrics = {"delta_ig": 0.0}
        conflicts = {"total": 0.5}
        
        novelty = calculator._calculate_novelty_reward(metrics, conflicts)
        
        assert novelty == 0.0

    def test_novelty_reward_negative(self, calculator):
        """Test novelty reward with negative information gain."""
        metrics = {"delta_ig": -0.3}
        conflicts = {"total": 0.1}
        
        novelty = calculator._calculate_novelty_reward(metrics, conflicts)
        
        assert novelty == 0.0  # Clamped to 0

    def test_novelty_reward_missing_ig(self, calculator):
        """Test novelty reward with missing information gain."""
        metrics = {"other_metric": 1.0}
        conflicts = {"total": 0.0}
        
        novelty = calculator._calculate_novelty_reward(metrics, conflicts)
        
        assert novelty == 0.0

    def test_novelty_reward_backward_compatibility(self, calculator):
        """Test that conflicts parameter is ignored (backward compatibility)."""
        metrics = {"delta_ig": 0.5}
        
        # Same result regardless of conflicts
        novelty1 = calculator._calculate_novelty_reward(metrics, {"total": 0.0})
        novelty2 = calculator._calculate_novelty_reward(metrics, {"total": 1.0})
        novelty3 = calculator._calculate_novelty_reward(metrics, None)
        
        assert novelty1 == novelty2 == novelty3 == 0.5


class TestTotalRewardCalculation:
    """Test total reward calculation combining all components."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with balanced weights."""
        config = Mock()
        config.graph = Mock()
        config.graph.weight_ged = 0.5
        config.graph.weight_ig = 0.5
        config.graph.optimal_graph_size = 100
        
        return RewardCalculator(config)

    def test_total_reward_all_positive(self, calculator):
        """Test total reward with all positive components."""
        metrics = {
            "delta_ged": -0.6,  # Good improvement
            "delta_ig": 0.8,    # High information gain
            "graph_size_current": 100  # Optimal size
        }
        conflicts = {"total": 0.1}
        
        reward = calculator.calculate_reward(metrics, conflicts)
        
        # Calculate expected values
        expected_base = 0.5 * (-0.6) + 0.5 * 0.8  # = 0.1
        expected_structure = 1.0  # Optimal size
        expected_novelty = 0.8    # = delta_ig
        expected_total = expected_base + expected_structure + expected_novelty
        
        assert abs(reward["base"] - expected_base) < 0.001
        assert reward["structure"] == expected_structure
        assert reward["novelty"] == expected_novelty
        assert abs(reward["total"] - expected_total) < 0.001

    def test_total_reward_mixed(self, calculator):
        """Test total reward with mixed positive/negative components."""
        metrics = {
            "delta_ged": 0.2,   # Worse (positive GED is bad)
            "delta_ig": 0.3,    # Some information gain
            "graph_size_current": 150  # 50% off optimal
        }
        conflicts = {"total": 0.5}
        
        reward = calculator.calculate_reward(metrics, conflicts)
        
        expected_base = 0.5 * 0.2 + 0.5 * 0.3  # = 0.25
        expected_structure = 0.5  # 50% deviation
        expected_novelty = 0.3
        expected_total = expected_base + expected_structure + expected_novelty
        
        assert abs(reward["total"] - expected_total) < 0.001

    def test_total_reward_all_zero(self, calculator):
        """Test total reward when all components are zero."""
        metrics = {
            "delta_ged": 0.0,
            "delta_ig": 0.0,
            "graph_size_current": 0
        }
        conflicts = {"total": 0.0}
        
        reward = calculator.calculate_reward(metrics, conflicts)
        
        assert reward["base"] == 0.0
        assert reward["structure"] == 0.0
        assert reward["novelty"] == 0.0
        assert reward["total"] == 0.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def calculator(self):
        """Create a basic calculator."""
        return RewardCalculator()

    def test_reward_with_nan_values(self, calculator):
        """Test reward calculation with NaN values."""
        import numpy as np
        
        metrics = {
            "delta_ged": np.nan,
            "delta_ig": 0.5,
            "graph_size_current": 100
        }
        
        # Should handle NaN gracefully
        reward = calculator.calculate_reward(metrics, {})
        
        # NaN values should result in NaN in calculations
        assert np.isnan(reward["base"])

    def test_reward_with_inf_values(self, calculator):
        """Test reward calculation with infinite values."""
        import numpy as np
        
        metrics = {
            "delta_ged": -np.inf,
            "delta_ig": np.inf,
            "graph_size_current": 100
        }
        
        reward = calculator.calculate_reward(metrics, {})
        
        # Should produce NaN when mixing inf values
        # The calculation -inf * weight + inf * weight = NaN
        assert np.isnan(reward["base"])

    def test_reward_with_very_large_values(self, calculator):
        """Test reward calculation with very large values."""
        metrics = {
            "delta_ged": -1e10,
            "delta_ig": 1e10,
            "graph_size_current": 1e10
        }
        
        reward = calculator.calculate_reward(metrics, {})
        
        # Should handle large values
        assert isinstance(reward["base"], float)
        assert isinstance(reward["total"], float)

    def test_reward_type_consistency(self, calculator):
        """Test that all reward values are floats."""
        metrics = {
            "delta_ged": -0.5,
            "delta_ig": 0.3,
            "graph_size_current": 100
        }
        
        reward = calculator.calculate_reward(metrics, {})
        
        # All values should be float type
        assert isinstance(reward["base"], float)
        assert isinstance(reward["structure"], float)
        assert isinstance(reward["novelty"], float)
        assert isinstance(reward["total"], float)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_insight_spike_scenario(self):
        """Test reward calculation for an insight spike scenario."""
        calculator = RewardCalculator()
        
        # Metrics indicating an insight spike
        metrics = {
            "delta_ged": -0.7,  # Significant structural improvement
            "delta_ig": 0.8,    # High information gain
            "graph_size_current": 150  # Reasonable graph size
        }
        conflicts = {"total": 0.1}  # Low conflict
        
        reward = calculator.calculate_reward(metrics, conflicts)
        
        # Should produce positive total reward
        # The exact value depends on the optimal graph size
        assert reward["total"] > 0.5
        assert reward["novelty"] == 0.8
        assert reward["base"] < 0  # GED component dominates

    def test_poor_performance_scenario(self):
        """Test reward calculation for poor performance."""
        calculator = RewardCalculator()
        
        # Metrics indicating poor performance
        metrics = {
            "delta_ged": 0.5,   # Structural degradation
            "delta_ig": -0.2,   # Information loss
            "graph_size_current": 500  # Too large
        }
        conflicts = {"total": 0.8}  # High conflict
        
        reward = calculator.calculate_reward(metrics, conflicts)
        
        # Should produce low or negative total reward
        assert reward["base"] > 0  # Positive due to positive GED
        assert reward["novelty"] == 0.0  # Clamped
        assert reward["structure"] == 0.0  # Far from optimal

    def test_config_presets_compatibility(self, config_experiment):
        """Test compatibility with different config presets."""
        # Test with experiment config
        calculator = RewardCalculator(config_experiment)
        
        metrics = {"delta_ged": -0.5, "delta_ig": 0.5}
        reward = calculator.calculate_reward(metrics, {})
        
        assert "total" in reward
        assert isinstance(reward["total"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])