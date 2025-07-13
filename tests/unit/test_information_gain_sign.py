"""
Test Information Gain sign convention
====================================

Verifies that the sign convention is correct:
- Positive ΔIG means information gain (entropy reduction)
- Negative ΔIG means information loss (entropy increase)
"""

import pytest
import numpy as np
from insightspike.algorithms.information_gain import InformationGain, compute_delta_ig


class TestInformationGainSignConvention:
    """Test that IG calculations follow correct sign convention."""

    def test_entropy_reduction_gives_positive_ig(self):
        """Test that reducing entropy gives positive IG."""
        # High entropy state (uniform distribution)
        data_before = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Low entropy state (concentrated distribution)
        data_after = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        calculator = InformationGain(method="shannon")
        result = calculator.calculate(data_before, data_after)

        # Information gain should be positive (entropy decreased)
        assert result.ig_value > 0
        assert result.entropy_before > result.entropy_after

    def test_entropy_increase_gives_negative_ig(self):
        """Test that increasing entropy gives negative IG."""
        # Low entropy state (concentrated distribution)
        data_before = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        # High entropy state (uniform distribution)
        data_after = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        calculator = InformationGain(method="shannon")
        result = calculator.calculate(data_before, data_after)

        # Information gain should be negative (entropy increased)
        assert result.ig_value < 0
        assert result.entropy_before < result.entropy_after

    def test_no_change_gives_zero_ig(self):
        """Test that no change gives zero IG."""
        data = np.array([1, 2, 3, 4, 5])

        calculator = InformationGain(method="shannon")
        result = calculator.calculate(data, data)

        # Information gain should be zero (no change)
        assert abs(result.ig_value) < 1e-10
        assert abs(result.entropy_before - result.entropy_after) < 1e-10

    def test_clustering_method_sign_convention(self):
        """Test sign convention with clustering method."""
        # Create low-dimensional data for easier clustering
        np.random.seed(42)

        # High entropy: scattered points
        data_before = np.random.randn(50, 2) * 5

        # Low entropy: clustered points
        cluster1 = np.random.randn(25, 2) * 0.5 + [5, 5]
        cluster2 = np.random.randn(25, 2) * 0.5 + [-5, -5]
        data_after = np.vstack([cluster1, cluster2])

        calculator = InformationGain(method="clustering", k_clusters=4)
        result = calculator.calculate(data_before, data_after)

        # Information gain should be positive (better clustering = lower entropy)
        assert result.ig_value > 0

    def test_convenience_function_sign_convention(self):
        """Test that convenience function follows correct sign convention."""
        # High entropy to low entropy
        state_before = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        state_after = np.array([1, 1, 1, 1, 2, 2, 2, 2])

        delta_ig = compute_delta_ig(state_before, state_after, method="shannon")

        # Should be positive (information gain)
        assert delta_ig > 0

    def test_insight_detection_threshold(self):
        """Test that typical insight detection threshold works correctly."""
        # Create a scenario that should trigger insight detection
        # High entropy state
        state_before = np.random.randint(0, 10, 100)

        # Low entropy state (simulating insight/understanding)
        state_after = np.array([1] * 50 + [2] * 50)

        calculator = InformationGain(method="shannon")
        result = calculator.calculate(state_before, state_after)

        # For insight detection, we expect ΔIG ≥ 0.2
        # This should be a significant entropy reduction
        assert result.ig_value > 0.2
        assert result.is_significant


def test_docstring_example():
    """Test the example from the module docstring."""
    # According to the docstring:
    # ΔIG ≥ 0.2 threshold typically indicates EurekaSpike

    # Create a scenario with significant information gain
    # Random data (high entropy)
    before = np.random.choice(10, 100)

    # Organized data (low entropy)
    after = np.array([0] * 33 + [1] * 33 + [2] * 34)

    calculator = InformationGain(method="shannon")
    delta_ig = calculator.compute_delta_ig(before, after)

    # Should show significant information gain
    assert delta_ig > 0  # Positive means information gain
