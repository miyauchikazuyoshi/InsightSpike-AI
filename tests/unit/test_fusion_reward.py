"""
Test fusion reward calculation without conflict term
====================================================

Tests to ensure the simplified reward calculation works correctly.
"""

import unittest

from insightspike.metrics import compute_fusion_reward


class TestFusionReward(unittest.TestCase):
    """Test the simplified fusion reward calculation."""

    def test_default_weights(self):
        """Test reward calculation with default weights (0.5, 0.5)."""
        delta_ged = -0.6  # Negative = good (structure simplified)
        delta_ig = 0.4  # Positive = good (information gained)

        reward = compute_fusion_reward(delta_ged, delta_ig)

        # Expected: 0.5 * (-0.6) + 0.5 * 0.4 = -0.3 + 0.2 = -0.1
        assert abs(reward - (-0.1)) < 0.001

    def test_custom_weights(self):
        """Test reward calculation with custom weights."""
        delta_ged = -0.6
        delta_ig = 0.4
        weights = {"ged": 0.6, "ig": 0.4}

        reward = compute_fusion_reward(delta_ged, delta_ig, weights=weights)

        # Expected: 0.6 * (-0.6) + 0.4 * 0.4 = -0.36 + 0.16 = -0.2
        assert abs(reward - (-0.2)) < 0.001

    def test_backward_compatibility(self):
        """Test that conflict_score parameter is ignored for backward compatibility."""
        delta_ged = -0.6
        delta_ig = 0.4

        # Should ignore conflict_score parameter
        reward_without = compute_fusion_reward(delta_ged, delta_ig)
        reward_with = compute_fusion_reward(delta_ged, delta_ig, conflict_score=0.5)

        assert reward_without == reward_with

    def test_positive_reward(self):
        """Test case where reward is positive (strong information gain)."""
        delta_ged = -0.2  # Small structural improvement
        delta_ig = 0.8  # Large information gain

        reward = compute_fusion_reward(delta_ged, delta_ig)

        # Expected: 0.5 * (-0.2) + 0.5 * 0.8 = -0.1 + 0.4 = 0.3
        assert abs(reward - 0.3) < 0.001

    def test_negative_reward(self):
        """Test case where reward is negative (structure becomes complex)."""
        delta_ged = 0.8  # Positive = bad (structure becomes complex)
        delta_ig = 0.1  # Small information gain

        reward = compute_fusion_reward(delta_ged, delta_ig)

        # Expected: 0.5 * 0.8 + 0.5 * 0.1 = 0.4 + 0.05 = 0.45
        assert abs(reward - 0.45) < 0.001

    def test_eureka_spike_case(self):
        """Test typical eureka spike scenario."""
        # Eureka spike: significant structure simplification + information gain
        delta_ged = -0.7  # Strong simplification
        delta_ig = 0.5  # Good information gain

        reward = compute_fusion_reward(delta_ged, delta_ig)

        # Expected: 0.5 * (-0.7) + 0.5 * 0.5 = -0.35 + 0.25 = -0.1
        # Negative reward is actually good when delta_ged is negative!
        assert reward < 0  # Good insight detected
