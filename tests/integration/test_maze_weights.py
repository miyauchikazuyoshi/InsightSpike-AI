"""
Integration test for maze navigation with vector weights.
Demonstrates the improvement from applying dimension-specific weights.
"""

import numpy as np
from insightspike.config.vector_weights import VectorWeightConfig
from insightspike.core.weight_vector_manager import WeightVectorManager


def test_maze_navigation_improvement():
    """Test that weights improve maze navigation similarity calculation."""
    # Baseline (weights disabled)
    config_off = VectorWeightConfig(enabled=False)
    manager_off = WeightVectorManager(config_off)
    
    # With weights (direction scaled down)
    config_on = VectorWeightConfig(
        enabled=True,
        weights=[1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3]
    )
    manager_on = WeightVectorManager(config_on)
    
    # Two vectors: nearby positions, opposite directions
    vec1 = np.array([0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0, 0.3])
    vec2 = np.array([0.51, 0.51, -1.0, 0.0, 1.0, 0.0, 0.0, 0.28])
    
    # Without weights: direction dominates
    unweighted1 = manager_off.apply_weights(vec1)
    unweighted2 = manager_off.apply_weights(vec2)
    cosine_off = np.dot(unweighted1, unweighted2) / (
        np.linalg.norm(unweighted1) * np.linalg.norm(unweighted2)
    )
    
    # With weights: position matters more
    weighted1 = manager_on.apply_weights(vec1)
    weighted2 = manager_on.apply_weights(vec2)
    cosine_on = np.dot(weighted1, weighted2) / (
        np.linalg.norm(weighted1) * np.linalg.norm(weighted2)
    )
    
    # Weighted should show higher similarity (nearby positions)
    assert cosine_on > cosine_off
    
    print(f"Similarity without weights: {cosine_off:.3f}")
    print(f"Similarity with weights: {cosine_on:.3f}")
    print(f"Improvement: {(cosine_on - cosine_off) / cosine_off * 100:.1f}%")


def test_direction_influence_reduction():
    """Test that direction influence is reduced with weights."""
    config = VectorWeightConfig(
        enabled=True,
        weights=[1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3]
    )
    manager = WeightVectorManager(config)
    
    # Three episodes at similar positions, different directions
    right = np.array([0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0, 0.3])
    up = np.array([0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.3])
    left = np.array([0.5, 0.5, -1.0, 0.0, 1.0, 0.0, 0.0, 0.3])
    
    # Apply weights
    w_right = manager.apply_weights(right)
    w_up = manager.apply_weights(up)
    w_left = manager.apply_weights(left)
    
    # Calculate similarities
    sim_right_up = np.dot(w_right, w_up) / (np.linalg.norm(w_right) * np.linalg.norm(w_up))
    sim_right_left = np.dot(w_right, w_left) / (np.linalg.norm(w_right) * np.linalg.norm(w_left))
    sim_up_left = np.dot(w_up, w_left) / (np.linalg.norm(w_up) * np.linalg.norm(w_left))
    
    # All should have high similarity despite different directions
    assert sim_right_up > 0.9
    assert sim_right_left > 0.9
    assert sim_up_left > 0.9
    
    print(f"Right-Up similarity: {sim_right_up:.3f}")
    print(f"Right-Left similarity: {sim_right_left:.3f}")
    print(f"Up-Left similarity: {sim_up_left:.3f}")


def test_position_distance_preservation():
    """Test that position distance is preserved with weights."""
    config = VectorWeightConfig(
        enabled=True,
        weights=[2.0, 2.0, 0.05, 0.05, 0.3, 0.3, 0.3, 0.1]  # Aggressive position focus
    )
    manager = WeightVectorManager(config)
    
    # Near position
    near = np.array([0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0, 0.3])
    # Far position  
    far = np.array([0.1, 0.1, 1.0, 0.0, 1.0, 0.0, 0.0, 0.7])
    # Query position
    query = np.array([0.51, 0.51, 0.0, 0.0, 0.5, 0.5, 0.0, 0.25])
    
    # Apply weights
    w_near = manager.apply_weights(near)
    w_far = manager.apply_weights(far)
    w_query = manager.apply_weights(query)
    
    # Calculate similarities
    sim_near = np.dot(w_query, w_near) / (np.linalg.norm(w_query) * np.linalg.norm(w_near))
    sim_far = np.dot(w_query, w_far) / (np.linalg.norm(w_query) * np.linalg.norm(w_far))
    
    # Near should have much higher similarity
    assert sim_near > sim_far
    assert sim_near - sim_far > 0.1  # Significant difference
    
    print(f"Near similarity: {sim_near:.3f}")
    print(f"Far similarity: {sim_far:.3f}")
    print(f"Difference: {sim_near - sim_far:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Maze Navigation Weight Tests")
    print("=" * 60)
    
    print("\nTest 1: Overall Improvement")
    print("-" * 40)
    test_maze_navigation_improvement()
    
    print("\nTest 2: Direction Influence Reduction")
    print("-" * 40)
    test_direction_influence_reduction()
    
    print("\nTest 3: Position Distance Preservation")
    print("-" * 40)
    test_position_distance_preservation()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)