"""
Test layer1_error_monitor functionality
"""
import pytest
from insightspike.core.layers.layer1_error_monitor import uncertainty

def test_uncertainty():
    """Test uncertainty calculation"""
    # Test with diverse probabilities (should have high uncertainty)
    diverse_probs = [0.2, 0.8]
    val = uncertainty(diverse_probs)
    assert val > 0
    
    # Test with uniform probabilities (should have even higher uncertainty)  
    uniform_probs = [0.5, 0.5]
    val2 = uncertainty(uniform_probs)
    assert val2 > 0
    
    # Test with single probability (should have low uncertainty)
    single_prob = [1.0]
    val3 = uncertainty(single_prob)
    assert val3 >= 0
