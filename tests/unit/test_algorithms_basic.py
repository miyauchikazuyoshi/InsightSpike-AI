"""Basic tests for algorithm modules to improve coverage"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from insightspike.algorithms import CalculateDeltaGED, CalculateDeltaIG


def test_calculate_delta_ged_init():
    """Test CalculateDeltaGED initialization."""
    calc = CalculateDeltaGED()
    assert calc is not None
    assert hasattr(calc, 'calculate')


def test_calculate_delta_ig_init():
    """Test CalculateDeltaIG initialization."""
    calc = CalculateDeltaIG()
    assert calc is not None
    assert hasattr(calc, 'calculate')


def test_algorithms_module_imports():
    """Test algorithm module level imports."""
    import insightspike.algorithms as alg
    
    # Test convenience functions exist
    assert hasattr(alg, 'calculate_delta_ged')
    assert hasattr(alg, 'calculate_delta_ig')
    
    # Test they are callable
    assert callable(alg.calculate_delta_ged)
    assert callable(alg.calculate_delta_ig)