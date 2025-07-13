"""Basic tests for algorithm modules to improve coverage"""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from insightspike.algorithms import GraphEditDistance, InformationGain


def test_graph_edit_distance_init():
    """Test GraphEditDistance initialization."""
    calc = GraphEditDistance()
    assert calc is not None
    assert hasattr(calc, "compute")


def test_information_gain_init():
    """Test InformationGain initialization."""
    calc = InformationGain()
    assert calc is not None
    assert hasattr(calc, "compute")


def test_algorithms_module_imports():
    """Test algorithm module level imports."""
    import insightspike.algorithms as alg

    # Test convenience functions exist
    assert hasattr(alg, "compute_delta_ged")
    assert hasattr(alg, "compute_delta_ig")

    # Test they are callable
    assert callable(alg.compute_delta_ged)
    assert callable(alg.compute_delta_ig)
