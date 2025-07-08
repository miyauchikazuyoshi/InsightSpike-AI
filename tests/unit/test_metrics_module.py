"""Tests for metrics module to improve coverage"""
import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from insightspike.metrics import (
    GraphEditDistance, 
    InformationGain, 
    ConflictDetector,
    calculate_conflict,
    calculate_structural_conflict,
    calculate_semantic_conflict,
    calculate_temporal_conflict
)


def test_graph_edit_distance_class():
    """Test GraphEditDistance class."""
    ged = GraphEditDistance()
    assert ged is not None
    assert hasattr(ged, 'calculate')


def test_information_gain_class():
    """Test InformationGain class."""
    ig = InformationGain()
    assert ig is not None
    assert hasattr(ig, 'calculate')


def test_conflict_detector_init():
    """Test ConflictDetector initialization."""
    detector = ConflictDetector()
    assert detector is not None
    assert hasattr(detector, 'detect_conflicts')


def test_calculate_functions_exist():
    """Test that calculate functions are available."""
    # Test they exist and are callable
    assert callable(calculate_conflict)
    assert callable(calculate_structural_conflict)
    assert callable(calculate_semantic_conflict)
    assert callable(calculate_temporal_conflict)


def test_metrics_convenience_imports():
    """Test metrics module convenience imports."""
    import insightspike.metrics as metrics
    
    # Should have these classes
    assert hasattr(metrics, 'GraphEditDistance')
    assert hasattr(metrics, 'InformationGain')
    assert hasattr(metrics, 'ConflictDetector')
    
    # Should have delta calculation functions
    assert hasattr(metrics, 'calculate_delta_ged')
    assert hasattr(metrics, 'calculate_delta_ig')