"""Tests for mock factory utilities"""
import pytest
from insightspike.utils.mock_factory import create_mock_components


def test_create_mock_components():
    """Test mock components creation."""
    MockGED, MockIG, MockDetector = create_mock_components()
    
    # Test MockGraphEditDistance
    ged = MockGED()
    assert hasattr(ged, 'calculate_distance')
    assert ged.calculate_distance(None, None) == 1.0
    
    # Test MockInformationGain
    ig = MockIG()
    assert hasattr(ig, 'calculate_gain')
    assert ig.calculate_gain() == 0.5
    assert ig.calculate_gain("arg1", "arg2") == 0.5
    
    # Test MockInsightDetector
    detector = MockDetector()
    assert hasattr(detector, 'detect_insights')
    assert detector.detect_insights() == []
    assert detector.detect_insights("arg1", "arg2") == []


def test_mock_components_consistency():
    """Test that mock components are consistent across calls."""
    MockGED1, MockIG1, MockDetector1 = create_mock_components()
    MockGED2, MockIG2, MockDetector2 = create_mock_components()
    
    # Classes should be the same
    assert MockGED1 == MockGED2
    assert MockIG1 == MockIG2
    assert MockDetector1 == MockDetector2