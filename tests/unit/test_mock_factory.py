"""Tests for mock factory utilities"""
import pytest
from insightspike.utils.mock_factory import create_mock_components


def test_create_mock_components():
    """Test mock components creation."""
    MockGED, MockIG, MockDetector = create_mock_components()

    # Test MockGraphEditDistance
    ged = MockGED()
    assert hasattr(ged, "calculate_distance")
    assert ged.calculate_distance(None, None) == 1.0

    # Test MockInformationGain
    ig = MockIG()
    assert hasattr(ig, "calculate_gain")
    assert ig.calculate_gain() == 0.5
    assert ig.calculate_gain("arg1", "arg2") == 0.5

    # Test MockInsightDetector
    detector = MockDetector()
    assert hasattr(detector, "detect_insights")
    assert detector.detect_insights() == []
    assert detector.detect_insights("arg1", "arg2") == []


def test_mock_components_consistency():
    """Test that mock components are consistent across calls."""
    MockGED1, MockIG1, MockDetector1 = create_mock_components()
    MockGED2, MockIG2, MockDetector2 = create_mock_components()

    # Test that they have the same functionality
    ged1 = MockGED1()
    ged2 = MockGED2()
    assert ged1.calculate_distance(None, None) == ged2.calculate_distance(None, None)

    ig1 = MockIG1()
    ig2 = MockIG2()
    assert ig1.calculate_gain() == ig2.calculate_gain()
