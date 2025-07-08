"""Factory for creating mock components for testing and CI."""


def create_mock_components():
    """Create mock components for CI/LITE mode compatibility."""

    class MockGraphEditDistance:
        def calculate_distance(self, g1, g2):
            return 1.0

    class MockInformationGain:
        def calculate_gain(self, *args):
            return 0.5

    class MockInsightDetector:
        def detect_insights(self, *args):
            return []

    return MockGraphEditDistance, MockInformationGain, MockInsightDetector
