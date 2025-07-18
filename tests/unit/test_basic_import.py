"""Basic import test to diagnose CI issues."""

def test_basic_import():
    """Test basic Python import."""
    import sys
    print(f"Python path in test: {sys.path}")
    assert True

def test_insightspike_import():
    """Test insightspike package import."""
    import insightspike
    assert insightspike is not None