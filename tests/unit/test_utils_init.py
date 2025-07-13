"""Tests for utils __init__ module"""
from unittest.mock import MagicMock, patch

import pytest

from insightspike import utils


def test_utils_imports():
    """Test utils module imports."""
    # Test that expected modules are available
    assert hasattr(utils, "clean_text")
    assert hasattr(utils, "iter_text")
    assert hasattr(utils, "embedder")

    # Test functions work
    assert utils.clean_text("  test  ") == "test"

    # Test get_available_models
    assert hasattr(utils, "get_available_models")
    models = utils.get_available_models()
    assert isinstance(models, list)


def test_utils_platform_check():
    """Test platform utilities."""
    from insightspike.utils import platform_utils

    # Should have platform check functions
    assert hasattr(platform_utils, "is_macos")
    assert hasattr(platform_utils, "is_linux")
    assert hasattr(platform_utils, "is_windows")

    # One of them should be True
    platforms = [
        platform_utils.is_macos(),
        platform_utils.is_linux(),
        platform_utils.is_windows(),
    ]
    assert any(platforms)


def test_monitoring_module():
    """Test monitoring module."""
    from insightspike import monitoring

    # Should have graph monitor
    assert hasattr(monitoring, "GraphOperationMonitor")
    assert hasattr(monitoring, "MonitoredOperation")
