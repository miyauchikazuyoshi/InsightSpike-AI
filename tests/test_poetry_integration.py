"""
Tests for Poetry integration module - Fixed version
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

from insightspike.utils.poetry_integration import (
    PoetryIntegration,
    PoetryNotFoundError,
    PyprojectTomlError,
    DependencyInstallationError,
)
from insightspike.utils.dependency_resolver import DependencySpec, ResolvedDependency
from insightspike.utils.platform_utils import PlatformInfo


class TestPoetryIntegration:
    """Test PoetryIntegration class."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_pyproject_toml(self):
        """Sample pyproject.toml content."""
        return {
            "tool": {
                "poetry": {
                    "name": "test-project",
                    "version": "0.1.0",
                    "description": "Test project",
                    "dependencies": {"python": "^3.8", "numpy": "^1.20.0"},
                    "group": {"dev": {"dependencies": {"pytest": "^7.0.0"}}},
                }
            }
        }

    @pytest.fixture
    def platform_info(self):
        """Mock platform info."""
        return PlatformInfo(
            platform="linux",
            architecture="x64",
            python_version="3.9.0",
            gpu_available=True,
        )

    def test_poetry_integration_init(self, platform_info):
        """Test PoetryIntegration initialization."""
        integration = PoetryIntegration()

        assert integration.dependency_resolver is not None
        assert hasattr(integration, "export_requirements")

    def test_export_requirements(self, platform_info):
        """Test exporting requirements format."""
        integration = PoetryIntegration()

        # Mock the dependency resolver
        mock_deps = [
            ResolvedDependency(
                name="torch", version="2.0.0", extras=["cuda"], platform_specific=True
            ),
            ResolvedDependency(
                name="numpy", version="1.24.0", extras=[], platform_specific=False
            ),
        ]

        with patch.object(
            integration.dependency_resolver,
            "resolve_dependencies",
            return_value=mock_deps,
        ):
            requirements = integration.export_requirements(platform_info)

            assert "torch[cuda]==2.0.0" in requirements
            assert "numpy==1.24.0" in requirements
            assert (
                f"# Generated requirements for {platform_info.platform}" in requirements
            )

    def test_get_current_dependencies_success(self):
        """Test getting current dependencies successfully."""
        integration = PoetryIntegration()

        mock_output = "torch 2.0.0\nnumpy 1.24.0\nscipy 1.10.0"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = mock_output
            mock_run.return_value.returncode = 0

            deps = integration.get_current_dependencies()

            assert "torch" in deps
            assert deps["torch"] == "2.0.0"
            assert "numpy" in deps
            assert deps["numpy"] == "1.24.0"
