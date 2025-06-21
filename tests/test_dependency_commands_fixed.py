"""
Tests for CLI dependency commands module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import click
from click.testing import CliRunner

from insightspike.cli.dependency_commands import (
    platform_info,
    list_deps,
    add
)
from insightspike.utils.platform_utils import PlatformInfo


class TestPlatformInfoCommand:
    """Test platform info command."""
    
    def test_platform_info_command(self):
        """Test platform info command execution."""
        runner = CliRunner()
        
        with patch('insightspike.cli.dependency_commands.PlatformDetector') as mock_detector:
            mock_platform_info = Mock(spec=PlatformInfo)
            mock_platform_info.platform = "test_platform"
            mock_platform_info.architecture = "test_arch"
            mock_platform_info.gpu_available = True
            mock_platform_info.python_version = "3.11.0"
            
            mock_detector.return_value.detect_platform.return_value = mock_platform_info
            
            result = runner.invoke(platform_info)
            
            assert result.exit_code == 0
            assert "Platform: test_platform" in result.output
            assert "Architecture: test_arch" in result.output
            assert "GPU Available: True" in result.output
            assert "Python Version: 3.11.0" in result.output


class TestListDepsCommand:
    """Test list-deps command."""
    
    def test_list_deps_current_platform(self):
        """Test listing dependencies for current platform."""
        runner = CliRunner()
        
        mock_deps = {
            "test_package": {
                "version": "1.0.0", 
                "description": "Test package description"
            }
        }
        
        with patch('insightspike.cli.dependency_commands.PlatformDetector') as mock_detector, \
             patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            
            mock_platform_info = Mock(spec=PlatformInfo)
            mock_platform_info.platform = "test_platform"
            mock_detector.return_value.detect_platform.return_value = mock_platform_info
            
            mock_resolver.return_value.get_platform_dependencies.return_value = mock_deps
            
            result = runner.invoke(list_deps)
            
            assert result.exit_code == 0
            assert "Dependencies for test_platform:" in result.output
            assert "test_package (1.0.0): Test package description" in result.output


class TestAddCommand:
    """Test add command."""
    
    def test_add_package_without_version(self):
        """Test adding a package without specific version."""
        runner = CliRunner()
        
        result = runner.invoke(add, ["test_package"])
        
        assert result.exit_code == 0
        assert "Adding test_package" in result.output
    
    def test_add_package_with_version(self):
        """Test adding a package with specific version."""
        runner = CliRunner()
        
        result = runner.invoke(add, ["test_package", "--version", "1.0.0"])
        
        assert result.exit_code == 0
        assert "Adding test_package@1.0.0" in result.output
