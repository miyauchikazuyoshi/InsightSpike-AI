"""
Tests for platform utilities module with corrected API.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import platform
import subprocess

from insightspike.utils.platform_utils import (
    PlatformInfo,
    PlatformDetector
)


class TestPlatformInfo:
    """Test PlatformInfo dataclass."""
    
    def test_platform_info_creation(self):
        """Test basic PlatformInfo creation."""
        info = PlatformInfo(
            platform="linux",
            architecture="x64",
            gpu_available=True,
            python_version="3.11.0"
        )
        
        assert info.platform == "linux"
        assert info.architecture == "x64"
        assert info.gpu_available is True
        assert info.python_version == "3.11.0"

    def test_platform_info_defaults(self):
        """Test PlatformInfo with default values."""
        info = PlatformInfo(
            platform="macos",
            architecture="arm64"
        )
        
        assert info.platform == "macos"
        assert info.architecture == "arm64"
        assert info.gpu_available is False  # Default value
        assert info.python_version  # Should be auto-populated by __post_init__


class TestPlatformDetector:
    """Test PlatformDetector functionality."""
    
    def test_init(self):
        """Test PlatformDetector initialization."""
        detector = PlatformDetector()
        assert hasattr(detector, 'detect_platform')

    @patch('platform.system')
    @patch('platform.machine')
    def test_detect_platform_linux(self, mock_machine, mock_system):
        """Test platform detection for Linux."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        
        detector = PlatformDetector()
        with patch.object(detector, '_check_gpu_availability', return_value=True):
            info = detector.detect_platform()
            
            assert info.platform == "linux"
            assert info.architecture == "x64"
            assert info.gpu_available is True

    @patch('platform.system')
    @patch('platform.machine')
    def test_detect_platform_macos(self, mock_machine, mock_system):
        """Test platform detection for macOS."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        
        detector = PlatformDetector()
        with patch.object(detector, '_check_gpu_availability', return_value=False):
            info = detector.detect_platform()
            
            assert info.platform == "macos"
            assert info.architecture == "arm64"
            assert info.gpu_available is False

    @patch('platform.system')
    @patch('platform.machine')
    def test_detect_platform_windows(self, mock_machine, mock_system):
        """Test platform detection for Windows."""
        mock_system.return_value = "Windows"
        mock_machine.return_value = "AMD64"
        
        detector = PlatformDetector()
        with patch.object(detector, '_check_gpu_availability', return_value=True):
            info = detector.detect_platform()
            
            assert info.platform == "windows"
            assert info.architecture == "x64"
            assert info.gpu_available is True

    def test_check_gpu_availability_no_torch(self):
        """Test GPU detection when torch is not available."""
        detector = PlatformDetector()
        
        with patch('importlib.import_module', side_effect=ImportError("No module named 'torch'")):
            gpu_available = detector._check_gpu_availability()
            assert gpu_available is False

    def test_check_gpu_availability_with_cuda(self):
        """Test GPU detection when CUDA is available."""
        detector = PlatformDetector()
        
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        
        with patch('importlib.import_module', return_value=mock_torch):
            gpu_available = detector._check_gpu_availability()
            assert gpu_available is True
