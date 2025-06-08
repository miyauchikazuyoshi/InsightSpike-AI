"""
Tests for platform detection utilities.
"""

import pytest
import platform
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.insightspike.utils.platform_utils import (
    PlatformDetector,
    PlatformValidator,
    PlatformInfo,
    get_current_platform_info,
    is_platform_compatible,
    get_optimal_dependency_config
)


class TestPlatformInfo:
    """Test PlatformInfo dataclass."""
    
    def test_platform_info_creation(self):
        """Test basic PlatformInfo creation."""
        info = PlatformInfo(
            system="Linux",
            machine="x86_64",
            processor="x86_64",
            python_version="3.11.0",
            architecture="64bit",
            is_gpu_available=True,
            cuda_version="12.1"
        )
        
        assert info.system == "Linux"
        assert info.machine == "x86_64"
        assert info.is_gpu_available is True
        assert info.cuda_version == "12.1"


class TestPlatformDetector:
    """Test PlatformDetector functionality."""
    
    def test_init(self):
        """Test PlatformDetector initialization."""
        detector = PlatformDetector()
        assert detector._platform_info is None
        assert detector._gpu_info_cache is None
    
    @patch('platform.system')
    @patch('platform.machine')
    @patch('platform.processor')
    @patch('platform.architecture')
    def test_detect_platform_basic(self, mock_arch, mock_proc, mock_machine, mock_system):
        """Test basic platform detection."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        mock_proc.return_value = "x86_64"
        mock_arch.return_value = ("64bit", "ELF")
        
        detector = PlatformDetector()
        with patch.object(detector, '_detect_gpu_info', return_value=(False, None, None)):
            info = detector._detect_platform()
        
        assert info.system == "Linux"
        assert info.machine == "x86_64"
        assert info.processor == "x86_64"
        assert info.architecture == "64bit"
        assert f"{sys.version_info.major}.{sys.version_info.minor}" in info.python_version
    
    def test_gpu_detection_no_torch(self):
        """Test GPU detection when torch is not available."""
        detector = PlatformDetector()
        
        with patch('builtins.__import__', side_effect=ImportError):
            gpu_available, cuda_version, gpu_info = detector._detect_gpu_info()
        
        assert gpu_available is False
        assert cuda_version is None
        assert gpu_info['available'] is False
    
    @patch('torch.cuda.is_available')
    @patch('torch.version.cuda', '12.1')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    def test_gpu_detection_with_torch(self, mock_device_name, mock_device_count, mock_cuda_available):
        """Test GPU detection with torch available."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_device_name.return_value = "NVIDIA GeForce RTX 3080"
        
        detector = PlatformDetector()
        gpu_available, cuda_version, gpu_info = detector._detect_gpu_info()
        
        assert gpu_available is True
        assert cuda_version == '12.1'
        assert len(gpu_info['devices']) == 1
        assert "RTX 3080" in gpu_info['devices'][0]
    
    def test_platform_constraints_darwin(self):
        """Test platform constraints for macOS."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Darwin",
                machine="arm64",
                processor="arm",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=False
            )
            
            constraints = detector.get_platform_constraints()
        
        assert 'sys_platform == "darwin"' in constraints
        assert 'platform_machine == "arm64"' in constraints
        assert any('python_version >= "3.11"' in c for c in constraints)
    
    def test_platform_constraints_linux(self):
        """Test platform constraints for Linux."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Linux",
                machine="x86_64",
                processor="x86_64",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=True,
                cuda_version="12.1"
            )
            
            constraints = detector.get_platform_constraints()
        
        assert 'sys_platform == "linux"' in constraints
        assert 'platform_machine == "x86_64"' in constraints
    
    def test_platform_constraints_windows(self):
        """Test platform constraints for Windows."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Windows",
                machine="AMD64",
                processor="Intel64 Family",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=True
            )
            
            constraints = detector.get_platform_constraints()
        
        assert 'sys_platform == "win32"' in constraints
    
    def test_gpu_recommended_linux_with_gpu(self):
        """Test GPU recommendation for Linux with GPU."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Linux",
                machine="x86_64",
                processor="x86_64",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=True,
                cuda_version="12.1"
            )
            
            assert detector.is_gpu_recommended() is True
    
    def test_gpu_not_recommended_macos(self):
        """Test GPU not recommended for macOS."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Darwin",
                machine="arm64",
                processor="arm",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=False
            )
            
            assert detector.is_gpu_recommended() is False
    
    def test_recommended_torch_index_cuda12(self):
        """Test recommended PyTorch index for CUDA 12."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Linux",
                machine="x86_64",
                processor="x86_64",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=True,
                cuda_version="12.1"
            )
            
            index_url = detector.get_recommended_torch_index()
            assert "cu121" in index_url
    
    def test_recommended_torch_index_cuda11(self):
        """Test recommended PyTorch index for CUDA 11."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Linux",
                machine="x86_64",
                processor="x86_64",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=True,
                cuda_version="11.8"
            )
            
            index_url = detector.get_recommended_torch_index()
            assert "cu118" in index_url
    
    def test_recommended_torch_index_cpu(self):
        """Test recommended PyTorch index for CPU."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Darwin",
                machine="arm64",
                processor="arm",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=False
            )
            
            index_url = detector.get_recommended_torch_index()
            assert "cpu" in index_url
    
    def test_caching_behavior(self):
        """Test that platform info is cached."""
        detector = PlatformDetector()
        
        with patch.object(detector, '_detect_platform') as mock_detect:
            mock_info = PlatformInfo(
                system="Linux",
                machine="x86_64",
                processor="x86_64",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=False
            )
            mock_detect.return_value = mock_info
            
            # First call
            info1 = detector.get_platform_info()
            # Second call
            info2 = detector.get_platform_info()
            
            # Should only call _detect_platform once
            mock_detect.assert_called_once()
            assert info1 is info2


class TestPlatformValidator:
    """Test PlatformValidator functionality."""
    
    def test_init(self):
        """Test PlatformValidator initialization."""
        validator = PlatformValidator()
        assert validator.detector is not None
        
        # Test with custom detector
        custom_detector = PlatformDetector()
        validator = PlatformValidator(custom_detector)
        assert validator.detector is custom_detector
    
    def test_torch_cuda_validation_no_gpu(self):
        """Test PyTorch CUDA validation when no GPU is available."""
        mock_detector = Mock()
        mock_detector.get_platform_info.return_value = PlatformInfo(
            system="Linux",
            machine="x86_64",
            processor="x86_64",
            python_version="3.11.0",
            architecture="64bit",
            is_gpu_available=False
        )
        
        validator = PlatformValidator(mock_detector)
        result = validator.validate_dependency_compatibility("torch", "torch+cu121")
        
        assert len(result['warnings']) > 0
        assert "GPU detected" in result['warnings'][0]
        assert "CPU version" in result['recommendations'][0]
    
    def test_torch_cuda_validation_macos(self):
        """Test PyTorch CUDA validation on macOS."""
        mock_detector = Mock()
        mock_detector.get_platform_info.return_value = PlatformInfo(
            system="Darwin",
            machine="arm64",
            processor="arm",
            python_version="3.11.0",
            architecture="64bit",
            is_gpu_available=False
        )
        
        validator = PlatformValidator(mock_detector)
        result = validator.validate_dependency_compatibility("torch", "torch+cu121")
        
        assert result['compatible'] is False
        assert "macOS" in result['warnings'][0]
    
    def test_faiss_gpu_validation_no_gpu(self):
        """Test FAISS GPU validation when no GPU is available."""
        mock_detector = Mock()
        mock_detector.get_platform_info.return_value = PlatformInfo(
            system="Linux",
            machine="x86_64",
            processor="x86_64",
            python_version="3.11.0",
            architecture="64bit",
            is_gpu_available=False
        )
        
        validator = PlatformValidator(mock_detector)
        result = validator.validate_dependency_compatibility("faiss-gpu", "1.7.0")
        
        assert len(result['warnings']) > 0
        assert "faiss-cpu" in result['recommendations'][0]
    
    def test_get_platform_specific_alternatives_faiss(self):
        """Test getting alternatives for FAISS GPU."""
        mock_detector = Mock()
        mock_detector.get_platform_info.return_value = PlatformInfo(
            system="Linux",
            machine="x86_64",
            processor="x86_64",
            python_version="3.11.0",
            architecture="64bit",
            is_gpu_available=False
        )
        
        validator = PlatformValidator(mock_detector)
        alternatives = validator.get_platform_specific_alternatives("faiss-gpu")
        
        assert "faiss-cpu" in alternatives
    
    def test_get_platform_specific_alternatives_torch_cuda(self):
        """Test getting alternatives for PyTorch CUDA."""
        mock_detector = Mock()
        mock_detector.get_platform_info.return_value = PlatformInfo(
            system="Darwin",
            machine="arm64",
            processor="arm",
            python_version="3.11.0",
            architecture="64bit",
            is_gpu_available=False
        )
        
        validator = PlatformValidator(mock_detector)
        alternatives = validator.get_platform_specific_alternatives("torch-cuda")
        
        assert any("cpu" in alt for alt in alternatives)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_current_platform_info(self):
        """Test get_current_platform_info function."""
        with patch('src.insightspike.utils.platform_utils.PlatformDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_info = PlatformInfo(
                system="Linux",
                machine="x86_64",
                processor="x86_64",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=True
            )
            mock_detector.get_platform_info.return_value = mock_info
            mock_detector_class.return_value = mock_detector
            
            info = get_current_platform_info()
            
            assert info.system == "Linux"
            assert info.machine == "x86_64"
    
    def test_is_platform_compatible(self):
        """Test is_platform_compatible function."""
        with patch('src.insightspike.utils.platform_utils.PlatformDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_detector.get_platform_constraints.return_value = [
                'sys_platform == "linux"',
                'platform_machine == "x86_64"'
            ]
            mock_detector_class.return_value = mock_detector
            
            # Test compatible marker
            assert is_platform_compatible('sys_platform == "linux"') is True
            
            # Test incompatible marker
            assert is_platform_compatible('sys_platform == "win32"') is False
    
    def test_get_optimal_dependency_config(self):
        """Test get_optimal_dependency_config function."""
        with patch('src.insightspike.utils.platform_utils.PlatformDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_info = PlatformInfo(
                system="Linux",
                machine="x86_64",
                processor="x86_64",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=True,
                cuda_version="12.1"
            )
            mock_detector.get_platform_info.return_value = mock_info
            mock_detector.get_recommended_torch_index.return_value = "https://download.pytorch.org/whl/cu121"
            mock_detector_class.return_value = mock_detector
            
            config = get_optimal_dependency_config()
            
            assert config['platform'] == 'linux'
            assert config['gpu_available'] is True
            assert config['recommended_packages']['torch'] == 'torch[cuda]'
            assert config['recommended_packages']['faiss'] == 'faiss-gpu'
            assert len(config['extra_index_urls']) > 0
    
    def test_get_optimal_dependency_config_cpu(self):
        """Test get_optimal_dependency_config for CPU-only platform."""
        with patch('src.insightspike.utils.platform_utils.PlatformDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_info = PlatformInfo(
                system="Darwin",
                machine="arm64",
                processor="arm",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=False
            )
            mock_detector.get_platform_info.return_value = mock_info
            mock_detector.get_recommended_torch_index.return_value = None
            mock_detector_class.return_value = mock_detector
            
            config = get_optimal_dependency_config()
            
            assert config['platform'] == 'darwin'
            assert config['gpu_available'] is False
            assert config['recommended_packages']['torch'] == 'torch'
            assert config['recommended_packages']['faiss'] == 'faiss-cpu'


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_gpu_detection_with_exceptions(self):
        """Test GPU detection with various exceptions."""
        detector = PlatformDetector()
        
        # Mock torch to raise different exceptions
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = [ImportError(), ImportError()]
            
            gpu_available, cuda_version, gpu_info = detector._detect_gpu_info()
            
            assert gpu_available is False
            assert cuda_version is None
    
    def test_platform_constraints_unknown_machine(self):
        """Test platform constraints with unknown machine type."""
        detector = PlatformDetector()
        
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = PlatformInfo(
                system="Linux",
                machine="unknown_arch",
                processor="unknown",
                python_version="3.11.0",
                architecture="64bit",
                is_gpu_available=False
            )
            
            constraints = detector.get_platform_constraints()
            
            # Should still have basic constraints
            assert 'sys_platform == "linux"' in constraints
            assert any('python_version' in c for c in constraints)
    
    def test_validator_with_unknown_package(self):
        """Test validator with unknown package type."""
        mock_detector = Mock()
        mock_detector.get_platform_info.return_value = PlatformInfo(
            system="Linux",
            machine="x86_64",
            processor="x86_64",
            python_version="3.11.0",
            architecture="64bit",
            is_gpu_available=True
        )
        
        validator = PlatformValidator(mock_detector)
        result = validator.validate_dependency_compatibility("unknown-package", "1.0.0")
        
        assert result['compatible'] is True
        assert len(result['warnings']) == 0
        assert len(result['recommendations']) == 0


if __name__ == "__main__":
    pytest.main([__file__])
