"""
Tests for dependency resolver module - Fixed version matching actual implementation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from insightspike.utils.dependency_resolver import (
    DependencySpec,
    ResolvedDependency, 
    ValidationResult,
    DependencyResolver
)
from insightspike.utils.platform_utils import PlatformInfo


class TestDependencySpec:
    """Test DependencySpec dataclass."""
    
    def test_dependency_spec_creation(self):
        """Test creating a DependencySpec."""
        spec = DependencySpec(
            name="torch",
            version=">=1.0.0",
            environment_markers="python_version >= '3.8'",
            extras=["cuda"]
        )
        
        assert spec.name == "torch"
        assert spec.version == ">=1.0.0"
        assert spec.environment_markers == "python_version >= '3.8'"
        assert spec.extras == ["cuda"]

    def test_dependency_spec_defaults(self):
        """Test default values for DependencySpec."""
        spec = DependencySpec(name="numpy")
        
        assert spec.name == "numpy"
        assert spec.version == ""
        assert spec.environment_markers == ""
        assert spec.extras == []


class TestResolvedDependency:
    """Test ResolvedDependency dataclass."""
    
    def test_resolved_dependency_creation(self):
        """Test creating a ResolvedDependency."""
        dependency = ResolvedDependency(
            name="torch",
            version="1.13.0",
            extras=["cuda"],
            platform_specific=True
        )
        
        assert dependency.name == "torch"
        assert dependency.version == "1.13.0"
        assert dependency.extras == ["cuda"]
        assert dependency.platform_specific is True


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            issues=[],
            warnings=["Warning message"]
        )
        
        assert result.is_valid is True
        assert result.issues == []
        assert result.warnings == ["Warning message"]

    def test_validation_result_defaults(self):
        """Test ValidationResult with default warnings."""
        result = ValidationResult(
            is_valid=False,
            issues=["Error message"]
        )
        
        assert result.is_valid is False
        assert result.issues == ["Error message"]
        assert result.warnings == []


class TestDependencyResolver:
    """Test DependencyResolver class."""
    
    @pytest.fixture
    def mock_platform_info(self):
        """Mock platform info for testing."""
        return PlatformInfo(
            platform="linux",
            architecture="x64",
            python_version="3.9.0",
            gpu_available=True
        )
    
    @pytest.fixture
    def resolver(self):
        """Create a DependencyResolver instance."""
        return DependencyResolver()
    
    def test_resolver_initialization(self, resolver):
        """Test DependencyResolver initialization."""
        assert resolver.platform_configs is not None
        assert "linux" in resolver.platform_configs
        assert "macos" in resolver.platform_configs
        assert "windows" in resolver.platform_configs

    def test_get_platform_dependencies_linux(self, resolver):
        """Test getting dependencies for Linux platform."""
        deps = resolver.get_platform_dependencies("linux")
        
        assert "torch" in deps
        assert "torchvision" in deps
        assert "faiss-gpu" in deps
        assert deps["torch"]["version"] == ">=2.0.0"

    def test_get_platform_dependencies_macos(self, resolver):
        """Test getting dependencies for macOS platform."""
        deps = resolver.get_platform_dependencies("macos")
        
        assert "torch" in deps
        assert "torchvision" in deps
        assert "faiss-cpu" in deps  # CPU-only for macOS
        assert deps["torch"]["version"] == ">=2.0.0"

    def test_get_platform_dependencies_unknown(self, resolver):
        """Test getting dependencies for unknown platform."""
        deps = resolver.get_platform_dependencies("unknown")
        
        assert deps == {}

    def test_resolve_dependencies(self, resolver, mock_platform_info):
        """Test resolving dependencies for a platform."""
        resolved_deps = resolver.resolve_dependencies(mock_platform_info)
        
        assert len(resolved_deps) > 0
        
        # Check that we get ResolvedDependency objects
        for dep in resolved_deps:
            assert isinstance(dep, ResolvedDependency)
            assert dep.name
            assert dep.version
            assert dep.platform_specific is True

    def test_validate_environment_valid(self, resolver):
        """Test environment validation with valid setup."""
        platform_info = PlatformInfo(
            platform="linux",
            architecture="x64",
            python_version="3.9.0",
            gpu_available=True
        )
        
        result = resolver.validate_environment(platform_info)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_validate_environment_old_python(self, resolver):
        """Test environment validation with old Python version."""
        platform_info = PlatformInfo(
            platform="linux",
            architecture="x64",
            python_version="3.7.0",
            gpu_available=True
        )
        
        result = resolver.validate_environment(platform_info)
        
        assert result.is_valid is False
        assert "Python 3.8 or higher required" in result.issues

    def test_validate_environment_arm64_macos(self, resolver):
        """Test environment validation on ARM64 macOS."""
        platform_info = PlatformInfo(
            platform="macos",
            architecture="arm64",
            python_version="3.9.0",
            gpu_available=False
        )
        
        result = resolver.validate_environment(platform_info)
        
        assert result.is_valid is True  # Should be valid
        assert "ARM64 macOS detected" in str(result.warnings)

    def test_validate_environment_linux_no_gpu(self, resolver):
        """Test environment validation on Linux without GPU."""
        platform_info = PlatformInfo(
            platform="linux",
            architecture="x64",
            python_version="3.9.0",
            gpu_available=False
        )
        
        result = resolver.validate_environment(platform_info)
        
        assert result.is_valid is True  # Should still be valid
        assert "No GPU detected" in str(result.warnings)


class TestDependencyResolverEdgeCases:
    """Test edge cases for DependencyResolver."""
    
    def test_empty_platform_config(self):
        """Test resolver with empty platform configuration."""
        resolver = DependencyResolver()
        
        # Test with a platform that doesn't exist
        deps = resolver.get_platform_dependencies("nonexistent")
        assert deps == {}
        
        # Test resolution with unknown platform
        platform_info = PlatformInfo(
            platform="nonexistent",
            architecture="unknown",
            python_version="3.9.0",
            gpu_available=False
        )
        
        resolved_deps = resolver.resolve_dependencies(platform_info)
        assert resolved_deps == []
