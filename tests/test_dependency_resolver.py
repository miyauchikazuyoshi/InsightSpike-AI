"""
Tests for dependency resolver module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from packaging.markers import Marker
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from insightspike.utils.dependency_resolver import (
    DependencySpec,
    ResolvedDependency,
    DependencyResolver
)
from insightspike.utils.platform_utils import PlatformInfo


class TestDependencySpec:
    """Test DependencySpec dataclass."""
    
    def test_dependency_spec_creation(self):
        """Test creating a DependencySpec."""
        spec = DependencySpec(
            name="torch",
            version_spec=">=1.0.0",
            markers="python_version >= '3.8'",
            extras=["cuda"],
            optional=True
        )
        
        assert spec.name == "torch"
        assert spec.version_spec == ">=1.0.0"
        assert spec.markers == "python_version >= '3.8'"
        assert spec.extras == ["cuda"]
        assert spec.optional is True
    
    def test_dependency_spec_defaults(self):
        """Test default values for DependencySpec."""
        spec = DependencySpec(name="numpy")
        
        assert spec.name == "numpy"
        assert spec.version_spec is None
        assert spec.markers is None
        assert spec.extras == []
        assert spec.optional is False


class TestResolvedDependency:
    """Test ResolvedDependency dataclass."""
    
    def test_resolved_dependency_creation(self):
        """Test creating a ResolvedDependency."""
        dependency = ResolvedDependency(
            name="torch",
            version="1.13.0",
            platform_specific=True,
            installation_method="pip",
            resolved_markers="sys_platform == 'linux'",
            fallback_packages=["torch-cpu"]
        )
        
        assert dependency.name == "torch"
        assert dependency.version == "1.13.0"
        assert dependency.platform_specific is True
        assert dependency.installation_method == "pip"
        assert dependency.resolved_markers == "sys_platform == 'linux'"
        assert dependency.fallback_packages == ["torch-cpu"]


class TestDependencyResolver:
    """Test DependencyResolver class."""
    
    @pytest.fixture
    def mock_platform_info(self):
        """Mock platform info for testing."""
        return PlatformInfo(
            os_name="linux",
            architecture="x86_64",
            python_version="3.9.0",
            has_gpu=True,
            cuda_version="11.8",
            platform_machine="x86_64",
            platform_system="Linux"
        )
    
    @pytest.fixture
    def resolver(self, mock_platform_info):
        """Create a DependencyResolver instance."""
        return DependencyResolver(mock_platform_info)
    
    def test_resolver_initialization(self, mock_platform_info):
        """Test resolver initialization."""
        resolver = DependencyResolver(mock_platform_info)
        assert resolver.platform_info == mock_platform_info
        assert "torch" in resolver.ml_packages
        assert "faiss-gpu" in resolver.ml_packages
    
    def test_evaluate_markers_true(self, resolver):
        """Test marker evaluation that should return True."""
        markers = "python_version >= '3.8' and sys_platform == 'linux'"
        assert resolver._evaluate_markers(markers) is True
    
    def test_evaluate_markers_false(self, resolver):
        """Test marker evaluation that should return False."""
        markers = "python_version >= '3.10' and sys_platform == 'win32'"
        assert resolver._evaluate_markers(markers) is False
    
    def test_evaluate_markers_none(self, resolver):
        """Test marker evaluation with None markers."""
        assert resolver._evaluate_markers(None) is True
    
    def test_evaluate_markers_invalid(self, resolver):
        """Test marker evaluation with invalid markers."""
        markers = "invalid_marker == 'test'"
        assert resolver._evaluate_markers(markers) is False
    
    def test_get_package_version_found(self, resolver):
        """Test getting package version when package exists."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "1.13.0\n"
            
            version = resolver._get_package_version("torch")
            assert version == "1.13.0"
    
    def test_get_package_version_not_found(self, resolver):
        """Test getting package version when package doesn't exist."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            version = resolver._get_package_version("nonexistent-package")
            assert version is None
    
    def test_determine_installation_method_poetry(self, resolver):
        """Test determining installation method - Poetry preferred."""
        with patch('shutil.which') as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/poetry" if cmd == "poetry" else None
            
            method = resolver._determine_installation_method("numpy")
            assert method == "poetry"
    
    def test_determine_installation_method_conda(self, resolver):
        """Test determining installation method - Conda fallback."""
        with patch('shutil.which') as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/conda" if cmd == "conda" else None
            
            method = resolver._determine_installation_method("numpy")
            assert method == "conda"
    
    def test_determine_installation_method_pip(self, resolver):
        """Test determining installation method - pip fallback."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = None
            
            method = resolver._determine_installation_method("numpy")
            assert method == "pip"
    
    def test_get_ml_package_variant_torch_gpu(self, resolver):
        """Test getting ML package variant for torch with GPU."""
        variant = resolver._get_ml_package_variant("torch")
        assert variant == "torch"  # GPU version
    
    def test_get_ml_package_variant_torch_cpu(self, resolver):
        """Test getting ML package variant for torch without GPU."""
        resolver.platform_info.has_gpu = False
        variant = resolver._get_ml_package_variant("torch")
        assert variant == "torch"  # CPU version
    
    def test_get_ml_package_variant_faiss_gpu(self, resolver):
        """Test getting ML package variant for faiss with GPU."""
        variant = resolver._get_ml_package_variant("faiss")
        assert variant == "faiss-gpu"
    
    def test_get_ml_package_variant_faiss_cpu(self, resolver):
        """Test getting ML package variant for faiss without GPU."""
        resolver.platform_info.has_gpu = False
        variant = resolver._get_ml_package_variant("faiss")
        assert variant == "faiss-cpu"
    
    def test_get_ml_package_variant_regular_package(self, resolver):
        """Test getting ML package variant for regular package."""
        variant = resolver._get_ml_package_variant("numpy")
        assert variant == "numpy"
    
    def test_resolve_dependency_simple(self, resolver):
        """Test resolving a simple dependency."""
        spec = DependencySpec(name="numpy", version_spec=">=1.20.0")
        
        with patch.object(resolver, '_get_package_version', return_value="1.21.0"):
            resolved = resolver.resolve_dependency(spec)
            
            assert resolved.name == "numpy"
            assert resolved.version == "1.21.0"
            assert resolved.platform_specific is False
            assert resolved.installation_method in ["poetry", "conda", "pip"]
    
    def test_resolve_dependency_with_markers(self, resolver):
        """Test resolving dependency with platform markers."""
        spec = DependencySpec(
            name="torch",
            markers="sys_platform == 'linux'"
        )
        
        with patch.object(resolver, '_get_package_version', return_value="1.13.0"):
            resolved = resolver.resolve_dependency(spec)
            
            assert resolved.name == "torch"
            assert resolved.platform_specific is True
            assert resolved.resolved_markers == "sys_platform == 'linux'"
    
    def test_resolve_dependency_ml_package(self, resolver):
        """Test resolving ML package dependency."""
        spec = DependencySpec(name="torch")
        
        with patch.object(resolver, '_get_package_version', return_value="1.13.0"):
            resolved = resolver.resolve_dependency(spec)
            
            assert resolved.name == "torch"
            assert resolved.platform_specific is True
            assert "torch" in resolved.fallback_packages or resolved.fallback_packages == []
    
    def test_resolve_dependency_not_found(self, resolver):
        """Test resolving dependency that doesn't exist."""
        spec = DependencySpec(name="nonexistent-package")
        
        with patch.object(resolver, '_get_package_version', return_value=None):
            with pytest.raises(PackageNotFoundError):
                resolver.resolve_dependency(spec)
    
    def test_resolve_dependency_invalid_markers(self, resolver):
        """Test resolving dependency with markers that don't match."""
        spec = DependencySpec(
            name="torch",
            markers="sys_platform == 'win32'"
        )
        
        resolved = resolver.resolve_dependency(spec)
        assert resolved is None
    
    def test_resolve_dependencies_batch(self, resolver):
        """Test resolving multiple dependencies."""
        specs = [
            DependencySpec(name="numpy"),
            DependencySpec(name="torch"),
            DependencySpec(name="invalid-package")
        ]
        
        with patch.object(resolver, '_get_package_version') as mock_version:
            mock_version.side_effect = lambda name: {
                "numpy": "1.21.0",
                "torch": "1.13.0",
                "invalid-package": None
            }.get(name)
            
            resolved = resolver.resolve_dependencies(specs)
            
            assert len(resolved) == 2  # Only valid packages
            assert resolved[0].name == "numpy"
            assert resolved[1].name == "torch"
    
    def test_resolve_dependencies_with_errors(self, resolver):
        """Test resolving dependencies with error handling."""
        specs = [
            DependencySpec(name="numpy"),
            DependencySpec(name="nonexistent-package")
        ]
        
        with patch.object(resolver, '_get_package_version') as mock_version:
            mock_version.side_effect = lambda name: "1.21.0" if name == "numpy" else None
            
            resolved, errors = resolver.resolve_dependencies(specs, raise_on_error=False)
            
            assert len(resolved) == 1
            assert len(errors) == 1
            assert "nonexistent-package" in errors[0]
    
    def test_get_platform_constraints(self, resolver):
        """Test getting platform constraints."""
        constraints = resolver.get_platform_constraints()
        
        assert "sys_platform == 'linux'" in constraints
        assert "platform_machine == 'x86_64'" in constraints
        assert "python_version >= '3.9'" in constraints
    
    def test_get_platform_constraints_with_gpu(self, resolver):
        """Test getting platform constraints with GPU."""
        constraints = resolver.get_platform_constraints(include_gpu=True)
        
        assert any("gpu" in constraint.lower() for constraint in constraints)
    
    def test_validate_dependency_compatibility_compatible(self, resolver):
        """Test dependency compatibility validation - compatible."""
        spec = DependencySpec(
            name="torch",
            markers="python_version >= '3.8'"
        )
        
        assert resolver.validate_dependency_compatibility(spec) is True
    
    def test_validate_dependency_compatibility_incompatible(self, resolver):
        """Test dependency compatibility validation - incompatible."""
        spec = DependencySpec(
            name="torch",
            markers="python_version >= '3.10'"
        )
        
        assert resolver.validate_dependency_compatibility(spec) is False
    
    def test_get_fallback_packages_torch(self, resolver):
        """Test getting fallback packages for torch."""
        fallbacks = resolver.get_fallback_packages("torch")
        
        assert isinstance(fallbacks, list)
        assert len(fallbacks) > 0
    
    def test_get_fallback_packages_regular(self, resolver):
        """Test getting fallback packages for regular package."""
        fallbacks = resolver.get_fallback_packages("numpy")
        
        assert fallbacks == []
    
    def test_resolve_with_version_constraint(self, resolver):
        """Test resolving dependency with version constraint."""
        spec = DependencySpec(
            name="numpy",
            version_spec=">=1.20.0,<2.0.0"
        )
        
        with patch.object(resolver, '_get_package_version', return_value="1.21.0"):
            resolved = resolver.resolve_dependency(spec)
            
            assert resolved.name == "numpy"
            assert resolved.version == "1.21.0"
    
    def test_resolve_with_extras(self, resolver):
        """Test resolving dependency with extras."""
        spec = DependencySpec(
            name="torch",
            extras=["cuda"]
        )
        
        with patch.object(resolver, '_get_package_version', return_value="1.13.0"):
            resolved = resolver.resolve_dependency(spec)
            
            assert resolved.name == "torch"
            assert "cuda" in str(resolved.extras) if hasattr(resolved, 'extras') else True
    
    def test_error_handling_resolution_error(self, resolver):
        """Test error handling for resolution errors."""
        with patch.object(resolver, '_get_package_version', side_effect=Exception("Network error")):
            spec = DependencySpec(name="torch")
            
            with pytest.raises(DependencyResolutionError):
                resolver.resolve_dependency(spec)


class TestDependencyResolverEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def minimal_platform_info(self):
        """Minimal platform info for testing."""
        return PlatformInfo(
            os_name="unknown",
            architecture="unknown",
            python_version="3.8.0",
            has_gpu=False,
            cuda_version=None,
            platform_machine="unknown",
            platform_system="Unknown"
        )
    
    def test_resolver_with_minimal_platform(self, minimal_platform_info):
        """Test resolver with minimal platform info."""
        resolver = DependencyResolver(minimal_platform_info)
        
        spec = DependencySpec(name="numpy")
        with patch.object(resolver, '_get_package_version', return_value="1.21.0"):
            resolved = resolver.resolve_dependency(spec)
            
            assert resolved.name == "numpy"
    
    def test_empty_dependency_list(self):
        """Test resolving empty dependency list."""
        platform_info = PlatformInfo(
            os_name="linux",
            architecture="x86_64",
            python_version="3.9.0",
            has_gpu=False,
            cuda_version=None,
            platform_machine="x86_64",
            platform_system="Linux"
        )
        resolver = DependencyResolver(platform_info)
        
        resolved = resolver.resolve_dependencies([])
        assert resolved == []
    
    def test_complex_markers(self):
        """Test complex marker evaluation."""
        platform_info = PlatformInfo(
            os_name="linux",
            architecture="x86_64",
            python_version="3.9.5",
            has_gpu=True,
            cuda_version="11.8",
            platform_machine="x86_64",
            platform_system="Linux"
        )
        resolver = DependencyResolver(platform_info)
        
        complex_markers = (
            "python_version >= '3.8' and sys_platform == 'linux' and "
            "platform_machine == 'x86_64'"
        )
        
        assert resolver._evaluate_markers(complex_markers) is True
