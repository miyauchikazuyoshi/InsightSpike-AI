"""
Tests for Poetry integration module.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import toml

from insightspike.utils.poetry_integration import (
    PoetryIntegration,
    PoetryNotFoundError,
    PyprojectTomlError,
    DependencyInstallationError
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
                    "dependencies": {
                        "python": "^3.8",
                        "numpy": "^1.20.0"
                    },
                    "group": {
                        "dev": {
                            "dependencies": {
                                "pytest": "^7.0.0"
                            }
                        }
                    }
                }
            }
        }
    
    @pytest.fixture
    def platform_info(self):
        """Mock platform info."""
        return PlatformInfo(
            os_name="linux",
            architecture="x86_64",
            python_version="3.9.0",
            has_gpu=True,
            cuda_version="11.8",
            platform_machine="x86_64",
            platform_system="Linux"
        )
    
    def test_poetry_integration_init(self, temp_project_dir, platform_info):
        """Test PoetryIntegration initialization."""
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        assert integration.project_path == temp_project_dir
        assert integration.platform_info == platform_info
        assert integration.pyproject_path == temp_project_dir / "pyproject.toml"
    
    def test_poetry_integration_init_string_path(self, temp_project_dir, platform_info):
        """Test PoetryIntegration initialization with string path."""
        integration = PoetryIntegration(str(temp_project_dir), platform_info)
        
        assert integration.project_path == temp_project_dir
    
    def test_check_poetry_available_found(self, temp_project_dir, platform_info):
        """Test checking Poetry availability when found."""
        with patch('shutil.which', return_value="/usr/bin/poetry"):
            integration = PoetryIntegration(temp_project_dir, platform_info)
            assert integration.check_poetry_available() is True
    
    def test_check_poetry_available_not_found(self, temp_project_dir, platform_info):
        """Test checking Poetry availability when not found."""
        with patch('shutil.which', return_value=None):
            integration = PoetryIntegration(temp_project_dir, platform_info)
            assert integration.check_poetry_available() is False
    
    def test_load_pyproject_toml_exists(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test loading existing pyproject.toml."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        config = integration.load_pyproject_toml()
        
        assert config["tool"]["poetry"]["name"] == "test-project"
        assert "numpy" in config["tool"]["poetry"]["dependencies"]
    
    def test_load_pyproject_toml_not_exists(self, temp_project_dir, platform_info):
        """Test loading non-existent pyproject.toml."""
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        with pytest.raises(PyprojectTomlError):
            integration.load_pyproject_toml()
    
    def test_load_pyproject_toml_invalid(self, temp_project_dir, platform_info):
        """Test loading invalid pyproject.toml."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            f.write("invalid toml content [")
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        with pytest.raises(PyprojectTomlError):
            integration.load_pyproject_toml()
    
    def test_save_pyproject_toml(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test saving pyproject.toml."""
        integration = PoetryIntegration(temp_project_dir, platform_info)
        integration.save_pyproject_toml(sample_pyproject_toml)
        
        # Verify file was created and content is correct
        pyproject_path = temp_project_dir / "pyproject.toml"
        assert pyproject_path.exists()
        
        with open(pyproject_path) as f:
            saved_config = toml.load(f)
        
        assert saved_config["tool"]["poetry"]["name"] == "test-project"
    
    def test_add_dependency_simple(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test adding a simple dependency."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(name="scipy", version_spec="^1.7.0")
        integration.add_dependency(spec)
        
        config = integration.load_pyproject_toml()
        assert "scipy" in config["tool"]["poetry"]["dependencies"]
        assert config["tool"]["poetry"]["dependencies"]["scipy"] == "^1.7.0"
    
    def test_add_dependency_with_markers(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test adding dependency with platform markers."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(
            name="torch",
            version_spec="^1.13.0",
            markers="sys_platform == 'linux'"
        )
        integration.add_dependency(spec)
        
        config = integration.load_pyproject_toml()
        torch_dep = config["tool"]["poetry"]["dependencies"]["torch"]
        assert torch_dep["version"] == "^1.13.0"
        assert torch_dep["markers"] == "sys_platform == 'linux'"
    
    def test_add_dependency_with_extras(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test adding dependency with extras."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(
            name="torch",
            version_spec="^1.13.0",
            extras=["cuda"]
        )
        integration.add_dependency(spec)
        
        config = integration.load_pyproject_toml()
        torch_dep = config["tool"]["poetry"]["dependencies"]["torch"]
        assert torch_dep["version"] == "^1.13.0"
        assert torch_dep["extras"] == ["cuda"]
    
    def test_add_dependency_to_group(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test adding dependency to specific group."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(name="black", version_spec="^22.0.0")
        integration.add_dependency(spec, group="dev")
        
        config = integration.load_pyproject_toml()
        assert "black" in config["tool"]["poetry"]["group"]["dev"]["dependencies"]
    
    def test_add_dependency_optional(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test adding optional dependency."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(name="torch", version_spec="^1.13.0", optional=True)
        integration.add_dependency(spec)
        
        config = integration.load_pyproject_toml()
        torch_dep = config["tool"]["poetry"]["dependencies"]["torch"]
        assert torch_dep["optional"] is True
    
    def test_remove_dependency(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test removing a dependency."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        integration.remove_dependency("numpy")
        
        config = integration.load_pyproject_toml()
        assert "numpy" not in config["tool"]["poetry"]["dependencies"]
    
    def test_remove_dependency_from_group(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test removing dependency from group."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        integration.remove_dependency("pytest", group="dev")
        
        config = integration.load_pyproject_toml()
        assert "pytest" not in config["tool"]["poetry"]["group"]["dev"]["dependencies"]
    
    def test_list_dependencies(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test listing dependencies."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        deps = integration.list_dependencies()
        
        assert "python" in deps
        assert "numpy" in deps
        assert deps["numpy"] == "^1.20.0"
    
    def test_list_dependencies_with_groups(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test listing dependencies including groups."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        deps = integration.list_dependencies(include_groups=True)
        
        assert "pytest" in deps  # From dev group
    
    @patch('subprocess.run')
    def test_install_dependencies_poetry(self, mock_run, temp_project_dir, platform_info):
        """Test installing dependencies with Poetry."""
        mock_run.return_value.returncode = 0
        
        with patch('shutil.which', return_value="/usr/bin/poetry"):
            integration = PoetryIntegration(temp_project_dir, platform_info)
            
            resolved_deps = [
                ResolvedDependency(
                    name="numpy",
                    version="1.21.0",
                    platform_specific=False,
                    installation_method="poetry"
                )
            ]
            
            integration.install_dependencies(resolved_deps)
            
            mock_run.assert_called()
    
    @patch('subprocess.run')
    def test_install_dependencies_pip(self, mock_run, temp_project_dir, platform_info):
        """Test installing dependencies with pip."""
        mock_run.return_value.returncode = 0
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        resolved_deps = [
            ResolvedDependency(
                name="numpy",
                version="1.21.0",
                platform_specific=False,
                installation_method="pip"
            )
        ]
        
        integration.install_dependencies(resolved_deps)
        
        mock_run.assert_called()
    
    @patch('subprocess.run')
    def test_install_dependencies_conda(self, mock_run, temp_project_dir, platform_info):
        """Test installing dependencies with conda."""
        mock_run.return_value.returncode = 0
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        resolved_deps = [
            ResolvedDependency(
                name="numpy",
                version="1.21.0",
                platform_specific=False,
                installation_method="conda"
            )
        ]
        
        integration.install_dependencies(resolved_deps)
        
        mock_run.assert_called()
    
    @patch('subprocess.run')
    def test_install_dependencies_failure(self, mock_run, temp_project_dir, platform_info):
        """Test handling installation failure."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Installation failed"
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        resolved_deps = [
            ResolvedDependency(
                name="numpy",
                version="1.21.0",
                platform_specific=False,
                installation_method="pip"
            )
        ]
        
        with pytest.raises(DependencyInstallationError):
            integration.install_dependencies(resolved_deps)
    
    def test_install_dependencies_dry_run(self, temp_project_dir, platform_info):
        """Test dry run installation."""
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        resolved_deps = [
            ResolvedDependency(
                name="numpy",
                version="1.21.0",
                platform_specific=False,
                installation_method="pip"
            )
        ]
        
        # Should not raise any exceptions
        result = integration.install_dependencies(resolved_deps, dry_run=True)
        assert result is None or isinstance(result, list)
    
    @patch('subprocess.run')
    def test_update_lock_file(self, mock_run, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test updating Poetry lock file."""
        mock_run.return_value.returncode = 0
        
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        with patch('shutil.which', return_value="/usr/bin/poetry"):
            integration = PoetryIntegration(temp_project_dir, platform_info)
            integration.update_lock_file()
            
            mock_run.assert_called()
    
    @patch('subprocess.run')
    def test_update_lock_file_failure(self, mock_run, temp_project_dir, platform_info):
        """Test handling lock file update failure."""
        mock_run.return_value.returncode = 1
        
        with patch('shutil.which', return_value="/usr/bin/poetry"):
            integration = PoetryIntegration(temp_project_dir, platform_info)
            
            with pytest.raises(DependencyInstallationError):
                integration.update_lock_file()
    
    def test_generate_requirements_file(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test generating requirements file."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        resolved_deps = [
            ResolvedDependency(
                name="numpy",
                version="1.21.0",
                platform_specific=False,
                installation_method="pip"
            ),
            ResolvedDependency(
                name="torch",
                version="1.13.0",
                platform_specific=True,
                installation_method="pip",
                resolved_markers="sys_platform == 'linux'"
            )
        ]
        
        requirements_path = temp_project_dir / "requirements.txt"
        integration.generate_requirements_file(resolved_deps, requirements_path)
        
        assert requirements_path.exists()
        
        with open(requirements_path) as f:
            content = f.read()
        
        assert "numpy==1.21.0" in content
        assert "torch==1.13.0" in content
        assert "sys_platform == 'linux'" in content
    
    def test_generate_requirements_file_default_path(self, temp_project_dir, platform_info):
        """Test generating requirements file with default path."""
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        resolved_deps = [
            ResolvedDependency(
                name="numpy",
                version="1.21.0",
                platform_specific=False,
                installation_method="pip"
            )
        ]
        
        integration.generate_requirements_file(resolved_deps)
        
        requirements_path = temp_project_dir / "requirements.txt"
        assert requirements_path.exists()
    
    def test_validate_dependencies_success(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test successful dependency validation."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        # Should not raise exceptions
        result = integration.validate_dependencies()
        assert isinstance(result, (bool, list))
    
    def test_get_platform_specific_dependencies(self, temp_project_dir, platform_info):
        """Test getting platform-specific dependencies."""
        # Create config with platform-specific deps
        config = {
            "tool": {
                "poetry": {
                    "name": "test-project",
                    "dependencies": {
                        "python": "^3.8",
                        "numpy": "^1.20.0",
                        "torch": {
                            "version": "^1.13.0",
                            "markers": "sys_platform == 'linux'"
                        }
                    }
                }
            }
        }
        
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(config, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        platform_deps = integration.get_platform_specific_dependencies()
        
        assert "torch" in platform_deps
        assert platform_deps["torch"]["markers"] == "sys_platform == 'linux'"
    
    def test_create_dependency_group(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test creating a new dependency group."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        integration.create_dependency_group("gpu")
        
        config = integration.load_pyproject_toml()
        assert "gpu" in config["tool"]["poetry"]["group"]
        assert "dependencies" in config["tool"]["poetry"]["group"]["gpu"]
    
    def test_add_platform_specific_dependency(self, temp_project_dir, sample_pyproject_toml, platform_info):
        """Test adding platform-specific dependency with auto-generated markers."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(sample_pyproject_toml, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(name="torch", version_spec="^1.13.0")
        integration.add_platform_specific_dependency(spec)
        
        config = integration.load_pyproject_toml()
        torch_dep = config["tool"]["poetry"]["dependencies"]["torch"]
        assert "markers" in torch_dep
        assert "linux" in torch_dep["markers"]
    

class TestPoetryIntegrationEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def platform_info(self):
        """Mock platform info."""
        return PlatformInfo(
            os_name="linux",
            architecture="x86_64",
            python_version="3.9.0",
            has_gpu=True,
            cuda_version="11.8",
            platform_machine="x86_64",
            platform_system="Linux"
        )
    
    def test_add_dependency_to_nonexistent_group(self, temp_project_dir, platform_info):
        """Test adding dependency to non-existent group."""
        config = {
            "tool": {
                "poetry": {
                    "name": "test-project",
                    "dependencies": {"python": "^3.8"}
                }
            }
        }
        
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(config, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(name="pytest", version_spec="^7.0.0")
        integration.add_dependency(spec, group="test")
        
        # Should create the group automatically
        updated_config = integration.load_pyproject_toml()
        assert "test" in updated_config["tool"]["poetry"]["group"]
        assert "pytest" in updated_config["tool"]["poetry"]["group"]["test"]["dependencies"]
    
    def test_malformed_pyproject_structure(self, temp_project_dir, platform_info):
        """Test handling malformed pyproject.toml structure."""
        config = {"tool": {}}  # Missing poetry section
        
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            toml.dump(config, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(name="numpy", version_spec="^1.20.0")
        integration.add_dependency(spec)
        
        # Should create missing structure
        updated_config = integration.load_pyproject_toml()
        assert "poetry" in updated_config["tool"]
        assert "numpy" in updated_config["tool"]["poetry"]["dependencies"]
    
    def test_empty_dependency_list_installation(self, temp_project_dir, platform_info):
        """Test installing empty dependency list."""
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        result = integration.install_dependencies([])
        assert result is None or result == []
    
    def test_complex_dependency_specification(self, temp_project_dir, platform_info):
        """Test handling complex dependency specifications."""
        pyproject_path = temp_project_dir / "pyproject.toml"
        config = {
            "tool": {
                "poetry": {
                    "name": "test-project",
                    "dependencies": {"python": "^3.8"}
                }
            }
        }
        with open(pyproject_path, "w") as f:
            toml.dump(config, f)
        
        integration = PoetryIntegration(temp_project_dir, platform_info)
        
        spec = DependencySpec(
            name="torch",
            version_spec=">=1.13.0,<2.0.0",
            markers="sys_platform == 'linux' and python_version >= '3.8'",
            extras=["cuda", "vision"],
            optional=True
        )
        
        integration.add_dependency(spec)
        
        config = integration.load_pyproject_toml()
        torch_dep = config["tool"]["poetry"]["dependencies"]["torch"]
        
        assert torch_dep["version"] == ">=1.13.0,<2.0.0"
        assert torch_dep["markers"] == "sys_platform == 'linux' and python_version >= '3.8'"
        assert torch_dep["extras"] == ["cuda", "vision"]
        assert torch_dep["optional"] is True
