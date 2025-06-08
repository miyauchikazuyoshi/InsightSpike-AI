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
    add,
    install_preset,
    export_requirements,
    validate,
    update_lock
)
from insightspike.utils.platform_utils import PlatformInfo
from insightspike.utils.dependency_resolver import DependencySpec, ResolvedDependency


class TestPlatformInfoCommand:
    """Test platform-info command."""
    
    def test_platform_info_command(self):
        """Test platform info command execution."""
        runner = CliRunner()
        
        mock_platform_info = PlatformInfo(
            os_name="linux",
            architecture="x86_64",
            python_version="3.9.0",
            has_gpu=True,
            cuda_version="11.8",
            platform_machine="x86_64",
            platform_system="Linux"
        )
        
        with patch('insightspike.cli.dependency_commands.PlatformDetector') as mock_detector:
            mock_detector.return_value.detect_platform.return_value = mock_platform_info
            
            result = runner.invoke(platform_info)
            
            assert result.exit_code == 0
            assert "Platform Information" in result.output
            assert "linux" in result.output
            assert "x86_64" in result.output
            assert "3.9.0" in result.output
            assert "11.8" in result.output
    
    def test_platform_info_command_no_gpu(self):
        """Test platform info command without GPU."""
        runner = CliRunner()
        
        mock_platform_info = PlatformInfo(
            os_name="darwin",
            architecture="arm64",
            python_version="3.10.0",
            has_gpu=False,
            cuda_version=None,
            platform_machine="arm64",
            platform_system="Darwin"
        )
        
        with patch('insightspike.cli.dependency_commands.PlatformDetector') as mock_detector:
            mock_detector.return_value.detect_platform.return_value = mock_platform_info
            
            result = runner.invoke(platform_info)
            
            assert result.exit_code == 0
            assert "darwin" in result.output
            assert "arm64" in result.output
            assert "No GPU detected" in result.output
    
    def test_platform_info_command_with_constraints(self):
        """Test platform info command with constraints."""
        runner = CliRunner()
        
        mock_platform_info = PlatformInfo(
            os_name="linux",
            architecture="x86_64",
            python_version="3.9.0",
            has_gpu=True,
            cuda_version="11.8",
            platform_machine="x86_64",
            platform_system="Linux"
        )
        
        mock_constraints = [
            "sys_platform == 'linux'",
            "platform_machine == 'x86_64'",
            "python_version >= '3.9'"
        ]
        
        with patch('insightspike.cli.dependency_commands.PlatformDetector') as mock_detector:
            mock_detector.return_value.detect_platform.return_value = mock_platform_info
            
            with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
                mock_resolver.return_value.get_platform_constraints.return_value = mock_constraints
                
                result = runner.invoke(platform_info, ['--show-constraints'])
                
                assert result.exit_code == 0
                assert "Platform Constraints" in result.output
                assert "sys_platform == 'linux'" in result.output


class TestListDepsCommand:
    """Test list-deps command."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_list_deps_command(self, temp_project_dir):
        """Test list dependencies command."""
        runner = CliRunner()
        
        mock_dependencies = {
            "numpy": "^1.20.0",
            "torch": {
                "version": "^1.13.0",
                "markers": "sys_platform == 'linux'"
            }
        }
        
        with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
            mock_integration.return_value.list_dependencies.return_value = mock_dependencies
            
            result = runner.invoke(list_deps, ['--project-path', str(temp_project_dir)])
            
            assert result.exit_code == 0
            assert "Dependencies" in result.output
            assert "numpy" in result.output
            assert "torch" in result.output
    
    def test_list_deps_command_with_groups(self, temp_project_dir):
        """Test list dependencies command including groups."""
        runner = CliRunner()
        
        mock_dependencies = {
            "numpy": "^1.20.0",
            "pytest": "^7.0.0"  # Dev dependency
        }
        
        with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
            mock_integration.return_value.list_dependencies.return_value = mock_dependencies
            
            result = runner.invoke(list_deps, [
                '--project-path', str(temp_project_dir),
                '--include-groups'
            ])
            
            assert result.exit_code == 0
            assert "pytest" in result.output
    
    def test_list_deps_command_platform_specific_only(self, temp_project_dir):
        """Test list dependencies command for platform-specific only."""
        runner = CliRunner()
        
        mock_dependencies = {
            "torch": {
                "version": "^1.13.0",
                "markers": "sys_platform == 'linux'"
            }
        }
        
        with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
            mock_integration.return_value.get_platform_specific_dependencies.return_value = mock_dependencies
            
            result = runner.invoke(list_deps, [
                '--project-path', str(temp_project_dir),
                '--platform-specific-only'
            ])
            
            assert result.exit_code == 0
            assert "torch" in result.output
            assert "Platform-Specific Dependencies" in result.output


class TestAddCommand:
    """Test add command."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_add_command_simple(self, temp_project_dir):
        """Test add command with simple dependency."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="numpy",
            version="1.21.0",
            platform_specific=False,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(add, [
                    'numpy',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                assert "Successfully resolved" in result.output
                assert "numpy" in result.output
                mock_integration.return_value.add_dependency.assert_called()
    
    def test_add_command_with_version(self, temp_project_dir):
        """Test add command with version specification."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="torch",
            version="1.13.0",
            platform_specific=True,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(add, [
                    'torch',
                    '--version', '^1.13.0',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                spec_call = mock_resolver.return_value.resolve_dependency.call_args[0][0]
                assert spec_call.version_spec == '^1.13.0'
    
    def test_add_command_with_markers(self, temp_project_dir):
        """Test add command with platform markers."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="torch",
            version="1.13.0",
            platform_specific=True,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(add, [
                    'torch',
                    '--markers', "sys_platform == 'linux'",
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                spec_call = mock_resolver.return_value.resolve_dependency.call_args[0][0]
                assert spec_call.markers == "sys_platform == 'linux'"
    
    def test_add_command_with_extras(self, temp_project_dir):
        """Test add command with extras."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="torch",
            version="1.13.0",
            platform_specific=True,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(add, [
                    'torch',
                    '--extras', 'cuda,vision',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                spec_call = mock_resolver.return_value.resolve_dependency.call_args[0][0]
                assert 'cuda' in spec_call.extras
                assert 'vision' in spec_call.extras
    
    def test_add_command_to_group(self, temp_project_dir):
        """Test add command to specific group."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="pytest",
            version="7.1.0",
            platform_specific=False,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(add, [
                    'pytest',
                    '--group', 'dev',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                mock_integration.return_value.add_dependency.assert_called()
                call_args = mock_integration.return_value.add_dependency.call_args
                assert call_args[1]['group'] == 'dev'
    
    def test_add_command_optional(self, temp_project_dir):
        """Test add command with optional flag."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="torch",
            version="1.13.0",
            platform_specific=True,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(add, [
                    'torch',
                    '--optional',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                spec_call = mock_resolver.return_value.resolve_dependency.call_args[0][0]
                assert spec_call.optional is True
    
    def test_add_command_dry_run(self, temp_project_dir):
        """Test add command with dry run."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="numpy",
            version="1.21.0",
            platform_specific=False,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(add, [
                    'numpy',
                    '--dry-run',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                assert "DRY RUN" in result.output
                mock_integration.return_value.add_dependency.assert_not_called()
    
    def test_add_command_with_install(self, temp_project_dir):
        """Test add command with installation."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="numpy",
            version="1.21.0",
            platform_specific=False,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(add, [
                    'numpy',
                    '--install',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                mock_integration.return_value.install_dependencies.assert_called()
    
    def test_add_command_package_not_found(self, temp_project_dir):
        """Test add command when package is not found."""
        runner = CliRunner()
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = None
            
            result = runner.invoke(add, [
                'nonexistent-package',
                '--project-path', str(temp_project_dir)
            ])
            
            assert result.exit_code != 0
            assert "not compatible" in result.output or "Could not resolve" in result.output


class TestInstallPresetCommand:
    """Test install-preset command."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_install_preset_ml(self, temp_project_dir):
        """Test install ML preset."""
        runner = CliRunner()
        
        mock_resolved_deps = [
            ResolvedDependency("numpy", "1.21.0", False, "poetry"),
            ResolvedDependency("pandas", "1.3.0", False, "poetry"),
            ResolvedDependency("scikit-learn", "1.0.0", False, "poetry")
        ]
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependencies.return_value = mock_resolved_deps
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(install_preset, [
                    'ml',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                assert "Installing ML preset" in result.output
                mock_integration.return_value.install_dependencies.assert_called_with(
                    mock_resolved_deps, dry_run=False
                )
    
    def test_install_preset_gpu(self, temp_project_dir):
        """Test install GPU preset."""
        runner = CliRunner()
        
        mock_resolved_deps = [
            ResolvedDependency("torch", "1.13.0", True, "poetry"),
            ResolvedDependency("torchvision", "0.14.0", True, "poetry")
        ]
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependencies.return_value = mock_resolved_deps
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(install_preset, [
                    'gpu',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                assert "Installing GPU preset" in result.output
    
    def test_install_preset_dry_run(self, temp_project_dir):
        """Test install preset with dry run."""
        runner = CliRunner()
        
        mock_resolved_deps = [
            ResolvedDependency("numpy", "1.21.0", False, "poetry")
        ]
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependencies.return_value = mock_resolved_deps
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(install_preset, [
                    'basic',
                    '--dry-run',
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                assert "DRY RUN" in result.output
                mock_integration.return_value.install_dependencies.assert_called_with(
                    mock_resolved_deps, dry_run=True
                )
    
    def test_install_preset_invalid(self, temp_project_dir):
        """Test install invalid preset."""
        runner = CliRunner()
        
        result = runner.invoke(install_preset, [
            'invalid-preset',
            '--project-path', str(temp_project_dir)
        ])
        
        assert result.exit_code != 0
        assert "Unknown preset" in result.output


class TestExportRequirementsCommand:
    """Test export-requirements command."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_export_requirements_command(self, temp_project_dir):
        """Test export requirements command."""
        runner = CliRunner()
        
        mock_resolved_deps = [
            ResolvedDependency("numpy", "1.21.0", False, "pip"),
            ResolvedDependency("torch", "1.13.0", True, "pip", "sys_platform == 'linux'")
        ]
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependencies.return_value = mock_resolved_deps
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(export_requirements, [
                    '--project-path', str(temp_project_dir)
                ])
                
                assert result.exit_code == 0
                assert "Requirements file generated" in result.output
                mock_integration.return_value.generate_requirements_file.assert_called()
    
    def test_export_requirements_custom_output(self, temp_project_dir):
        """Test export requirements with custom output file."""
        runner = CliRunner()
        
        mock_resolved_deps = [
            ResolvedDependency("numpy", "1.21.0", False, "pip")
        ]
        
        custom_output = temp_project_dir / "custom-requirements.txt"
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependencies.return_value = mock_resolved_deps
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                result = runner.invoke(export_requirements, [
                    '--project-path', str(temp_project_dir),
                    '--output', str(custom_output)
                ])
                
                assert result.exit_code == 0
                call_args = mock_integration.return_value.generate_requirements_file.call_args
                assert call_args[0][1] == custom_output


class TestValidateCommand:
    """Test validate command."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_validate_command_success(self, temp_project_dir):
        """Test successful validation."""
        runner = CliRunner()
        
        with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
            mock_integration.return_value.validate_dependencies.return_value = True
            
            result = runner.invoke(validate, [
                '--project-path', str(temp_project_dir)
            ])
            
            assert result.exit_code == 0
            assert "Dependencies are valid" in result.output
    
    def test_validate_command_failure(self, temp_project_dir):
        """Test validation failure."""
        runner = CliRunner()
        
        with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
            mock_integration.return_value.validate_dependencies.return_value = [
                "torch requires CUDA but none found"
            ]
            
            result = runner.invoke(validate, [
                '--project-path', str(temp_project_dir)
            ])
            
            assert result.exit_code != 0
            assert "Validation errors found" in result.output
            assert "torch requires CUDA" in result.output


class TestUpdateLockCommand:
    """Test update-lock command."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_update_lock_command(self, temp_project_dir):
        """Test update lock command."""
        runner = CliRunner()
        
        with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
            result = runner.invoke(update_lock, [
                '--project-path', str(temp_project_dir)
            ])
            
            assert result.exit_code == 0
            assert "Lock file updated successfully" in result.output
            mock_integration.return_value.update_lock_file.assert_called()
    
    def test_update_lock_command_failure(self, temp_project_dir):
        """Test update lock command failure."""
        runner = CliRunner()
        
        with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
            mock_integration.return_value.update_lock_file.side_effect = Exception("Lock update failed")
            
            result = runner.invoke(update_lock, [
                '--project-path', str(temp_project_dir)
            ])
            
            assert result.exit_code != 0
            assert "Failed to update lock file" in result.output


class TestCommandEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_commands_with_invalid_project_path(self):
        """Test commands with invalid project path."""
        runner = CliRunner()
        invalid_path = "/nonexistent/path"
        
        # Test multiple commands with invalid path
        commands_to_test = [
            (list_deps, ['--project-path', invalid_path]),
            (add, ['numpy', '--project-path', invalid_path]),
            (validate, ['--project-path', invalid_path])
        ]
        
        for command, args in commands_to_test:
            result = runner.invoke(command, args)
            # Should handle gracefully - either exit with error or show warning
            assert result.exit_code != 0 or "not found" in result.output.lower()
    
    def test_add_command_without_project_path(self):
        """Test add command without project path (should use current directory)."""
        runner = CliRunner()
        
        mock_resolved = ResolvedDependency(
            name="numpy",
            version="1.21.0",
            platform_specific=False,
            installation_method="poetry"
        )
        
        with patch('insightspike.cli.dependency_commands.DependencyResolver') as mock_resolver:
            mock_resolver.return_value.resolve_dependency.return_value = mock_resolved
            
            with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
                with patch('pathlib.Path.cwd') as mock_cwd:
                    mock_cwd.return_value = Path("/current/dir")
                    
                    result = runner.invoke(add, ['numpy'])
                    
                    # Should use current directory
                    assert result.exit_code == 0
                    mock_integration.assert_called_with(Path("/current/dir"), mock.ANY)
    
    def test_commands_handle_keyboard_interrupt(self):
        """Test commands handle keyboard interrupt gracefully."""
        runner = CliRunner()
        
        with patch('insightspike.cli.dependency_commands.PoetryIntegration') as mock_integration:
            mock_integration.side_effect = KeyboardInterrupt()
            
            result = runner.invoke(list_deps, ['--project-path', '/tmp'])
            
            # Should exit gracefully on interrupt
            assert result.exit_code != 0
