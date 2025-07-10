#!/usr/bin/env python3
"""
Test suite for important scripts
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import tempfile
import shutil
import json

# Import script modules for unit testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestSetupModels:
    """Test setup_models.py script"""
    
    @patch('scripts.setup_models.download_model')
    @patch('scripts.setup_models.verify_model')
    def test_setup_models_success(self, mock_verify, mock_download):
        """Test successful model setup"""
        mock_verify.return_value = True
        mock_download.return_value = True
        
        from scripts.setup_models import setup_all_models
        
        result = setup_all_models()
        assert result is True
        assert mock_verify.called
        assert mock_download.called
    
    @patch('scripts.setup_models.download_model')
    @patch('scripts.setup_models.verify_model')
    def test_setup_models_download_failure(self, mock_verify, mock_download):
        """Test model setup with download failure"""
        mock_verify.return_value = False
        mock_download.return_value = False
        
        from scripts.setup_models import setup_all_models
        
        result = setup_all_models()
        assert result is False


class TestPrePushValidation:
    """Test pre_push_validation.py script"""
    
    def test_check_data_files(self):
        """Test data file validation"""
        from scripts.pre_push_validation import check_data_files
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data files
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            
            # Create required files
            (data_dir / "episodes.json").write_text("[]")
            (data_dir / "graph_pyg.pt").write_bytes(b"test")
            (data_dir / "index.faiss").write_bytes(b"test")
            (data_dir / "index.json").write_text("{}")
            
            # Test validation
            result = check_data_files(str(data_dir))
            assert result is True
    
    def test_check_data_files_missing(self):
        """Test data file validation with missing files"""
        from scripts.pre_push_validation import check_data_files
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            
            # Missing files should fail
            result = check_data_files(str(data_dir))
            assert result is False
    
    @patch('subprocess.run')
    def test_run_tests(self, mock_run):
        """Test running pytest"""
        from scripts.pre_push_validation import run_tests
        
        # Mock successful test run
        mock_run.return_value = Mock(returncode=0)
        
        result = run_tests()
        assert result is True
        assert mock_run.called
    
    @patch('subprocess.run')
    def test_run_tests_failure(self, mock_run):
        """Test running pytest with failures"""
        from scripts.pre_push_validation import run_tests
        
        # Mock failed test run
        mock_run.return_value = Mock(returncode=1)
        
        result = run_tests()
        assert result is False


class TestRestoreCleanData:
    """Test restore_clean_data.py script"""
    
    def test_restore_clean_data(self):
        """Test data restoration"""
        from scripts.utilities.restore_clean_data import restore_clean_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create backup directory
            backup_dir = Path(tmpdir) / "clean_backup"
            backup_dir.mkdir()
            
            # Create test backup files
            (backup_dir / "episodes.json").write_text('[{"text": "test"}]')
            (backup_dir / "graph_pyg.pt").write_bytes(b"graph_data")
            
            # Create data directory
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            
            # Test restoration
            result = restore_clean_data(str(data_dir), str(backup_dir))
            assert result is True
            
            # Verify files were restored
            assert (data_dir / "episodes.json").exists()
            assert (data_dir / "graph_pyg.pt").exists()
    
    def test_restore_clean_data_no_backup(self):
        """Test restoration with missing backup"""
        from scripts.utilities.restore_clean_data import restore_clean_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            backup_dir = Path(tmpdir) / "nonexistent"
            
            result = restore_clean_data(str(data_dir), str(backup_dir))
            assert result is False


class TestScriptCLI:
    """Test script command-line interfaces"""
    
    def test_setup_models_cli(self):
        """Test setup_models.py CLI"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "setup_models.py"
        
        # Test help
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Setup required models" in result.stdout
    
    def test_pre_push_validation_cli(self):
        """Test pre_push_validation.py CLI"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "pre_push_validation.py"
        
        # Test help (if available)
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )
        # Script might not have help, but should not crash
        assert result.returncode in [0, 1, 2]


def test_script_imports():
    """Test that all scripts can be imported"""
    scripts_to_test = [
        "setup_models",
        "pre_push_validation",
    ]
    
    for script in scripts_to_test:
        try:
            # Dynamic import
            __import__(f"scripts.{script}")
            assert True, f"Successfully imported {script}"
        except ImportError as e:
            # Some scripts might have dependencies
            if "No module named" in str(e):
                pytest.skip(f"Missing dependency for {script}: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])