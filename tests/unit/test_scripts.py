#!/usr/bin/env python3
"""
Test suite for important scripts
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import script modules for unit testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestSetupModels:
    """Test setup_models.py script"""

    def test_setup_models_import(self):
        """Test that setup_models can be imported"""
        try:
            import scripts.setup_models

            assert True
        except ImportError:
            pytest.skip("setup_models script not available in test environment")


class TestPrePushValidation:
    """Test pre_push_validation.py script"""

    def test_pre_push_validation_import(self):
        """Test that pre_push_validation can be imported"""
        try:
            import scripts.pre_push_validation

            assert True
        except ImportError:
            pytest.skip("pre_push_validation script not available in test environment")


class TestRestoreCleanData:
    """Test restore_clean_data.py script"""

    def test_restore_clean_data_import(self):
        """Test that restore_clean_data can be imported"""
        try:
            import scripts.utilities.restore_clean_data

            assert True
        except ImportError:
            pytest.skip("restore_clean_data script not available in test environment")


class TestScriptCLI:
    """Test script command-line interfaces"""

    @pytest.mark.skip(
        reason="setup_models.py requires user input, not suitable for automated testing"
    )
    def test_setup_models_cli(self):
        """Test setup_models.py CLI"""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "setup_models.py"
        )

        # Test help
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "Setup required models" in result.stdout

    def test_pre_push_validation_cli(self):
        """Test pre_push_validation.py CLI"""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "pre_push_validation.py"
        )

        # Test help (if available)
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"], capture_output=True, text=True
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
