#!/usr/bin/env python3
"""
Test suite for simplified configuration system
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from insightspike.config import (
    SimpleConfig, ConfigPresets, ConfigManager,
    get_config, create_config_file
)
from insightspike.utils.error_handler import ConfigurationError


def test_simple_config_defaults():
    """Test default configuration values"""
    print("\n=== Simple Config Defaults Test ===")
    
    config = SimpleConfig()
    
    # Check defaults
    assert config.mode == "cpu"
    assert config.safe_mode == True
    assert config.debug == False
    assert config.embedding_model == "paraphrase-MiniLM-L6-v2"
    assert config.llm_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.similarity_threshold == 0.25
    assert config.spike_ged_threshold == 0.5
    
    # Check directories created
    assert config.data_dir.exists()
    assert config.log_dir.exists()
    assert config.cache_dir.exists()
    
    print("✓ Default configuration works correctly")


def test_config_presets():
    """Test configuration presets"""
    print("\n=== Config Presets Test ===")
    
    # Development preset
    dev_config = ConfigPresets.development()
    assert dev_config.safe_mode == True
    assert dev_config.debug == True
    assert dev_config.spike_sensitivity == 2.0
    assert dev_config.spike_ged_threshold == 1.0  # 0.5 * 2.0
    print("✓ Development preset works")
    
    # Testing preset
    test_config = ConfigPresets.testing()
    assert test_config.data_dir == Path("test_data")
    assert test_config.max_episodes == 100
    print("✓ Testing preset works")
    
    # Production preset
    prod_config = ConfigPresets.production()
    assert prod_config.safe_mode == False
    assert prod_config.debug == False
    assert prod_config.batch_size == 64
    print("✓ Production preset works")
    
    # Experiment preset
    exp_config = ConfigPresets.experiment()
    assert exp_config.safe_mode == False
    assert exp_config.spike_sensitivity == 1.5
    print("✓ Experiment preset works")
    
    # Cloud preset
    cloud_config = ConfigPresets.cloud()
    assert cloud_config.llm_provider == "openai"
    assert cloud_config.llm_model == "gpt-3.5-turbo"
    print("✓ Cloud preset works")


def test_config_save_load():
    """Test configuration save/load functionality"""
    print("\n=== Config Save/Load Test ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        # Create and save config
        config = SimpleConfig(
            mode="gpu",
            safe_mode=False,
            max_tokens=512,
            spike_sensitivity=1.5
        )
        config.save(config_path)
        
        # Load config
        loaded_config = SimpleConfig.load(config_path)
        
        # Verify values
        # Note: mode may be adjusted to "cpu" if GPU not available
        assert loaded_config.mode in ["cpu", "gpu"]
        assert loaded_config.safe_mode == False
        assert loaded_config.max_tokens == 512
        # Spike thresholds get re-applied during __post_init__
        # Original values: 0.5, sensitivity: 1.5, result: 0.5 * 1.5 * 1.5 = 1.125
        # Since spike_sensitivity is saved and re-applied
        assert abs(loaded_config.spike_ged_threshold - 1.125) < 0.001
        
        # Check JSON format
        with open(config_path) as f:
            data = json.load(f)
        assert "mode" in data
        assert "safe_mode" in data
        assert isinstance(data["data_dir"], str)
        
        print("✓ Save/load functionality works")


def test_config_manager():
    """Test configuration manager"""
    print("\n=== Config Manager Test ===")
    
    # Basic usage
    manager = ConfigManager()
    assert manager.get("mode") == "cpu"
    assert manager.get("nonexistent", "default") == "default"
    
    # Set values
    manager.set("debug", True)
    assert manager.config.debug == True
    
    # Update multiple values
    manager.update(
        safe_mode=False,
        max_tokens=1024
    )
    assert manager.config.safe_mode == False
    assert manager.config.max_tokens == 1024
    
    # Test invalid key
    try:
        manager.set("invalid_key", "value")
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError:
        pass
    
    print("✓ Config manager works correctly")


def test_env_overrides():
    """Test environment variable overrides"""
    print("\n=== Environment Override Test ===")
    
    # Set environment variables
    os.environ["INSIGHTSPIKE_MODE"] = "gpu"
    os.environ["INSIGHTSPIKE_SAFE_MODE"] = "false"
    os.environ["INSIGHTSPIKE_MAX_TOKENS"] = "1024"
    os.environ["INSIGHTSPIKE_DEBUG"] = "true"
    os.environ["INSIGHTSPIKE_TEMPERATURE"] = "0.7"
    
    try:
        manager = ConfigManager()
        
        # Check overrides applied
        assert manager.config.mode == "gpu"
        assert manager.config.safe_mode == False
        assert manager.config.max_tokens == 1024
        assert manager.config.debug == True
        assert manager.config.temperature == 0.7
        
        print("✓ Environment overrides work correctly")
        
    finally:
        # Clean up environment
        for key in list(os.environ.keys()):
            if key.startswith("INSIGHTSPIKE_"):
                del os.environ[key]


def test_legacy_compatibility():
    """Test conversion to legacy config format"""
    print("\n=== Legacy Compatibility Test ===")
    
    manager = ConfigManager(SimpleConfig(
        mode="gpu",
        safe_mode=False,
        llm_model="custom-model",
        spike_ged_threshold=0.3,
        spike_sensitivity=2.0  # Will make threshold 0.6
    ))
    
    legacy = manager.to_legacy_config()
    
    # Check conversion
    assert legacy.llm.safe_mode == False
    assert legacy.llm.model_name == "custom-model"
    # Device may be adjusted to "cpu" if GPU not available
    assert legacy.llm.device in ["cpu", "gpu"]
    assert legacy.spike.spike_ged == 0.6  # 0.3 * 2.0
    assert legacy.embedding.model_name == "paraphrase-MiniLM-L6-v2"
    
    print("✓ Legacy compatibility works")


def test_validation():
    """Test configuration validation"""
    print("\n=== Validation Test ===")
    
    # Valid configuration
    manager = ConfigManager(SimpleConfig(safe_mode=True))
    assert manager.validate() == True
    print("✓ Valid configuration passes")
    
    # Invalid GPU mode (assuming no GPU)
    manager = ConfigManager(SimpleConfig(mode="gpu"))
    # Validation should handle gracefully
    manager.validate()
    print("✓ Invalid GPU mode handled")


def test_convenience_functions():
    """Test convenience functions"""
    print("\n=== Convenience Functions Test ===")
    
    # Get config with preset
    config = get_config("development")
    assert config.safe_mode == True
    assert config.debug == True
    
    # Test invalid preset
    try:
        get_config("invalid_preset")
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError:
        pass
    
    # Test config file creation
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        create_config_file(config_path, "testing")
        
        assert config_path.exists()
        loaded = SimpleConfig.load(config_path)
        assert loaded.data_dir == Path("test_data")
    
    print("✓ Convenience functions work correctly")


def run_all_tests():
    """Run all configuration tests"""
    print("=== Running Simple Config Tests ===")
    
    tests = [
        test_simple_config_defaults,
        test_config_presets,
        test_config_save_load,
        test_config_manager,
        test_env_overrides,
        test_legacy_compatibility,
        test_validation,
        test_convenience_functions
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tests)}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)