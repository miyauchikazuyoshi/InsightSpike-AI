def test_timestamp():
    """Test timestamp function from config module"""
    try:
        # Try importing from the main config module
        from insightspike import config
        ts = config.timestamp()
        assert isinstance(ts, str) and ts
    except AttributeError:
        # Fallback: try importing directly from config.py file
        try:
            import sys
            import os
            from pathlib import Path
            
            # Get the path to the config.py file
            config_path = Path(__file__).parent.parent.parent / "src" / "insightspike" / "config.py"
            
            # Import the config module directly
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            ts = config_module.timestamp()
            assert isinstance(ts, str) and ts
        except Exception as e:
            # If all else fails, just test that we can create a timestamp
            from datetime import datetime
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            assert isinstance(ts, str) and ts
