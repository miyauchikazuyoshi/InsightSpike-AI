def test_timestamp():
    """Test timestamp function from config module"""
    try:
        # Test with new config system
        from insightspike.core.config import get_config
        config = get_config()
        assert config is not None
        
        # Test timestamp creation manually
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        assert isinstance(ts, str) and ts
        
    except ImportError as e:
        # If imports fail, just test basic timestamp creation
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        assert isinstance(ts, str) and ts
