"""
Main entry point for InsightSpike-AI

This is the composition root where all dependencies are wired together.
"""

import logging
import sys
from pathlib import Path

# Setup basic logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_app():
    """
    Composition Root - Where all dependencies are resolved and injected.
    """
    try:
        # Import here to avoid circular imports
        from .config.loader import load_config
        from .implementations.datastore.factory import DataStoreFactory
        from .implementations.agents.main_agent import MainAgent
        from .cli import spike
        
        logger.info("Starting InsightSpike composition root...")
        
        # 1. Load configuration
        try:
            config = load_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using default")
            from .config.presets import get_preset_config
            config = get_preset_config("development")
        
        # 2. Create DataStore based on configuration
        if hasattr(config, "datastore") and config.datastore:
            datastore_config = {
                "type": config.datastore.type if hasattr(config.datastore, "type") else "filesystem",
                "params": {
                    "base_path": config.datastore.base_path if hasattr(config.datastore, "base_path") else "./data"
                }
            }
        else:
            datastore_config = {
                "type": "filesystem",
                "params": {"base_path": "./data"}
            }
        datastore = DataStoreFactory.create_from_config(datastore_config)
        logger.info(f"Created DataStore: {datastore.__class__.__name__}")
        
        # 3. Pass dependencies to CLI
        logger.info("Starting CLI application...")
        
        # Run the CLI with dependency injection
        spike.run_cli(config=config, datastore=datastore)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in composition root: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_app()
