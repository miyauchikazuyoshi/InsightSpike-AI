"""
InsightSpike Main Entry Point
=============================

This is the composition root where all dependencies are wired together.
The main function handles:
1. Configuration loading
2. Dependency injection
3. Application startup
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from .cli.spike import app as cli_app
from .config.loader import load_config
from .config.models import InsightSpikeConfig, DataStoreConfig
from .core.base.datastore import DataStore
from .implementations.agents.main_agent import MainAgent
from .implementations.datastore.factory import DataStoreFactory

logger = logging.getLogger(__name__)


class Application:
    """Main application container with all dependencies."""

    def __init__(self, config: InsightSpikeConfig, agent: MainAgent):
        self.config = config
        self.agent = agent
        self._running = False

    def start_cli(self):
        """Start the CLI interface."""
        # Import state from CLI to avoid circular imports
        from .cli.spike import state

        # Store agent in CLI state for commands to access
        state.agent = self.agent
        state.config = self.config
        state.initialized_from_main = True

        # Run the Typer app
        try:
            cli_app()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
            sys.exit(1)


def create_application(config_path: Optional[Path] = None) -> Application:
    """
    Create and wire up the entire application.

    This is the composition root where all dependencies are resolved.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configured Application instance
    """
    # 1. Load configuration
    logger.info("Loading configuration...")
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Fallback to safe development preset
        from .config.presets import ConfigPresets

        config = ConfigPresets.development()

    # 2. Create DataStore based on configuration
    logger.info("Creating DataStore...")
    from .utils.path_utils import resolve_project_relative
    # Prefer legacy paths.data_dir unless user explicitly configured datastore.root_path
    if hasattr(config, "datastore") and config.datastore:
        ds_type = getattr(config.datastore, "type", "filesystem")
        root_path = getattr(config.datastore, "root_path", None)
        base_path = getattr(config.datastore, "base_path", None)
        fallback_path = (
            str(getattr(getattr(config, "paths", None), "data_dir", "./data"))
        )
        # Detect explicit root_path change
        # Prefer marker set by loader; fallback to string comparison with model default
        default_root = DataStoreConfig().root_path
        root_explicit = bool(getattr(config.datastore, "explicit_root_path", False)) or (
            root_path is not None and str(root_path) != str(default_root)
        )
        if base_path:
            effective_path = base_path
        elif root_explicit:
            effective_path = root_path
        else:
            # Backward compatible default: use paths.data_dir (typically "data/")
            effective_path = fallback_path

        # Normalize relative paths against project root to avoid nested CWD artifacts
        effective_path = resolve_project_relative(effective_path)

        logger.info(f"Using DataStore base_path: {effective_path}")
        datastore_config = {
            "type": ds_type,
            "params": {"base_path": effective_path},
        }
    else:
        normalized = resolve_project_relative("./data")
        datastore_config = {
            "type": "filesystem",
            "params": {"base_path": normalized},
        }
        logger.info("Using DataStore base_path: ./data")
    datastore = DataStoreFactory.create_from_config(datastore_config)

    # 3. Create MainAgent with injected dependencies
    logger.info("Creating MainAgent...")
    agent = MainAgent(config=config, datastore=datastore)

    # 4. Initialize the agent
    if not agent.initialize():
        logger.error("Failed to initialize MainAgent")
        sys.exit(1)

    # 5. Try to load existing state
    logger.info("Loading agent state...")
    if agent.load_state():
        logger.info("Successfully loaded existing agent state")
    else:
        logger.info("No existing state found, starting fresh")

    # 6. Create application container
    app = Application(config=config, agent=agent)

    logger.info("Application created successfully")
    return app


def main(config_path: Optional[str] = None):
    """
    Main entry point for InsightSpike.

    Args:
        config_path: Optional path to configuration file
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress some verbose libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logger.info("Starting InsightSpike...")

    # Create application with all dependencies wired
    config_path_obj = Path(config_path) if config_path else None
    app = create_application(config_path_obj)

    # Start the CLI
    app.start_cli()

    # Cleanup on exit
    logger.info("Shutting down...")
    if app.agent:
        app.agent.save_state()
        app.agent.cleanup()

    logger.info("InsightSpike shutdown complete")


if __name__ == "__main__":
    # Allow running with optional config file argument
    import sys

    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_file)
