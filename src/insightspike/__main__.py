"""
Main entry point for InsightSpike-AI

This is the composition root where all dependencies are wired together.
"""

import logging
import sys
from pathlib import Path

# Setup basic logging before imports
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_app():
    """
    Composition Root - Where all dependencies are resolved and injected.
    """
    try:
        # Import here to avoid circular imports
        from .cli import spike
        from .config.loader import load_config
        from .implementations.agents.main_agent import MainAgent
        from .implementations.datastore.factory import DataStoreFactory

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
                "type": config.datastore.type
                if hasattr(config.datastore, "type")
                else "filesystem",
                "params": {
                    "base_path": config.datastore.base_path
                    if hasattr(config.datastore, "base_path")
                    else "./data"
                },
            }
        else:
            datastore_config = {"type": "filesystem", "params": {"base_path": "./data"}}
        datastore = DataStoreFactory.create_from_config(datastore_config)
        logger.info(f"Created DataStore: {datastore.__class__.__name__}")

        # 3. Pre-warm LLM models based on configuration
        if getattr(config, "pre_warm_models", True):  # Default to pre-warming
            logger.info("Pre-warming LLM models...")
            from .implementations.layers.layer4_llm_interface import (
                LLMProviderRegistry,
                LLMConfig,
                LLMProviderType,
            )

            try:
                # Pre-warm local model if configured
                if hasattr(config, "llm") and config.llm.provider == "local":
                    local_config = LLMConfig(
                        provider=LLMProviderType.LOCAL,
                        model_name=config.llm.model,
                        temperature=config.llm.temperature,
                        max_tokens=config.llm.max_tokens,
                        device=getattr(config.llm, "device", "cpu"),
                    )
                    logger.info(f"Pre-warming local model: {config.llm.model}")
                    LLMProviderRegistry.get_instance(local_config)

                # Always pre-warm clean provider as fallback
                clean_config = LLMConfig(provider=LLMProviderType.CLEAN)
                LLMProviderRegistry.get_instance(clean_config)

                # Pre-warm OpenAI if API key is available
                if (
                    hasattr(config, "llm")
                    and config.llm.provider == "openai"
                    and config.llm.api_key
                ):
                    openai_config = LLMConfig(
                        provider=LLMProviderType.OPENAI,
                        model_name=config.llm.model,
                        api_key=config.llm.api_key,
                        temperature=config.llm.temperature,
                        max_tokens=config.llm.max_tokens,
                    )
                    logger.info(f"Pre-warming OpenAI model: {config.llm.model}")
                    LLMProviderRegistry.get_instance(openai_config)

                logger.info(
                    f"Pre-warmed {len(LLMProviderRegistry.get_cached_providers())} model(s)"
                )
            except Exception as e:
                logger.warning(f"Failed to pre-warm some models: {e}")
                # Continue anyway - models will be loaded on demand

        # 4. Pass dependencies to CLI
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
