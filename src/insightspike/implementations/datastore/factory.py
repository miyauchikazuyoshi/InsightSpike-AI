"""
DataStore Factory
================

Factory for creating DataStore instances based on configuration.
"""

import logging
from typing import Dict, Optional, Type

from ...core.base.datastore import DataStore
from .filesystem_store import FileSystemDataStore
from .memory_store import InMemoryDataStore

logger = logging.getLogger(__name__)


class DataStoreFactory:
    """Factory for creating DataStore instances"""

    # Registry of available datastore implementations
    _registry: Dict[str, Type[DataStore]] = {
        "filesystem": FileSystemDataStore,
        "memory": InMemoryDataStore,
    }

    @classmethod
    def register(cls, name: str, datastore_class: Type[DataStore]):
        """Register a new DataStore implementation

        Args:
            name: Name to register the implementation under
            datastore_class: DataStore implementation class
        """
        cls._registry[name] = datastore_class
        logger.info(f"Registered DataStore implementation: {name}")

    @classmethod
    def create(cls, store_type: str = "filesystem", **kwargs) -> DataStore:
        """Create a DataStore instance

        Args:
            store_type: Type of store to create
            **kwargs: Arguments to pass to store constructor

        Returns:
            DataStore instance

        Raises:
            ValueError: If store_type is not registered
        """
        if store_type not in cls._registry:
            raise ValueError(
                f"Unknown store type: {store_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        store_class = cls._registry[store_type]
        store = store_class(**kwargs)

        logger.info(f"Created {store_type} DataStore")
        return store

    @classmethod
    def create_from_config(cls, config: Dict) -> DataStore:
        """Create DataStore from configuration dictionary

        Args:
            config: Configuration dict with 'type' and optional params

        Returns:
            DataStore instance
        """
        store_type = config.get("type", "filesystem")
        params = config.get("params", {})

        return cls.create(store_type, **params)


# Future DataStore implementations can be registered here
# Example for PostgreSQL:
"""
try:
    from .postgres_store import PostgreSQLDataStore
    DataStoreFactory.register("postgresql", PostgreSQLDataStore)
except ImportError:
    logger.debug("PostgreSQL support not available")
"""

# Example for Vector DBs:
"""
try:
    from .pinecone_store import PineconeDataStore
    DataStoreFactory.register("pinecone", PineconeDataStore)
except ImportError:
    logger.debug("Pinecone support not available")

try:
    from .weaviate_store import WeaviateDataStore
    DataStoreFactory.register("weaviate", WeaviateDataStore)
except ImportError:
    logger.debug("Weaviate support not available")
"""
