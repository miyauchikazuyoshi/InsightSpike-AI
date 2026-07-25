"""
DataStore Factory
================

Factory for creating DataStore instances based on configuration.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Type

from ...core.base.datastore import DataStore
from ...utils.path_utils import resolve_project_relative
from .filesystem_store import FileSystemDataStore
from .memory_store import InMemoryDataStore

logger = logging.getLogger(__name__)


class DataStoreFactory:
    """Factory for creating DataStore instances"""

    _aliases = {
        "in_memory": "memory",
    }

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
        normalized = cls._normalize_store_type(name)
        cls._registry[normalized] = datastore_class
        logger.info(f"Registered DataStore implementation: {normalized}")

    @classmethod
    def _normalize_store_type(cls, store_type: str) -> str:
        normalized = str(store_type or "filesystem").strip().lower()
        return cls._aliases.get(normalized, normalized)

    @classmethod
    def create(cls, store_type: Any = "filesystem", **kwargs) -> DataStore:
        """Create a DataStore instance

        Args:
            store_type: Type of store to create
            **kwargs: Arguments to pass to store constructor

        Returns:
            DataStore instance

        Raises:
            ValueError: If store_type is not registered
        """
        # Backward / test convenience: allow passing a config object directly.
        if not isinstance(store_type, str):
            if kwargs:
                raise TypeError(
                    "Constructor kwargs cannot be combined with a datastore config object"
                )
            return cls.create_from_config(store_type)

        store_type = cls._normalize_store_type(store_type)
        if store_type not in cls._registry:
            available = sorted(set(cls._registry) | set(cls._aliases))
            raise ValueError(
                f"Unknown store type: {store_type}. "
                f"Available types: {available}"
            )

        store_class = cls._registry[store_type]
        store = store_class(**kwargs)

        logger.info(f"Created {store_type} DataStore")
        return store

    @classmethod
    def create_from_config(cls, config: Any) -> DataStore:
        """Create a DataStore from raw constructor or application config.

        Args:
            config: ``{"type": ..., "params": ...}``, a DataStoreConfig, or
                a full InsightSpikeConfig.

        Returns:
            DataStore instance
        """
        if isinstance(config, Mapping) and "params" in config:
            store_type = config.get("type", "filesystem")
            params = dict(config.get("params") or {})
            return cls.create(store_type, **params)

        return cls.create_for_app_config(config)

    @staticmethod
    def _get_value(source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, Mapping):
            return source.get(key, default)
        return getattr(source, key, default)

    @classmethod
    def create_for_app_config(cls, config: Any) -> DataStore:
        """Compose a store with backend-specific arguments only.

        This is the shared composition path for quick start, CLI, and module
        entry points.  It intentionally extracts fields directly instead of
        dumping Pydantic models, because Pydantic v1 artifacts are handled in
        the later configuration-migration phase.
        """
        datastore_config = cls._get_value(config, "datastore")
        full_config = datastore_config is not None
        if datastore_config is None:
            datastore_config = config

        store_type = cls._normalize_store_type(
            cls._get_value(datastore_config, "type", "filesystem")
        )
        if store_type == "memory":
            return cls.create("memory")

        paths_config = cls._get_value(config, "paths") if full_config else None
        fallback_path = cls._get_value(paths_config, "data_dir")
        root_path = cls._get_value(
            datastore_config,
            "root_path",
            "./data/insight_store",
        )
        base_path = cls._get_value(datastore_config, "base_path")
        explicit_root = bool(
            cls._get_value(datastore_config, "explicit_root_path", False)
        )
        root_is_non_default = (
            root_path is not None
            and str(root_path) != "./data/insight_store"
        )

        if base_path:
            effective_path = base_path
        elif explicit_root or root_is_non_default or fallback_path is None:
            effective_path = root_path
        else:
            effective_path = fallback_path
        effective_path = resolve_project_relative(effective_path or "./data")

        if store_type == "filesystem":
            return cls.create("filesystem", root_path=effective_path)

        if store_type == "sqlite":
            configured_db_path = cls._get_value(datastore_config, "db_path")
            if configured_db_path:
                db_path = resolve_project_relative(configured_db_path)
            else:
                effective = Path(effective_path)
                if effective.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
                    db_path = str(effective)
                else:
                    db_path = str(effective / "insightspike.db")

            embedding_config = (
                cls._get_value(config, "embedding") if full_config else None
            )
            vector_dim = cls._get_value(datastore_config, "vector_dim")
            if vector_dim is None:
                vector_dim = cls._get_value(embedding_config, "dimension", 384)
            return cls.create(
                "sqlite",
                db_path=db_path,
                vector_dim=int(vector_dim),
            )

        # Registered third-party stores receive no accidental filesystem args.
        return cls.create(store_type)


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

# Optional: register SQLite implementation if available
try:  # pragma: no cover - import guarded
    from .sqlite_store import SQLiteDataStore  # type: ignore

    DataStoreFactory.register("sqlite", SQLiteDataStore)  # type: ignore[arg-type]
    logger.debug("SQLiteDataStore registered in DataStoreFactory")
except Exception:
    logger.debug("SQLiteDataStore not available or failed to import")
