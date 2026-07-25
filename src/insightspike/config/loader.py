"""Configuration loading, migration, validation, and persistence."""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .migration import ConfigMigrationDiagnostic, migrate_config
from .models import InsightSpikeConfig
from .presets import ConfigPresets
from .pydantic_compat import model_dump_compat, model_validate_compat

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load all supported sources into one strict canonical configuration."""

    ENV_PREFIX = "INSIGHTSPIKE_"
    CONFIG_PATH_ENV = "INSIGHTSPIKE_CONFIG_PATH"
    DEFAULT_DATASTORE_ROOT = "./data/insight_store"

    def __init__(self):
        self._config: Optional[InsightSpikeConfig] = None
        self._config_path: Optional[Path] = None
        self._diagnostics: list[ConfigMigrationDiagnostic] = []

    @property
    def diagnostics(self) -> tuple[ConfigMigrationDiagnostic, ...]:
        """Structured migrations produced by the latest load operation."""

        return tuple(self._diagnostics)

    def load(
        self,
        config_path: Optional[Union[str, Path]] = None,
        preset: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
    ) -> InsightSpikeConfig:
        """Load sources with priority ``preset < file < env < overrides``."""

        self._diagnostics = []
        # A loader may be reused.  Do not let a file selected by an earlier
        # load become the implicit save target for a later preset/default load.
        self._config_path = None
        preset_config = self._migrate_source(
            ConfigPresets.get_preset(preset) if preset else {},
            source="preset",
        )
        # An explicitly selected preset is self-contained unless a file path
        # (argument or environment variable) is also explicitly selected.
        should_discover_file = (
            preset is None
            or config_path is not None
            or bool(os.getenv(self.CONFIG_PATH_ENV))
        )
        file_config = self._migrate_source(
            self._load_from_file(config_path) if should_discover_file else {},
            source="file",
        )
        env_config = self._migrate_source(
            self._load_from_env(),
            source="environment",
        )
        override_config = self._migrate_source(
            overrides or {},
            source="override",
        )

        config_dict: dict[str, Any] = {}
        for source_config in (
            preset_config,
            file_config,
            env_config,
            override_config,
        ):
            config_dict = self._deep_merge(config_dict, source_config)

        explicit_root = self._resolve_explicit_root(
            file_config,
            env_config,
            override_config,
        )
        if explicit_root is not None:
            config_dict.setdefault("datastore", {})[
                "explicit_root_path"
            ] = explicit_root

        self._config = self._validate(config_dict)
        return self._config

    def _migrate_source(
        self,
        data: Mapping[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        result = migrate_config(
            data,
            emit_warnings=True,
            source=source,
        )
        self._diagnostics.extend(result.diagnostics)
        return result.config

    @staticmethod
    def _validate(data: Mapping[str, Any]) -> InsightSpikeConfig:
        return model_validate_compat(InsightSpikeConfig, data)

    @staticmethod
    def _explicit_root_intent(
        data: Mapping[str, Any],
    ) -> Optional[bool]:
        """Return a source's explicit datastore-root intent, if expressed.

        ``explicit_root_path`` is an internal round-trip marker accepted for
        compatibility.  Otherwise, the presence of ``root_path`` means that
        the source intentionally selected that path.
        """

        datastore = data.get("datastore", {})
        if not isinstance(datastore, Mapping):
            return None
        if "explicit_root_path" in datastore:
            return bool(datastore["explicit_root_path"])
        if datastore.get("root_path") is not None:
            return True
        return None

    @classmethod
    def _resolve_explicit_root(
        cls,
        *sources: Mapping[str, Any],
    ) -> Optional[bool]:
        """Resolve root intent in source-priority order."""

        resolved: Optional[bool] = None
        for source in sources:
            intent = cls._explicit_root_intent(source)
            if intent is not None:
                resolved = intent
        return resolved

    @classmethod
    def _has_explicit_root(cls, data: Mapping[str, Any]) -> bool:
        """Compatibility helper retained for callers of the old predicate."""

        return cls._explicit_root_intent(data) is True

    def _load_from_file(
        self,
        config_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        """Decode a YAML/JSON document without applying compatibility rules."""

        path = self._resolve_config_path(config_path)
        if path is None:
            return {}
        if not path.exists():
            logger.debug("Config file not found: %s", path)
            return {}

        self._config_path = path
        logger.info("Loading config from: %s", path)
        with path.open("r", encoding="utf-8") as stream:
            if path.suffix.lower() in {".yaml", ".yml"}:
                decoded = yaml.safe_load(stream)
            elif path.suffix.lower() == ".json":
                decoded = json.load(stream)
            else:
                content = stream.read()
                try:
                    decoded = json.loads(content)
                except json.JSONDecodeError:
                    decoded = yaml.safe_load(content)

        if decoded is None:
            return {}
        if not isinstance(decoded, Mapping):
            raise TypeError(
                f"Configuration document must contain a mapping: {path}"
            )
        return copy.deepcopy(dict(decoded))

    def _resolve_config_path(
        self,
        config_path: Optional[Union[str, Path]],
    ) -> Optional[Path]:
        if config_path is not None:
            return Path(config_path)

        env_path = os.getenv(self.CONFIG_PATH_ENV)
        if env_path:
            return Path(env_path)

        for default_path in (
            Path("config.yaml"),
            Path(".insightspike.yaml"),
            Path("config.json"),
        ):
            if default_path.exists():
                return default_path
        return None

    def _load_from_env(self) -> dict[str, Any]:
        """Decode explicitly supported environment variables."""

        config: dict[str, Any] = {}
        # The second tuple item is the field's parser. Content-based coercion
        # of paths
        # and model names (for example "123") behaves differently between
        # Pydantic v1 and v2, so string fields must remain strings.
        env_mappings = {
            f"{self.ENV_PREFIX}LLM__PROVIDER": ("llm.provider", str),
            f"{self.ENV_PREFIX}LLM__MODEL": ("llm.model", str),
            f"{self.ENV_PREFIX}LLM__TEMPERATURE": (
                "llm.temperature",
                float,
            ),
            f"{self.ENV_PREFIX}LLM__MAX_TOKENS": ("llm.max_tokens", int),
            f"{self.ENV_PREFIX}MEMORY__EPISODIC_MEMORY_CAPACITY": (
                "memory.episodic_memory_capacity",
                int,
            ),
            f"{self.ENV_PREFIX}MEMORY__MAX_RETRIEVED_DOCS": (
                "memory.max_retrieved_docs",
                int,
            ),
            f"{self.ENV_PREFIX}DATASTORE__TYPE": ("datastore.type", str),
            f"{self.ENV_PREFIX}DATASTORE__ROOT_PATH": (
                "datastore.root_path",
                str,
            ),
            f"{self.ENV_PREFIX}DATASTORE__DB_PATH": (
                "datastore.db_path",
                str,
            ),
            f"{self.ENV_PREFIX}ENVIRONMENT": ("environment", str),
            f"{self.ENV_PREFIX}LOGGING__LEVEL": ("logging.level", str),
            f"{self.ENV_PREFIX}LOGGING__FILE_PATH": (
                "logging.file_path",
                str,
            ),
            # Legacy one-level names.
            f"{self.ENV_PREFIX}MODEL_NAME": (
                "embedding.model_name",
                str,
            ),
            f"{self.ENV_PREFIX}DATA_DIR": ("paths.data_dir", str),
            f"{self.ENV_PREFIX}LOG_DIR": ("paths.logs_dir", str),
        }

        for env_var, (config_path, parser) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested(
                    config,
                    config_path,
                    parser(value),
                )
        return config

    @staticmethod
    def _coerce_env_value(value: str) -> Any:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def _deep_merge(
        base: Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Deep merge two mappings without mutating either input."""

        result = copy.deepcopy(dict(base))
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], Mapping)
                and isinstance(value, Mapping)
            ):
                result[key] = ConfigLoader._deep_merge(
                    result[key],
                    value,
                )
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _set_nested(
        data: dict[str, Any],
        path: str,
        value: Any,
    ) -> None:
        keys = path.split(".")
        current = data
        for key in keys[:-1]:
            child = current.get(key)
            if not isinstance(child, dict):
                child = {}
                current[key] = child
            current = child
        current[keys[-1]] = value

    @staticmethod
    def _is_float(value: str) -> bool:
        """Retained for compatibility with callers of the old helper."""

        try:
            float(value)
            return "." in value
        except ValueError:
            return False

    def load_from_file(
        self,
        path: Union[str, Path],
    ) -> InsightSpikeConfig:
        """Load, migrate, and strictly validate one configuration file."""

        self._diagnostics = []
        self._config_path = None
        config_dict = self._migrate_source(
            self._load_from_file(path),
            source="file",
        )
        explicit_root = self._explicit_root_intent(config_dict)
        if explicit_root is not None:
            config_dict.setdefault("datastore", {})[
                "explicit_root_path"
            ] = explicit_root
        self._config = self._validate(config_dict)
        return self._config

    def _apply_env_overrides(
        self,
        config: InsightSpikeConfig,
    ) -> InsightSpikeConfig:
        """Apply validated environment overrides to an existing config."""

        env_config = self._migrate_source(
            self._load_from_env(),
            source="environment",
        )
        if not env_config:
            return config
        config_dict = self._deep_merge(
            model_dump_compat(config),
            env_config,
        )
        explicit_root = self._explicit_root_intent(env_config)
        if explicit_root is None:
            explicit_root = config.datastore.explicit_root_path
        config_dict.setdefault("datastore", {})[
            "explicit_root_path"
        ] = explicit_root
        return self._validate(config_dict)

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Persist the current configuration as portable YAML or JSON."""

        if self._config is None:
            raise ValueError("No configuration loaded")

        save_path = Path(path) if path is not None else self._config_path
        if save_path is None:
            save_path = Path("config.yaml")

        config_dict = model_dump_compat(self._config, mode="json")
        datastore = config_dict.get("datastore")
        if (
            isinstance(datastore, dict)
            and not self._config.datastore.explicit_root_path
            and datastore.get("root_path") == self.DEFAULT_DATASTORE_ROOT
        ):
            # Omitting the implicit default is what preserves the distinction
            # between "use paths.data_dir" and "the user selected root_path".
            # Validation restores the same default value on reload.
            datastore.pop("root_path", None)
        with save_path.open("w", encoding="utf-8") as stream:
            if save_path.suffix.lower() in {".yaml", ".yml"}:
                yaml.safe_dump(
                    config_dict,
                    stream,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            else:
                json.dump(
                    config_dict,
                    stream,
                    indent=2,
                    ensure_ascii=False,
                )
                stream.write("\n")

        logger.info("Configuration saved to: %s", save_path)


_loader = ConfigLoader()


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    preset: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> InsightSpikeConfig:
    """Load configuration using the process-wide loader."""

    return _loader.load(config_path, preset, overrides)


def get_config() -> InsightSpikeConfig:
    """Return the current configuration, loading defaults when needed."""

    if _loader._config is None:
        _loader.load()
    assert _loader._config is not None
    return _loader._config


__all__ = ["ConfigLoader", "get_config", "load_config"]
