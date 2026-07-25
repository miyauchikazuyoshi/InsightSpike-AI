"""Compatibility migration for configuration dictionaries.

This module operates on plain mappings before Pydantic validation.  It keeps
legacy spelling support out of the canonical configuration models while making
every compatibility decision observable to callers.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class ConfigMigrationDiagnostic:
    """A structured description of one compatibility migration."""

    code: str
    source: str
    old_path: str
    new_path: Optional[str]
    action: str
    message: str


@dataclass(frozen=True)
class ConfigMigrationResult:
    """Migrated configuration and the diagnostics produced while migrating it."""

    config: dict[str, Any]
    diagnostics: Tuple[ConfigMigrationDiagnostic, ...]

    @property
    def changed(self) -> bool:
        """Whether the input contained any migrated or removed values."""

        return bool(self.diagnostics)

    @property
    def data(self) -> dict[str, Any]:
        """Alias for callers that use ``data`` for a decoded configuration."""

        return self.config


class ConfigMigrationWarning(UserWarning):
    """Warning carrying the structured diagnostic that caused it."""

    def __init__(self, diagnostic: ConfigMigrationDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


_RENAMED_PATHS = (
    ("output.show_reasoning", "output.include_reasoning"),
    ("output.show_metadata", "output.include_metadata"),
    ("monitoring.enable_monitoring", "monitoring.enabled"),
    ("monitoring.track_memory_usage", "monitoring.performance_tracking"),
    ("llm.model_name", "llm.model"),
    ("embedding.model", "embedding.model_name"),
    ("datastore.path", "datastore.root_path"),
    ("paths.log_dir", "paths.logs_dir"),
    ("graph.use_multihop_gedig", "graph.enable_graph_search"),
    (
        "reasoning.episode_merge_threshold",
        "graph.episode_merge_threshold",
    ),
    (
        "reasoning.episode_split_threshold",
        "graph.episode_split_threshold",
    ),
    (
        "reasoning.episode_prune_threshold",
        "graph.episode_prune_threshold",
    ),
)

_REMOVED_PATHS = (
    "output.response_style",
    "monitoring.metrics_interval",
    "logging.format",
    "logging.file_enabled",
)


def _lookup_path(
    config: MutableMapping[str, Any], path: str
) -> tuple[bool, Any]:
    current: Any = config
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    if not isinstance(current, Mapping) or parts[-1] not in current:
        return False, None
    return True, current[parts[-1]]


def _remove_path(config: MutableMapping[str, Any], path: str) -> Any:
    current: Any = config
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    return current.pop(parts[-1])


def _set_path(config: MutableMapping[str, Any], path: str, value: Any) -> None:
    current: MutableMapping[str, Any] = config
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _diagnostic(
    *,
    code: str,
    source: str,
    old_path: str,
    new_path: Optional[str],
    action: str,
    detail: str,
) -> ConfigMigrationDiagnostic:
    return ConfigMigrationDiagnostic(
        code=code,
        source=source,
        old_path=old_path,
        new_path=new_path,
        action=action,
        message=f"Configuration migration ({source}): {detail}",
    )


def _remove_model_config_artifacts(
    value: Any,
    diagnostics: list[ConfigMigrationDiagnostic],
    source: str,
    path: str = "",
) -> None:
    """Remove Pydantic-v1 ``model_config`` serialization artifacts recursively."""

    if isinstance(value, MutableMapping):
        if "model_config" in value:
            artifact_path = (
                f"{path}.model_config" if path else "model_config"
            )
            value.pop("model_config")
            diagnostics.append(
                _diagnostic(
                    code="pydantic_v1_artifact",
                    source=source,
                    old_path=artifact_path,
                    new_path=None,
                    action="artifact_removed",
                    detail=(
                        f"removed generated Pydantic-v1 artifact "
                        f"'{artifact_path}'"
                    ),
                )
            )

        for key, child in list(value.items()):
            child_path = f"{path}.{key}" if path else str(key)
            _remove_model_config_artifacts(
                child,
                diagnostics,
                source,
                child_path,
            )
        return

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            _remove_model_config_artifacts(
                child,
                diagnostics,
                source,
                child_path,
            )


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge mappings recursively, with canonical ``override`` values winning."""

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def migrate_config(
    data: Mapping[str, Any],
    *,
    emit_warnings: bool = True,
    source: str = "config",
) -> ConfigMigrationResult:
    """Return a non-mutating migration of a decoded configuration mapping.

    Canonical values always win when a legacy and canonical key are both
    present.  The legacy key is removed in that case and a conflict diagnostic
    is emitted, so strict validation can run on the returned mapping.
    """

    if not isinstance(data, Mapping):
        raise TypeError("Configuration migration requires a mapping")

    migrated: dict[str, Any] = copy.deepcopy(dict(data))
    diagnostics: list[ConfigMigrationDiagnostic] = []

    for old_path, new_path in _RENAMED_PATHS:
        old_exists, old_value = _lookup_path(migrated, old_path)
        if not old_exists:
            continue

        new_exists, _ = _lookup_path(migrated, new_path)
        _remove_path(migrated, old_path)
        if new_exists:
            diagnostics.append(
                _diagnostic(
                    code="legacy_key_conflict",
                    source=source,
                    old_path=old_path,
                    new_path=new_path,
                    action="conflict",
                    detail=(
                        f"ignored legacy key '{old_path}' because canonical "
                        f"key '{new_path}' is also present"
                    ),
                )
            )
            continue

        _set_path(migrated, new_path, old_value)
        diagnostics.append(
            _diagnostic(
                code="legacy_key_renamed",
                source=source,
                old_path=old_path,
                new_path=new_path,
                action="migrated",
                detail=f"renamed legacy key '{old_path}' to '{new_path}'",
            )
        )

    for old_path in _REMOVED_PATHS:
        old_exists, _ = _lookup_path(migrated, old_path)
        if not old_exists:
            continue
        _remove_path(migrated, old_path)
        diagnostics.append(
            _diagnostic(
                code="legacy_key_removed",
                source=source,
                old_path=old_path,
                new_path=None,
                action="removed",
                detail=f"removed deprecated key '{old_path}'",
            )
        )

    l4_exists, l4_value = _lookup_path(migrated, "l4_config")
    if l4_exists:
        _remove_path(migrated, "l4_config")
        if isinstance(l4_value, MutableMapping) and "model_name" in l4_value:
            legacy_model = l4_value.pop("model_name")
            if "model" not in l4_value:
                l4_value["model"] = legacy_model
                diagnostics.append(
                    _diagnostic(
                        code="legacy_key_renamed",
                        source=source,
                        old_path="l4_config.model_name",
                        new_path="llm.model",
                        action="migrated",
                        detail=(
                            "renamed legacy key 'l4_config.model_name' "
                            "to 'llm.model'"
                        ),
                    )
                )
            else:
                diagnostics.append(
                    _diagnostic(
                        code="legacy_key_conflict",
                        source=source,
                        old_path="l4_config.model_name",
                        new_path="llm.model",
                        action="conflict",
                        detail=(
                            "ignored legacy key 'l4_config.model_name' "
                            "because 'l4_config.model' is also present"
                        ),
                    )
                )
        llm_exists, llm_value = _lookup_path(migrated, "llm")
        if isinstance(l4_value, Mapping) and isinstance(llm_value, Mapping):
            _set_path(migrated, "llm", _deep_merge(l4_value, llm_value))
            action = "conflict"
            code = "legacy_section_merged"
            detail = (
                "merged legacy section 'l4_config' into 'llm'; "
                "canonical llm fields took precedence"
            )
        elif not llm_exists:
            _set_path(migrated, "llm", l4_value)
            action = "migrated"
            code = "legacy_key_renamed"
            detail = "renamed legacy section 'l4_config' to 'llm'"
        else:
            action = "conflict"
            code = "legacy_key_conflict"
            detail = (
                "ignored legacy section 'l4_config' because canonical "
                "'llm' is also present and is not mergeable"
            )
        diagnostics.append(
            _diagnostic(
                code=code,
                source=source,
                old_path="l4_config",
                new_path="llm",
                action=action,
                detail=detail,
            )
        )

    datastore_type_exists, datastore_type = _lookup_path(
        migrated, "datastore.type"
    )
    if datastore_type_exists and datastore_type == "in_memory":
        _set_path(migrated, "datastore.type", "memory")
        diagnostics.append(
            _diagnostic(
                code="legacy_value_normalized",
                source=source,
                old_path="datastore.type",
                new_path="datastore.type",
                action="migrated",
                detail=(
                    "normalized legacy value 'datastore.type: in_memory' "
                    "to 'memory'"
                ),
            )
        )

    embedding_model_exists, embedding_model = _lookup_path(
        migrated,
        "embedding.model_name",
    )
    if embedding_model_exists and not embedding_model:
        default_embedding_model = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        _set_path(
            migrated,
            "embedding.model_name",
            default_embedding_model,
        )
        diagnostics.append(
            _diagnostic(
                code="legacy_value_normalized",
                source=source,
                old_path="embedding.model_name",
                new_path="embedding.model_name",
                action="migrated",
                detail=(
                    "replaced an empty legacy 'embedding.model_name' with "
                    f"'{default_embedding_model}'"
                ),
            )
        )

    environment_exists, environment = _lookup_path(migrated, "environment")
    if environment_exists and environment == "local":
        _set_path(migrated, "environment", "development")
        diagnostics.append(
            _diagnostic(
                code="legacy_value_normalized",
                source=source,
                old_path="environment",
                new_path="environment",
                action="migrated",
                detail=(
                    "normalized legacy value 'environment: local' "
                    "to 'development'"
                ),
            )
        )

    _remove_model_config_artifacts(migrated, diagnostics, source)

    if emit_warnings:
        emitted: set[tuple[str, str, Optional[str], str]] = set()
        for diagnostic in diagnostics:
            warning_key = (
                diagnostic.code,
                diagnostic.old_path,
                diagnostic.new_path,
                diagnostic.action,
            )
            if warning_key in emitted:
                continue
            emitted.add(warning_key)
            warnings.warn(
                ConfigMigrationWarning(diagnostic),
                stacklevel=2,
            )

    return ConfigMigrationResult(
        config=migrated,
        diagnostics=tuple(diagnostics),
    )


def migrate_config_dict(
    data: Mapping[str, Any],
    *,
    emit_warnings: bool = True,
    source: str = "config",
) -> dict[str, Any]:
    """Convenience wrapper returning only the migrated plain dictionary."""

    return migrate_config(
        data,
        emit_warnings=emit_warnings,
        source=source,
    ).config


__all__ = [
    "ConfigMigrationDiagnostic",
    "ConfigMigrationResult",
    "ConfigMigrationWarning",
    "migrate_config",
    "migrate_config_dict",
]
