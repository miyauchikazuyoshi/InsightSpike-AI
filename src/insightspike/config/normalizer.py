"""Canonical configuration normalization.

Strict application entry points validate unknown fields.  This module is the
explicit compatibility boundary for older dict/object configurations: legacy
keys are migrated, unknown fields are reported with structured warnings, and
the complete surviving document is validated without rebuilding a subset of
sections.
"""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from .migration import migrate_config
from .models import InsightSpikeConfig, LLMConfig
from .pydantic_compat import (
    UnknownConfigWarning,
    model_dump_compat,
    model_validate_compat,
    prune_unknown_fields,
)

logger = logging.getLogger(__name__)


def _plain_value(value: Any) -> Any:
    """Convert a legacy namespace-like object into builtin containers."""

    if isinstance(value, BaseModel):
        return model_dump_compat(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        attributes: dict[str, Any] = {}
        for owner in reversed(type(value).__mro__):
            for name, item in vars(owner).items():
                if (
                    name.startswith("_")
                    or isinstance(item, (classmethod, staticmethod, property))
                    or callable(item)
                ):
                    continue
                attributes[name] = getattr(value, name)
        for name, item in vars(value).items():
            if not name.startswith("_") and not callable(item):
                attributes[name] = item
        return {
            name: _plain_value(item)
            for name, item in attributes.items()
        }
    return copy.deepcopy(value)


class ConfigNormalizer:
    """Normalize supported inputs into the canonical configuration model."""

    @staticmethod
    def normalize(
        config: Any,
        *,
        tolerate_unknown: bool = True,
        source: str = "legacy",
    ) -> InsightSpikeConfig:
        """Normalize a model, mapping, or legacy namespace.

        ``tolerate_unknown`` exists only for compatibility call sites.  Every
        ignored key emits :class:`UnknownConfigWarning`; strict loaders bypass
        this path and fail validation instead.
        """

        if isinstance(config, InsightSpikeConfig):
            return config

        plain = _plain_value(config)
        if not isinstance(plain, Mapping):
            raise TypeError(f"Unsupported config type: {type(config)}")

        migrated = migrate_config(
            plain,
            emit_warnings=True,
            source=source,
        ).config

        if tolerate_unknown:
            pruned = prune_unknown_fields(
                InsightSpikeConfig,
                migrated,
                source=source,
            )
            for diagnostic in pruned.diagnostics:
                warnings.warn(
                    UnknownConfigWarning(diagnostic),
                    stacklevel=2,
                )
            migrated = pruned.config

        return model_validate_compat(InsightSpikeConfig, migrated)

    @staticmethod
    def _dict_to_config(config_dict: Mapping[str, Any]) -> InsightSpikeConfig:
        """Backward-compatible alias for callers of the old private helper."""

        return ConfigNormalizer.normalize(config_dict)

    @staticmethod
    def get_llm_config(config: Any) -> LLMConfig:
        """Extract the validated LLM section from any supported input."""

        try:
            return ConfigNormalizer.normalize(config).llm
        except (TypeError, ValueError):
            logger.warning("No valid LLM config found, using defaults")
            return LLMConfig()

    @staticmethod
    def merge_configs(
        base: InsightSpikeConfig,
        override: Mapping[str, Any],
    ) -> InsightSpikeConfig:
        """Merge a higher-priority override into an existing configuration."""

        migrated_override = migrate_config(
            override,
            emit_warnings=True,
            source="override",
        ).config
        merged = ConfigNormalizer._deep_merge(
            model_dump_compat(base),
            migrated_override,
        )
        return ConfigNormalizer.normalize(
            merged,
            tolerate_unknown=True,
            source="merged",
        )

    @staticmethod
    def _deep_merge(
        base: Mapping[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge mappings, with ``override`` values winning."""

        result = copy.deepcopy(dict(base))
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], Mapping)
                and isinstance(value, Mapping)
            ):
                result[key] = ConfigNormalizer._deep_merge(
                    result[key],
                    value,
                )
            else:
                result[key] = copy.deepcopy(value)
        return result


__all__ = ["ConfigNormalizer"]
