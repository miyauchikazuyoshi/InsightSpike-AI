"""Centralized read-only configuration access for ``MainAgent``."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import logging
from typing import Any, Dict


def _read_path(config: Any, path: str, default: Any) -> Any:
    current = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return default
    return current


class AgentConfigAccess:
    """Interpret agent configuration without retaining a stale snapshot."""

    def __init__(self, *, logger: logging.Logger):
        self._logger = logger

    @staticmethod
    def section(config: Any, name: str) -> Any:
        """Return a top-level section from a mapping or model."""

        if isinstance(config, dict):
            return config.get(name)
        return getattr(config, name, None)

    @staticmethod
    def value(config: Any, path: str, default: Any = None) -> Any:
        """Read one mapping-or-model path without retaining the source."""

        return _read_path(config, path, default)

    @staticmethod
    def normalized(config: Any) -> Any:
        """Build a fresh scalar facade from the current source config."""

        from insightspike.config.normalized import NormalizedConfig

        return NormalizedConfig.from_any(config)

    @staticmethod
    def normalized_override_fields(
        baseline: Any,
        override: Any,
    ) -> Dict[str, Any]:
        """Return explicitly changed normalized fields.

        ``MainAgent._normalized_config`` is an established test and
        integration patch point.  Tracking only fields changed through that
        seam lets unrelated values continue to follow the live source config.
        """

        if (
            baseline is None
            or override is None
            or type(baseline) is not type(override)
            or not is_dataclass(baseline)
        ):
            return {}

        ignored = {"_raw", "source_type", "applied_defaults"}
        changes: Dict[str, Any] = {}
        for item in fields(baseline):
            if not item.init or item.name in ignored:
                continue
            if getattr(baseline, item.name) != getattr(
                override,
                item.name,
            ):
                changes[item.name] = getattr(override, item.name)
        return changes

    def normalized_with_overrides(
        self,
        config: Any,
        overrides: Dict[str, Any],
    ) -> Any:
        """Build a live normalized facade and apply explicit field overlays."""

        normalized = self.normalized(config)
        if not overrides:
            return normalized
        return replace(normalized, **overrides)

    def two_threshold_params(
        self,
        config: Any,
    ) -> Dict[str, Any]:
        """Read and normalize two-threshold candidate-selection settings."""

        params: Dict[str, Any] = {
            "theta_cand": 0.45,
            "theta_link": 0.35,
            "k_cap": 32,
            "top_m": None,
            "ig_denominator": "legacy",
            "use_local_normalization": False,
        }
        if config is None:
            return params

        try:
            theta_cand = _read_path(
                config,
                "metrics.theta_cand",
                params["theta_cand"],
            )
            theta_link = _read_path(
                config,
                "metrics.theta_link",
                params["theta_link"],
            )
            k_cap = _read_path(
                config,
                "metrics.candidate_cap",
                params["k_cap"],
            )
            top_m = _read_path(
                config,
                "metrics.top_m",
                params["top_m"],
            )
            ig_mode = _read_path(
                config,
                "metrics.ig_denominator",
                params["ig_denominator"],
            )
            local_norm = _read_path(
                config,
                "metrics.use_local_normalization",
                params["use_local_normalization"],
            )

            if theta_cand is not None:
                params["theta_cand"] = float(theta_cand)
            if theta_link is not None:
                params["theta_link"] = float(theta_link)
            if k_cap is not None:
                params["k_cap"] = max(1, int(k_cap))
            if top_m is not None:
                try:
                    params["top_m"] = max(1, int(top_m))
                except Exception:
                    params["top_m"] = None
            if ig_mode is not None:
                params["ig_denominator"] = str(ig_mode).lower()
            params["use_local_normalization"] = bool(local_norm)
        except Exception as exc:
            self._logger.debug(
                "Two-threshold config fallback: %s",
                exc,
            )

        if params["theta_cand"] < params["theta_link"]:
            params["theta_cand"], params["theta_link"] = (
                params["theta_link"],
                params["theta_cand"],
            )
        return params

    @staticmethod
    def learning_snapshot(normalized_config: Any) -> Dict[str, Any]:
        """Return the stable configuration subset stored with patterns."""

        if normalized_config is not None:
            return {
                "similarity_threshold": (
                    normalized_config.similarity_threshold
                ),
                "hop_limit": normalized_config.hop_limit,
                "path_decay": normalized_config.path_decay,
                "max_retrieved_docs": (
                    normalized_config.max_retrieved_docs
                ),
                "spike_ged_threshold": (
                    normalized_config.spike_ged_threshold
                ),
                "spike_ig_threshold": (
                    normalized_config.spike_ig_threshold
                ),
            }
        return {
            "similarity_threshold": 0.3,
            "hop_limit": 2,
            "path_decay": 0.7,
            "max_retrieved_docs": 10,
            "spike_ged_threshold": -0.5,
            "spike_ig_threshold": 0.2,
        }


__all__ = ["AgentConfigAccess"]
