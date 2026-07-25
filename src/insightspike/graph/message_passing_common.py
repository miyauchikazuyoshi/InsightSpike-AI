"""Shared contracts for message-passing implementations."""

from __future__ import annotations

import warnings


MESSAGE_AGGREGATIONS = frozenset(
    {"weighted_mean", "mean", "max"}
)
DEPRECATED_MESSAGE_AGGREGATIONS = frozenset({"attention"})


def normalize_message_aggregation(value: str) -> str:
    """Validate an aggregation name and resolve compatibility aliases.

    ``attention`` was accepted by configuration but historically executed the
    simple-mean fallback in both implementations. Preserve those numerics
    during a deprecation window while making the behavior explicit.
    """

    if value in DEPRECATED_MESSAGE_AGGREGATIONS:
        warnings.warn(
            "message-passing aggregation='attention' has always used "
            "simple mean; use aggregation='mean' explicitly",
            FutureWarning,
            stacklevel=3,
        )
        return "mean"
    if value not in MESSAGE_AGGREGATIONS:
        supported = sorted(
            MESSAGE_AGGREGATIONS
            | DEPRECATED_MESSAGE_AGGREGATIONS
        )
        raise ValueError(
            "Unknown message-passing aggregation "
            f"{value!r}; expected one of {supported}"
        )
    return value


__all__ = [
    "DEPRECATED_MESSAGE_AGGREGATIONS",
    "MESSAGE_AGGREGATIONS",
    "normalize_message_aggregation",
]
