"""Canonical conversion between runtime episodes and persistence records."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Dict, Union

import numpy as np

from ...core.episode import Episode

EpisodeLike = Union[Episode, Mapping[str, Any]]


def episode_to_record(episode: EpisodeLike) -> Dict[str, Any]:
    """Return a backend-neutral episode record without mutating the input."""
    if isinstance(episode, Episode):
        source: Mapping[str, Any] = {
            "text": episode.text,
            "vec": episode.vec,
            "c": episode.c,
            "timestamp": episode.timestamp,
            "metadata": episode.metadata,
            "episode_type": episode.episode_type,
            "selection_count": episode.selection_count,
            "creation_time": episode.creation_time,
            "confidence": episode.confidence,
        }
    elif isinstance(episode, Mapping):
        source = episode
    else:
        raise TypeError(
            f"Expected Episode or mapping, got {type(episode).__name__}"
        )

    vector = source.get("vec", source.get("embedding"))
    if vector is None:
        raise ValueError("Episode persistence record requires 'vec' or 'embedding'")

    c_value = source.get(
        "c",
        source.get("c_value", source.get("confidence", 0.5)),
    )
    record: Dict[str, Any] = {
        "text": str(source["text"]),
        "vec": np.asarray(vector, dtype=np.float32).reshape(-1),
        "c": float(c_value),
        "c_value": float(c_value),
        "timestamp": float(source.get("timestamp") or time.time()),
        "metadata": dict(source.get("metadata") or {}),
        "episode_type": source.get("episode_type", "experience"),
        "selection_count": int(source.get("selection_count", 0)),
        "creation_time": float(source.get("creation_time") or time.time()),
    }
    if source.get("id") is not None:
        record["id"] = str(source["id"])
    return record


def record_to_episode(record: EpisodeLike) -> Episode:
    """Create a runtime :class:`Episode` from any supported store record."""
    normalized = episode_to_record(record)
    return Episode(
        text=normalized["text"],
        vec=normalized["vec"],
        c=normalized["c"],
        timestamp=normalized["timestamp"],
        metadata=normalized["metadata"],
        episode_type=normalized["episode_type"],
        selection_count=normalized["selection_count"],
        creation_time=normalized["creation_time"],
        confidence=normalized["c"],
    )


__all__ = ["EpisodeLike", "episode_to_record", "record_to_episode"]
