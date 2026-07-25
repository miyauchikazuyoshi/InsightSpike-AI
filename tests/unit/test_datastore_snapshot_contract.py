"""Cross-backend contracts for append and exact-snapshot persistence."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from insightspike.config.models import InsightSpikeConfig
from insightspike.core.episode import Episode
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.implementations.datastore.filesystem_store import (
    FileSystemDataStore,
)
from insightspike.implementations.datastore.memory_store import InMemoryDataStore
from insightspike.public import create_datastore


def _record(text, vector, c=0.5, metadata=None):
    return {
        "text": text,
        "vec": np.asarray(vector, dtype=np.float32),
        "c": c,
        "metadata": metadata or {},
    }


def test_filesystem_replace_is_exact_while_save_still_appends(tmp_path):
    store = FileSystemDataStore(root_path=str(tmp_path))
    old = _record("old", [1.0, 0.0])
    new = [
        _record("new-1", [0.0, 1.0]),
        _record("new-2", [0.5, 0.5]),
    ]

    assert store.save_episodes([old])
    assert store.save_episodes([old])
    assert len(store.load_episodes()) == 2

    assert store.replace_episodes(new)
    assert store.replace_episodes(new)
    loaded = store.load_episodes()

    assert [episode["text"] for episode in loaded] == ["new-1", "new-2"]
    assert len(loaded) == 2


def test_factory_and_public_api_normalize_memory_alias(tmp_path):
    config = InsightSpikeConfig(
        datastore={
            "type": "in_memory",
            "root_path": str(tmp_path / "must-not-be-passed"),
        }
    )

    assert isinstance(
        DataStoreFactory.create_for_app_config(config),
        InMemoryDataStore,
    )
    assert isinstance(
        DataStoreFactory.create_from_config(config.datastore),
        InMemoryDataStore,
    )
    assert isinstance(create_datastore("in_memory"), InMemoryDataStore)


def test_factory_uses_explicit_filesystem_root(tmp_path):
    config = InsightSpikeConfig(
        datastore={"type": "filesystem", "root_path": str(tmp_path)}
    )

    store = DataStoreFactory.create_for_app_config(config)

    assert isinstance(store, FileSystemDataStore)
    assert store.base_path == tmp_path.resolve()


def test_main_agent_state_round_trip_is_an_exact_snapshot(tmp_path):
    store = FileSystemDataStore(root_path=str(tmp_path))
    episodes = [
        Episode(
            text="alpha",
            vec=np.array([1.0, 0.0], dtype=np.float32),
            c=0.8,
            metadata={"source": "a"},
            episode_type="insight",
            selection_count=3,
            creation_time=123.0,
        ),
        Episode(
            text="beta",
            vec=np.array([0.0, 1.0], dtype=np.float32),
            c=0.6,
            metadata={"source": "b"},
        ),
    ]

    source = MainAgent.__new__(MainAgent)
    source.datastore = store
    source.l2_memory = SimpleNamespace(episodes=episodes)
    source.l3_graph = None

    assert source.save_state()
    assert source.save_state()
    assert len(store.load_episodes(namespace="agent_state")) == 2

    rebuilt = {"calls": 0}
    target = MainAgent.__new__(MainAgent)
    target.datastore = store
    target.l2_memory = SimpleNamespace(
        episodes=[],
        _rebuild_index=lambda: rebuilt.__setitem__(
            "calls",
            rebuilt["calls"] + 1,
        ),
    )
    target.l3_graph = None

    assert target.load_state()
    assert rebuilt["calls"] == 1
    assert [episode.text for episode in target.l2_memory.episodes] == [
        "alpha",
        "beta",
    ]
    assert [episode.c for episode in target.l2_memory.episodes] == pytest.approx(
        [0.8, 0.6]
    )
    assert target.l2_memory.episodes[0].metadata == {"source": "a"}
    assert target.l2_memory.episodes[0].episode_type == "insight"
    assert target.l2_memory.episodes[0].selection_count == 3
    assert target.l2_memory.episodes[0].creation_time == 123.0
    np.testing.assert_allclose(
        target.l2_memory.episodes[1].vec,
        np.array([0.0, 1.0], dtype=np.float32),
    )


def test_sqlite_replace_is_transactional_and_index_rebuilds(tmp_path):
    pytest.importorskip("aiosqlite")
    from insightspike.implementations.datastore.sqlite_store import SQLiteDataStore

    db_path = str(tmp_path / "episodes.db")
    store = SQLiteDataStore(db_path=db_path, vector_dim=2)
    original = [_record("old", [1.0, 0.0], metadata={"version": 1})]
    replacement = [
        _record("new", [0.0, 1.0], c=0.9, metadata={"version": 2})
    ]

    assert store.replace_episodes(original)
    assert store.replace_episodes(replacement)
    loaded = store.load_episodes()
    assert len(loaded) == 1
    assert loaded[0]["text"] == "new"
    assert loaded[0]["c"] == pytest.approx(0.9)
    assert loaded[0]["metadata"] == {"version": 2}

    restarted = SQLiteDataStore(db_path=db_path, vector_dim=2)
    matches = asyncio.run(
        restarted.search_episodes_by_vector(
            np.array([0.0, 1.0], dtype=np.float32),
            k=1,
            threshold=0.0,
        )
    )
    assert [match["text"] for match in matches] == ["new"]

    assert store.replace_episodes([])
    assert store.load_episodes() == []


def test_sqlite_invalid_snapshot_preserves_previous_rows(tmp_path):
    pytest.importorskip("aiosqlite")
    from insightspike.implementations.datastore.sqlite_store import SQLiteDataStore

    store = SQLiteDataStore(
        db_path=str(tmp_path / "episodes.db"),
        vector_dim=2,
    )
    assert store.replace_episodes([_record("valid", [1.0, 0.0])])

    assert not store.replace_episodes([_record("invalid", [1.0, 0.0, 0.0])])
    assert [episode["text"] for episode in store.load_episodes()] == ["valid"]
