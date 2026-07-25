from __future__ import annotations

import copy
import warnings

import pytest

from insightspike.config.migration import (
    ConfigMigrationWarning,
    migrate_config,
    migrate_config_dict,
)


def test_migrates_all_supported_legacy_keys_and_values() -> None:
    source = {
        "output": {
            "response_style": "concise",
            "show_reasoning": False,
            "show_metadata": False,
        },
        "monitoring": {
            "enable_monitoring": True,
            "track_memory_usage": True,
            "metrics_interval": 10,
        },
        "logging": {
            "format": "structured",
            "file_enabled": True,
        },
        "llm": {"model_name": "legacy-llm"},
        "embedding": {"model": "legacy-embedding"},
        "datastore": {
            "path": "/tmp/legacy-store",
            "type": "in_memory",
        },
    }

    result = migrate_config(source, emit_warnings=False)

    assert result.changed
    assert result.config == {
        "output": {
            "include_reasoning": False,
            "include_metadata": False,
        },
        "monitoring": {
            "enabled": True,
            "performance_tracking": True,
        },
        "logging": {},
        "llm": {"model": "legacy-llm"},
        "embedding": {"model_name": "legacy-embedding"},
        "datastore": {
            "root_path": "/tmp/legacy-store",
            "type": "memory",
        },
    }
    assert len(result.diagnostics) == 12
    assert {diagnostic.action for diagnostic in result.diagnostics} == {
        "migrated",
        "removed",
    }


@pytest.mark.parametrize(
    ("legacy_path", "canonical_path", "section"),
    [
        ("show_reasoning", "include_reasoning", "output"),
        ("show_metadata", "include_metadata", "output"),
        ("enable_monitoring", "enabled", "monitoring"),
        ("track_memory_usage", "performance_tracking", "monitoring"),
        ("model_name", "model", "llm"),
        ("model", "model_name", "embedding"),
        ("path", "root_path", "datastore"),
    ],
)
def test_canonical_value_wins_on_legacy_key_conflict(
    legacy_path: str,
    canonical_path: str,
    section: str,
) -> None:
    source = {
        section: {
            legacy_path: "legacy",
            canonical_path: "canonical",
        }
    }

    result = migrate_config(source, emit_warnings=False)

    assert result.config[section][canonical_path] == "canonical"
    assert legacy_path not in result.config[section]
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "legacy_key_conflict"
    assert result.diagnostics[0].action == "conflict"


def test_migration_does_not_mutate_input_or_nested_values() -> None:
    source = {
        "llm": {"model_name": "legacy"},
        "custom": {"nested": [{"value": 1}]},
    }
    before = copy.deepcopy(source)

    result = migrate_config(source, emit_warnings=False)
    result.config["custom"]["nested"][0]["value"] = 2

    assert source == before


def test_removes_model_config_artifacts_recursively() -> None:
    source = {
        "model_config": {"extra": "forbid"},
        "wake_sleep": {
            "search": {
                "model_config": {"json_schema_extra": {"example": True}},
                "strategy": "radius",
            },
            "items": [
                {"model_config": {"extra": "ignore"}, "enabled": True}
            ],
        },
    }

    result = migrate_config(source, emit_warnings=False)

    assert result.config == {
        "wake_sleep": {
            "search": {"strategy": "radius"},
            "items": [{"enabled": True}],
        }
    }
    assert [diagnostic.old_path for diagnostic in result.diagnostics] == [
        "model_config",
        "wake_sleep.search.model_config",
        "wake_sleep.items[0].model_config",
    ]
    assert all(
        diagnostic.action == "artifact_removed"
        for diagnostic in result.diagnostics
    )


def test_emits_structured_user_warning_for_each_changed_path() -> None:
    source = {
        "output": {
            "show_reasoning": False,
            "response_style": "concise",
        }
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = migrate_config(source)

    assert len(caught) == len(result.diagnostics) == 2
    assert all(
        isinstance(item.message, ConfigMigrationWarning) for item in caught
    )
    assert all(issubclass(item.category, UserWarning) for item in caught)
    assert [
        item.message.diagnostic for item in caught
    ] == list(result.diagnostics)


def test_canonical_input_is_a_noop_without_warnings() -> None:
    source = {
        "llm": {"model": "canonical"},
        "embedding": {"model_name": "canonical"},
        "datastore": {"type": "memory", "root_path": "./data"},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = migrate_config(source)

    assert result.config == source
    assert result.diagnostics == ()
    assert not result.changed
    assert caught == []


def test_dict_helper_and_invalid_input_contract() -> None:
    assert migrate_config_dict(
        {"datastore": {"type": "in_memory"}},
        emit_warnings=False,
    ) == {"datastore": {"type": "memory"}}

    with pytest.raises(TypeError, match="requires a mapping"):
        migrate_config(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_legacy_l4_section_merges_below_canonical_llm_fields() -> None:
    result = migrate_config(
        {
            "l4_config": {
                "provider": "mock",
                "model_name": "legacy-model",
            },
            "llm": {
                "model": "canonical-model",
                "temperature": 0.2,
            },
        },
        emit_warnings=False,
        source="legacy-test",
    )

    assert result.config["llm"] == {
        "provider": "mock",
        "model": "canonical-model",
        "temperature": 0.2,
    }
    assert all(
        diagnostic.source == "legacy-test"
        for diagnostic in result.diagnostics
    )
    assert {item.code for item in result.diagnostics} == {
        "legacy_key_renamed",
        "legacy_section_merged",
    }


def test_empty_legacy_embedding_model_is_normalized() -> None:
    result = migrate_config(
        {"embedding": {"model_name": None}},
        emit_warnings=False,
    )

    assert result.config["embedding"]["model_name"] == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_reasoning_episode_thresholds_move_to_graph_with_conflict_safety() -> None:
    result = migrate_config(
        {
            "reasoning": {
                "episode_merge_threshold": 0.75,
                "episode_split_threshold": 0.2,
                "episode_prune_threshold": 0.05,
            },
            "graph": {
                "episode_merge_threshold": 0.9,
            },
        },
        emit_warnings=False,
    )

    assert result.config["graph"]["episode_merge_threshold"] == 0.9
    assert result.config["graph"]["episode_split_threshold"] == 0.2
    assert result.config["graph"]["episode_prune_threshold"] == 0.05
    assert result.config["reasoning"] == {}
    assert [item.action for item in result.diagnostics] == [
        "conflict",
        "migrated",
        "migrated",
    ]
