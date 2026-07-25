from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from insightspike.config.loader import ConfigLoader
from insightspike.config.index_config import (
    IndexFeatureFlags,
    IntegratedIndexConfig,
)
from insightspike.config.migration import ConfigMigrationWarning
from insightspike.config.models import InsightSpikeConfig
from insightspike.config.normalizer import ConfigNormalizer
from insightspike.config.pydantic_compat import (
    UnknownConfigWarning,
    model_dump_compat,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _contains_key(value, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(item, target) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, target) for item in value)
    return False


def test_model_dump_has_no_pydantic_v1_artifacts_at_any_depth() -> None:
    config = InsightSpikeConfig(
        graph={
            "message_passing": {"alpha": 0.4},
        },
        wake_sleep={
            "wake": {"search": {"method": "donut"}},
        },
    )

    assert not _contains_key(model_dump_compat(config), "model_config")
    assert not _contains_key(config.dict(), "model_config")


@pytest.mark.parametrize(
    "payload",
    [
        {"graph": {"similiarity_threshold": 0.7}},
        {"metrics": {"spectral_evaluation": {"weigth": 0.4}}},
        {"wake_sleep": {"wake": {"search": {"max_neigbors": 4}}}},
        {"graph": {"message_passing": {"iteratons": 2}}},
    ],
)
def test_strict_models_reject_unknown_nested_keys(payload) -> None:
    with pytest.raises(ValidationError):
        InsightSpikeConfig(**payload)


def test_hybrid_weight_field_error_does_not_add_a_spurious_sum_error() -> None:
    with pytest.raises(ValidationError) as caught:
        InsightSpikeConfig(
            graph={
                "hybrid_weights": {
                    "structure": "not-a-number",
                }
            }
        )

    assert "Weights must sum" not in str(caught.value)


def test_legacy_normalizer_reports_unknown_path_and_migrates_l4() -> None:
    payload = {
        "l4_config": {
            "provider": "mock",
            "model": "legacy-model",
        },
        "graph": {
            "similiarity_threshold": 0.9,
        },
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = ConfigNormalizer.normalize(payload)

    assert config.llm.provider == "mock"
    assert config.llm.model == "legacy-model"
    unknowns = [
        item.message.diagnostic
        for item in caught
        if isinstance(item.message, UnknownConfigWarning)
    ]
    assert [item.dotted_path for item in unknowns] == [
        "graph.similiarity_threshold"
    ]
    assert any(
        isinstance(item.message, ConfigMigrationWarning)
        and item.message.diagnostic.old_path == "l4_config"
        for item in caught
    )


def test_normalizer_preserves_all_canonical_sections() -> None:
    config = ConfigNormalizer.normalize(
        {
            "environment": "research",
            "pre_warm_models": False,
            "llm": {"provider": "mock", "model": "preserved"},
            "embedding": {"dimension": 128},
            "memory": {"max_episodes": 321},
            "graph": {
                "candidate_topk": 7,
                "message_passing": {"alpha": 0.6},
            },
            "monitoring": {"enabled": True},
            "logging": {"level": "ERROR"},
            "paths": {"logs_dir": "~/custom-insight-logs"},
            "processing": {"mode": "batch"},
            "output": {"include_reasoning": False},
            "datastore": {"type": "sqlite", "db_path": "state.sqlite3"},
            "metrics": {"theta_cand": 0.62},
            "reasoning": {"max_cycles": 17},
            "performance": {"parallel_workers": 3},
            "vector_search": {"backend": "numpy"},
            "wake_sleep": {
                "mode": "auto",
                "wake": {"search": {"method": "donut"}},
            },
            "gedig": {"mode": "ab"},
        }
    )

    assert config.environment == "research"
    assert config.pre_warm_models is False
    assert config.llm.model == "preserved"
    assert config.embedding.dimension == 128
    assert config.memory.max_episodes == 321
    assert config.graph.candidate_topk == 7
    assert config.graph.message_passing is not None
    assert config.graph.message_passing.alpha == 0.6
    assert config.monitoring.enabled is True
    assert config.logging.level == "ERROR"
    assert config.paths.logs_dir == Path("~/custom-insight-logs").expanduser()
    assert config.processing.mode == "batch"
    assert config.output.include_reasoning is False
    assert config.datastore.db_path == "state.sqlite3"
    assert config.metrics.theta_cand == 0.62
    assert config.reasoning.max_cycles == 17
    assert config.performance.parallel_workers == 3
    assert config.vector_search.backend == "numpy"
    assert config.wake_sleep.mode == "auto"
    assert config.wake_sleep.wake.search.method == "donut"
    assert config.gedig.mode == "ab"


def test_each_source_is_migrated_before_priority_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        "llm:\n  model_name: file-model\n"
        "datastore:\n  path: file-store\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INSIGHTSPIKE_LLM__MODEL", "env-model")

    loader = ConfigLoader()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConfigMigrationWarning)
        config = loader.load(
            config_path=config_path,
            preset="development",
            overrides={
                "llm": {"model_name": "override-model"},
                "datastore": {"path": "override-store"},
            },
        )

    assert config.llm.model == "override-model"
    assert config.datastore.root_path == "override-store"
    assert config.datastore.explicit_root_path is True
    assert {item.source for item in loader.diagnostics} == {
        "file",
        "override",
    }


def test_explicit_preset_does_not_absorb_an_ambient_default_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.yaml").write_text(
        "environment: production\nllm:\n  provider: openai\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = ConfigLoader().load(preset="development")

    assert config.environment == "development"
    assert config.llm.provider == "mock"


def test_current_repository_config_is_canonical_and_strictly_loadable() -> None:
    loader = ConfigLoader()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = loader.load_from_file(REPOSITORY_ROOT / "config.yaml")

    assert config.output.include_reasoning is False
    assert config.output.include_metadata is False
    assert config.monitoring.enabled is True
    assert config.monitoring.performance_tracking is True
    assert not any(
        isinstance(item.message, ConfigMigrationWarning) for item in caught
    )


def test_loader_rejects_unknown_nested_key(tmp_path: Path) -> None:
    config_path = tmp_path / "typo.yaml"
    config_path.write_text(
        "graph:\n  similiarity_threshold: 0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        ConfigLoader().load_from_file(config_path)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (IntegratedIndexConfig, {"similiarity_threshold": 0.9}),
        (IndexFeatureFlags, {"rollout_percent": 50}),
    ],
)
def test_standalone_index_configs_are_strict(model, payload) -> None:
    with pytest.raises(ValidationError):
        model(**payload)


def test_string_environment_values_are_not_content_coerced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSIGHTSPIKE_LLM__MODEL", "123")
    monkeypatch.setenv("INSIGHTSPIKE_DATASTORE__ROOT_PATH", "456")
    monkeypatch.setenv("INSIGHTSPIKE_DATASTORE__DB_PATH", "789")
    monkeypatch.setenv("INSIGHTSPIKE_LOGGING__FILE_PATH", "101112")

    config = ConfigLoader().load(preset="development")

    assert config.llm.model == "123"
    assert config.datastore.root_path == "456"
    assert config.datastore.db_path == "789"
    assert config.logging.file_path == "101112"


def test_integer_environment_values_reject_fractional_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSIGHTSPIKE_LLM__MAX_TOKENS", "1.5")

    with pytest.raises(ValueError):
        ConfigLoader().load(preset="development")


def test_save_is_safe_yaml_and_round_trips(tmp_path: Path) -> None:
    source_path = tmp_path / "source.yaml"
    saved_path = tmp_path / "saved.yaml"
    source_path.write_text(
        "environment: custom\n"
        "paths:\n  logs_dir: ~/roundtrip-logs\n"
        "graph:\n  message_passing:\n    alpha: 0.55\n",
        encoding="utf-8",
    )

    loader = ConfigLoader()
    before = loader.load_from_file(source_path)
    loader.save(saved_path)

    saved_text = saved_path.read_text(encoding="utf-8")
    decoded = yaml.safe_load(saved_text)
    after = ConfigLoader().load_from_file(saved_path)

    assert isinstance(decoded, dict)
    assert "!!python" not in saved_text
    assert not _contains_key(decoded, "model_config")
    assert "root_path" not in decoded["datastore"]
    assert before.datastore.explicit_root_path is False
    assert after.datastore.explicit_root_path is False
    assert model_dump_compat(before, mode="json") == model_dump_compat(
        after,
        mode="json",
    )


def test_save_round_trip_preserves_an_explicit_default_datastore_root(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.yaml"
    saved_path = tmp_path / "saved.yaml"
    source_path.write_text(
        "paths:\n  data_dir: ./alternative-data\n"
        "datastore:\n  root_path: ./data/insight_store\n",
        encoding="utf-8",
    )

    loader = ConfigLoader()
    before = loader.load_from_file(source_path)
    loader.save(saved_path)
    decoded = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    after = ConfigLoader().load_from_file(saved_path)

    assert before.datastore.explicit_root_path is True
    assert decoded["datastore"]["root_path"] == "./data/insight_store"
    assert after.datastore.explicit_root_path is True


def test_reusing_loader_does_not_retain_a_previous_save_target(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.yaml"
    source_path.write_text("environment: custom\n", encoding="utf-8")

    loader = ConfigLoader()
    loader.load_from_file(source_path)
    assert loader._config_path == source_path

    loader.load(preset="development")

    assert loader._config_path is None


def test_missing_file_load_does_not_retain_a_previous_save_target(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.yaml"
    source_path.write_text("environment: custom\n", encoding="utf-8")

    loader = ConfigLoader()
    loader.load_from_file(source_path)
    loader.load_from_file(tmp_path / "missing.yaml")

    assert loader._config_path is None
