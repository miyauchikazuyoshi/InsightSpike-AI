from __future__ import annotations

from pathlib import Path

import pytest

from insightspike.config.loader import ConfigLoader
from insightspike.config.models import InsightSpikeConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIRECTORY = REPOSITORY_ROOT / "config_examples"
EXAMPLE_FILES = tuple(sorted(EXAMPLES_DIRECTORY.glob("*.yaml")))


def test_config_example_catalog_is_not_empty() -> None:
    assert EXAMPLE_FILES


@pytest.mark.parametrize(
    "example_path",
    EXAMPLE_FILES,
    ids=lambda path: path.name,
)
def test_every_yaml_example_is_strictly_canonical(
    example_path: Path,
) -> None:
    loader = ConfigLoader()

    config = loader.load_from_file(example_path)

    assert isinstance(config, InsightSpikeConfig)
    assert loader.diagnostics == ()


def test_examples_readme_uses_current_configuration_surface() -> None:
    readme = (EXAMPLES_DIRECTORY / "README.md").read_text(encoding="utf-8")

    assert "loader._load_yaml" not in readme
    assert "INSIGHTSPIKE_GRAPH_GED_ALGORITHM" not in readme
    assert "\ncore:" not in readme
    assert "\nretrieval:" not in readme
    assert "from insightspike.config.loader import ConfigLoader" in readme
    assert "INSIGHTSPIKE_LLM__PROVIDER" in readme
