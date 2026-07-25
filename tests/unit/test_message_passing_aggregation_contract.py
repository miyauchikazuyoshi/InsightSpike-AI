"""Cloud-safe message-passing aggregation contracts."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from insightspike.config.message_passing_config import (
    MessagePassingConfig,
)
from insightspike.graph.message_passing_common import (
    normalize_message_aggregation,
)


@pytest.mark.parametrize(
    "method",
    ["weighted_mean", "mean", "max"],
)
def test_canonical_aggregation_names_are_preserved(method: str) -> None:
    assert normalize_message_aggregation(method) == method
    assert MessagePassingConfig(aggregation=method).aggregation == method


def test_attention_is_an_explicit_deprecated_mean_alias() -> None:
    assert MessagePassingConfig(
        aggregation="attention"
    ).aggregation == "attention"

    with pytest.warns(
        FutureWarning,
        match="has always used simple mean",
    ):
        assert normalize_message_aggregation("attention") == "mean"


def test_unknown_aggregation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown.*aggregation"):
        normalize_message_aggregation("typo")

    with pytest.raises(ValidationError):
        MessagePassingConfig(aggregation="typo")
