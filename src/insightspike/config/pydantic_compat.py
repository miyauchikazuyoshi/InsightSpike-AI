"""Small, explicit compatibility layer for Pydantic v1 and v2."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Optional, Tuple, Union, get_args, get_origin

import pydantic
from pydantic import BaseModel

PYDANTIC_V2 = hasattr(BaseModel, "model_validate")

if PYDANTIC_V2:
    from pydantic import ConfigDict


if PYDANTIC_V2:

    class StrictConfigModel(BaseModel):
        """Base for canonical config models on Pydantic v2."""

        model_config = ConfigDict(
            extra="forbid",
        )

    class AssignableStrictConfigModel(StrictConfigModel):
        """Strict model that also validates direct field assignment."""

        model_config = ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

else:

    class StrictConfigModel(BaseModel):
        """Base for canonical config models on Pydantic v1."""

        class Config:
            extra = "forbid"

    class AssignableStrictConfigModel(StrictConfigModel):
        """Strict model that also validates direct field assignment."""

        class Config(StrictConfigModel.Config):
            validate_assignment = True


def model_validate_compat(model_cls, data: Any):
    """Validate ``data`` with a model class on either major Pydantic API."""
    if PYDANTIC_V2:
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def _remove_model_config_artifacts(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _remove_model_config_artifacts(item)
            for key, item in value.items()
            if key != "model_config"
        }
    if isinstance(value, list):
        return [_remove_model_config_artifacts(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_remove_model_config_artifacts(item) for item in value)
    return value


def json_safe(value: Any) -> Any:
    """Convert config values into JSON/YAML-safe builtin values."""
    if isinstance(value, BaseModel):
        return model_dump_compat(value, mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return value.item()
    return value


def model_dump_compat(
    model: BaseModel,
    *,
    mode: str = "python",
    exclude_none: bool = False,
) -> dict[str, Any]:
    """Dump a model without leaking Pydantic-v1 ``model_config`` artifacts."""
    if PYDANTIC_V2:
        dumped = model.model_dump(mode=mode, exclude_none=exclude_none)
    else:
        dumped = model.dict(exclude_none=exclude_none)
    cleaned = _remove_model_config_artifacts(dumped)
    return json_safe(cleaned) if mode == "json" else cleaned


def model_fields_compat(model_or_cls: Any) -> Mapping[str, Any]:
    """Return field definitions using the active Pydantic API."""
    model_cls = (
        model_or_cls
        if isinstance(model_or_cls, type)
        else type(model_or_cls)
    )
    if PYDANTIC_V2:
        return model_cls.model_fields
    return model_cls.__fields__


def _field_annotation(field: Any) -> Any:
    if PYDANTIC_V2:
        return field.annotation
    return getattr(field, "outer_type_", getattr(field, "type_", Any))


def _nested_model_type(annotation: Any) -> Optional[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin in (Union, UnionType, list, tuple, set, Sequence):
        for argument in get_args(annotation):
            nested = _nested_model_type(argument)
            if nested is not None:
                return nested
    return None


def _is_model_sequence(annotation: Any) -> bool:
    return get_origin(annotation) in (list, tuple, set, Sequence)


@dataclass(frozen=True)
class UnknownConfigDiagnostic:
    """Structured report for one field removed on a legacy-tolerant path."""

    code: str
    severity: str
    source: str
    dotted_path: str
    replacement: Optional[str]
    message: str


@dataclass(frozen=True)
class UnknownFieldPruneResult:
    """Cleaned legacy data and all unknown-field diagnostics."""

    config: dict[str, Any]
    diagnostics: Tuple[UnknownConfigDiagnostic, ...]


class UnknownConfigWarning(UserWarning):
    """Warning carrying a structured unknown-field diagnostic."""

    def __init__(self, diagnostic: UnknownConfigDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


def prune_unknown_fields(
    model_cls: type[BaseModel],
    data: Mapping[str, Any],
    *,
    source: str = "legacy",
) -> UnknownFieldPruneResult:
    """Remove unknown keys recursively for explicitly tolerant legacy callers."""
    diagnostics: list[UnknownConfigDiagnostic] = []

    def prune(
        current_model: type[BaseModel],
        current_data: Mapping[str, Any],
        prefix: str,
    ) -> dict[str, Any]:
        fields = model_fields_compat(current_model)
        cleaned: dict[str, Any] = {}
        for key, value in current_data.items():
            dotted_path = f"{prefix}.{key}" if prefix else str(key)
            field = fields.get(key)
            if field is None:
                diagnostics.append(
                    UnknownConfigDiagnostic(
                        code="unknown_config_key",
                        severity="warning",
                        source=source,
                        dotted_path=dotted_path,
                        replacement=None,
                        message=(
                            f"Unknown configuration key '{dotted_path}' "
                            f"was ignored on the {source} compatibility path"
                        ),
                    )
                )
                continue

            annotation = _field_annotation(field)
            nested_model = _nested_model_type(annotation)
            if nested_model is not None and isinstance(value, Mapping):
                cleaned[key] = prune(nested_model, value, dotted_path)
            elif (
                nested_model is not None
                and _is_model_sequence(annotation)
                and isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
            ):
                cleaned[key] = [
                    prune(nested_model, item, f"{dotted_path}[{index}]")
                    if isinstance(item, Mapping)
                    else copy.deepcopy(item)
                    for index, item in enumerate(value)
                ]
            else:
                cleaned[key] = copy.deepcopy(value)
        return cleaned

    return UnknownFieldPruneResult(
        config=prune(model_cls, data, ""),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "PYDANTIC_V2",
    "AssignableStrictConfigModel",
    "StrictConfigModel",
    "UnknownConfigDiagnostic",
    "UnknownConfigWarning",
    "UnknownFieldPruneResult",
    "json_safe",
    "model_dump_compat",
    "model_fields_compat",
    "model_validate_compat",
    "prune_unknown_fields",
]
