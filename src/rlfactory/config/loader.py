"""YAML configuration loading with strict validation and helpful errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from rlfactory.errors import ConfigError

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping using ``safe_load`` (no arbitrary object construction)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    data = load_yaml(path)
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid {model_type.__name__} in {path}:\n{exc}") from exc


def dump_yaml(data: BaseModel | dict[str, Any], path: Path) -> None:
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )


def write_json(data: BaseModel | dict[str, Any] | list[Any], path: Path) -> None:
    """Write JSON atomically (temp file + replace) so readers never observe partial files.

    Line endings are always LF, so artefacts are byte-identical across platforms."""
    import json

    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    tmp.replace(path)


def read_json_model(path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ConfigError(f"cannot load {model_type.__name__} from {path}: {exc}") from exc
