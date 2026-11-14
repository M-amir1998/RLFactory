"""Shared building blocks for domain records: base model, component references, hashing."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue

__all__ = [
    "CodeVersion",
    "ComponentRef",
    "DomainModel",
    "JsonValue",
    "canonical_json",
    "stable_hash",
    "stable_seed",
    "to_jsonable",
]


class DomainModel(BaseModel):
    """Immutable, strictly validated base class for every domain record."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ComponentRef(DomainModel):
    """Name and version of a versioned component (environment, verifier, reward, template)."""

    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


class CodeVersion(DomainModel):
    """Code identity recorded in every lineage record."""

    rlfactory_version: str
    git_sha: str | None = None
    git_dirty: bool | None = None


def to_jsonable(value: Any) -> Any:
    """Convert pydantic models (recursively) into plain JSON-compatible structures."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floats cannot be hashed canonically")
    return value


def canonical_json(value: Any) -> str:
    """Serialise ``value`` deterministically (sorted keys, no whitespace, UTF-8 preserved)."""
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_hash(*parts: Any, length: int = 16) -> str:
    """Return a deterministic hex digest of ``parts``.

    Unlike :func:`hash`, the result does not depend on ``PYTHONHASHSEED`` and is stable across
    processes, machines and Python versions, which makes it suitable for idempotent identifiers.
    """
    if not 4 <= length <= 64:
        raise ValueError("length must be between 4 and 64 hex characters")
    digest = hashlib.sha256()
    for part in parts:
        digest.update(canonical_json(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:length]


def stable_seed(*parts: Any) -> int:
    """Derive a non-negative 63-bit integer seed from ``parts``."""
    return int(stable_hash(*parts, length=16), 16) & ((1 << 63) - 1)
