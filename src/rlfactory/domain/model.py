"""Model, checkpoint and generation-configuration references."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue

from rlfactory.domain.common import DomainModel, stable_hash


class ModelReference(DomainModel):
    """A base model reference. Revisions should be pinned to a commit SHA for real runs."""

    name: str
    revision: str | None = None
    trust_remote_code: bool = False
    """Remote custom code is opt-in only (see SECURITY.md)."""
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"

    @property
    def is_pinned(self) -> bool:
        return self.revision is not None and len(self.revision) >= 7

    def describe(self) -> str:
        return f"{self.name}@{self.revision}" if self.revision else f"{self.name} (unpinned)"


class GenerationConfig(DomainModel):
    temperature: float = Field(default=1.0, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=512, ge=1)
    stop: tuple[str, ...] = ()
    timeout_s: float = Field(default=120.0, gt=0.0)

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    def fingerprint(self) -> str:
        return stable_hash(self.model_dump(mode="json"), length=12)


class CheckpointReference(DomainModel):
    """A trained (or base) policy checkpoint; weights are never stored in Git."""

    checkpoint_id: str
    base_model: ModelReference
    adapter_uri: str | None = None
    method: str | None = None
    training_run_id: str | None = None
    step: int | None = None
    simulated: bool = False
    """True for checkpoints emitted by the mock training backend (no weights exist)."""


class CheckpointMetadata(DomainModel):
    """Small metadata record written next to every checkpoint."""

    checkpoint: CheckpointReference
    training_method: str
    dataset: str
    dataset_fingerprint: str
    git_sha: str | None
    training_config: dict[str, JsonValue]
    framework: str
    framework_version: str | None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime
