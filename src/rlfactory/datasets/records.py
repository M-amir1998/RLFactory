"""Canonical RLFactory dataset records (framework-neutral; adapters convert them)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from pydantic import Field

from rlfactory.domain.common import DomainModel
from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import Message
from rlfactory.policies.base import to_chat_messages


class DatasetMessage(DomainModel):
    role: Literal["system", "user", "assistant"]
    content: str


def to_dataset_messages(messages: Sequence[Message]) -> tuple[DatasetMessage, ...]:
    """Same conversion as rollouts use for HTTP policies (tool results become user turns)."""
    return tuple(DatasetMessage.model_validate(m) for m in to_chat_messages(messages))


class RecordLineage(DomainModel):
    task_id: str
    task_family: str
    domain: str
    trajectory_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    policy_id: str | None = None
    checkpoint: str | None = None
    verifier: str | None = None
    reward_fn: str | None = None


class SFTRecord(DomainModel):
    record_id: str
    prompt: tuple[DatasetMessage, ...] = Field(min_length=1)
    completion: tuple[DatasetMessage, ...] = Field(min_length=1)
    reward: float | None = None
    score: float | None = None
    turn: int = 0
    lineage: RecordLineage


class PreferenceRecord(DomainModel):
    record_id: str
    prompt: tuple[DatasetMessage, ...] = Field(min_length=1)
    chosen: tuple[DatasetMessage, ...] = Field(min_length=1)
    rejected: tuple[DatasetMessage, ...] = Field(min_length=1)
    chosen_reward: float | None
    rejected_reward: float | None
    chosen_passed: bool
    rejected_passed: bool
    strategy: str
    lineage: RecordLineage


class RLPromptRecord(DomainModel):
    record_id: str
    prompt: tuple[DatasetMessage, ...] = Field(min_length=1)
    task: Task
    reward: str
    verifier: str
    pass_rate: float | None = None
    lineage: RecordLineage


RecordT = TypeVar("RecordT", SFTRecord, PreferenceRecord, RLPromptRecord)


@dataclass
class BuildResult(Generic[RecordT]):
    records: list[RecordT]
    considered: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    """Exclusion reason -> count (contamination, not passed, prompt mismatch, ...)."""

    def exclude(self, reason: str, count: int = 1) -> None:
        self.excluded[reason] = self.excluded.get(reason, 0) + count

    def report(self) -> dict[str, object]:
        return {
            "records": len(self.records),
            "considered": self.considered,
            "excluded": dict(sorted(self.excluded.items())),
        }
