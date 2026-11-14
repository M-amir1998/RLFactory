"""Deterministic train / validation / evaluation splits.

Assignment hashes a group key with a salt, so it is stable across runs, machines and task order.
``group_by="family"`` keeps whole template families together (held-out-template evaluation);
``group_by="task"`` splits instances (held-out instances of seen templates).
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlfactory.domain.common import stable_hash
from rlfactory.domain.task import Task


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    EVALUATION = "evaluation"


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    train: float = Field(default=0.8, ge=0.0, le=1.0)
    validation: float = Field(default=0.1, ge=0.0, le=1.0)
    evaluation: float = Field(default=0.1, ge=0.0, le=1.0)
    group_by: Literal["task", "family"] = "task"
    salt: str = "rlfactory-split-v1"

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitConfig:
        if abs(self.train + self.validation + self.evaluation - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


def _unit_interval(salt: str, key: str) -> float:
    return int(stable_hash(salt, key, length=15), 16) / float(16**15)


def assign_split(task: Task, config: SplitConfig) -> Split:
    key = task.family if config.group_by == "family" else task.task_id
    u = _unit_interval(config.salt, key)
    if u < config.train:
        return Split.TRAIN
    if u < config.train + config.validation:
        return Split.VALIDATION
    return Split.EVALUATION


def split_tasks(tasks: Iterable[Task], config: SplitConfig) -> dict[Split, list[Task]]:
    result: dict[Split, list[Task]] = {split: [] for split in Split}
    for task in tasks:
        result[assign_split(task, config)].append(task)
    return result
