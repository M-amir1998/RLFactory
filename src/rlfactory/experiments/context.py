"""Per-run context shared by experiment stages."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from rlfactory.config.settings import Settings
from rlfactory.domain.common import CodeVersion
from rlfactory.domain.task import Task
from rlfactory.errors import ExperimentError
from rlfactory.experiments.config import ExperimentConfig
from rlfactory.rollouts.orchestrator import ProgressReporter
from rlfactory.rollouts.service import RolloutDependencies
from rlfactory.runtime.disk import DiskGuard
from rlfactory.sandbox.factory import create_sandbox
from rlfactory.tasks.contamination import ContaminationFilter
from rlfactory.tasks.generators.catalog import TemplateRegistry, default_template_registry
from rlfactory.tasks.io import load_tasks
from rlfactory.trajectories.parquet_store import LocalParquetTrajectoryStore

Split = Literal["train", "validation", "evaluation"]
ProgressFactory = Callable[[str], ProgressReporter | None]


class StageBlockedError(ExperimentError):
    """A stage needs a manual, explicit action (e.g. launching paid remote GPU training)."""


@dataclass
class StageOutput:
    outputs: dict[str, str] = field(default_factory=dict)
    summary: dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    config: ExperimentConfig
    settings: Settings
    code: CodeVersion
    progress: ProgressFactory = lambda _stage: None
    outputs: dict[str, dict[str, str]] = field(default_factory=dict)
    """Outputs of completed (or cached) stages, keyed by stage name."""

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    @cached_property
    def templates(self) -> TemplateRegistry:
        return default_template_registry()

    @cached_property
    def deps(self) -> RolloutDependencies:
        return RolloutDependencies.default(
            sandbox=create_sandbox(self.config.sandbox),
            python_limits=self.config.sandbox.limits,
            verifier_specs=self.config.verifiers,
            reward_specs=self.config.rewards,
        )

    @cached_property
    def store(self) -> LocalParquetTrajectoryStore:
        return LocalParquetTrajectoryStore(
            self.path("rollouts"),
            retention=self.config.rollouts.retention,
            sample_rate=self.config.rollouts.retention_sample_rate,
        )

    @cached_property
    def disk_guard(self) -> DiskGuard:
        return DiskGuard(self.config.disk_reserve_gb)

    def tasks(self, split: Split) -> list[Task]:
        path = self.path("tasks", f"{split}.jsonl")
        if not path.exists():
            raise ExperimentError(f"{path} missing; run the tasks stage first")
        return load_tasks(path)

    def task_index(self) -> dict[str, Task]:
        index: dict[str, Task] = {}
        for split in ("train", "validation", "evaluation"):
            path = self.path("tasks", f"{split}.jsonl")
            if path.exists():
                index.update({t.task_id: t for t in load_tasks(path)})
        return index

    def contamination(self) -> ContaminationFilter:
        path = self.path("tasks", "contamination.json")
        if not path.exists():
            return ContaminationFilter()
        return ContaminationFilter.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def output(self, stage: str, key: str) -> str:
        try:
            return self.outputs[stage][key]
        except KeyError as exc:
            raise ExperimentError(f"stage {stage!r} has no output {key!r}") from exc
