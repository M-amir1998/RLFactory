"""Experiment configuration and the stage DAG."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rlfactory.datasets.builders import PreferenceBuilderConfig, RLBuilderConfig, SFTBuilderConfig
from rlfactory.domain.model import ModelReference
from rlfactory.domain.training import TrainingConfig, TrainingMethod
from rlfactory.evaluation.evaluator import EvaluationConfig
from rlfactory.evaluation.gates import DEFAULT_GATES, GateConfig
from rlfactory.flywheel.loop import FlywheelConfig
from rlfactory.policies.factory import PolicySpec
from rlfactory.rewards.registry import RewardSpec
from rlfactory.rollouts.config import RolloutConfig
from rlfactory.sandbox.factory import SandboxConfig
from rlfactory.tasks.splits import SplitConfig
from rlfactory.training.mock.backend import MockTrainingSettings
from rlfactory.training.trl.backend import TRLBackendSettings
from rlfactory.verifiers.registry import VerifierSpec

StageName = Literal[
    "tasks",
    "rollouts",
    "build_sft",
    "build_dpo",
    "build_grpo",
    "train",
    "evaluate_base",
    "evaluate_candidate",
    "gate",
    "analysis",
    "flywheel",
    "report",
]

STAGE_ORDER: tuple[StageName, ...] = (
    "tasks",
    "rollouts",
    "build_sft",
    "build_dpo",
    "build_grpo",
    "train",
    "evaluate_base",
    "evaluate_candidate",
    "gate",
    "analysis",
    "flywheel",
    "report",
)

DATASET_STAGE: dict[TrainingMethod, StageName] = {
    TrainingMethod.SFT: "build_sft",
    TrainingMethod.DPO: "build_dpo",
    TrainingMethod.GRPO: "build_grpo",
    TrainingMethod.REWARD_MODEL: "build_dpo",
}


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TasksSection(_Section):
    train_suites: tuple[str, ...] = Field(min_length=1)
    eval_suites: tuple[str, ...] = Field(min_length=1)
    split: SplitConfig | None = None
    """Optional split of the training suites; its validation part is the flywheel's dev set."""
    exclude_eval_families: bool = False
    """Also drop every training task from a template family used in evaluation."""
    validate_tasks: bool = True
    max_tasks_per_suite: int | None = Field(default=None, ge=1)


class DatasetsSection(_Section):
    sft: SFTBuilderConfig = SFTBuilderConfig()
    dpo: PreferenceBuilderConfig = PreferenceBuilderConfig()
    grpo: RLBuilderConfig = RLBuilderConfig()


class TrainingSection(_Section):
    backend: Literal["mock", "trl", "verl", "openrlhf"] = "mock"
    method: TrainingMethod = TrainingMethod.SFT
    model: ModelReference
    config: TrainingConfig = TrainingConfig()
    mock: MockTrainingSettings | None = None
    trl: TRLBackendSettings | None = None
    remote_host: str | None = None
    """Name of ``configs/remote/<name>.yaml``; remote launches never happen implicitly."""

    @model_validator(mode="after")
    def _method_matches(self) -> TrainingSection:
        if self.config.method_params.method != self.method.value:
            raise ValueError(
                f"training.method={self.method.value} but training.config.method_params is "
                f"{self.config.method_params.method}"
            )
        return self


class EvaluationSection(EvaluationConfig):
    gates: GateConfig = DEFAULT_GATES
    candidate_policy: PolicySpec | None = None
    """How to serve a real (non-mock) checkpoint for evaluation, e.g. a vLLM endpoint."""


class ExperimentConfig(_Section):
    name: str
    description: str = ""
    seed: int = 0
    tasks: TasksSection
    policy: PolicySpec
    sandbox: SandboxConfig = SandboxConfig()
    verifiers: dict[str, VerifierSpec] = Field(default_factory=dict)
    rewards: dict[str, RewardSpec] = Field(default_factory=dict)
    rollouts: RolloutConfig = RolloutConfig()
    datasets: DatasetsSection = DatasetsSection()
    training: TrainingSection | None = None
    evaluation: EvaluationSection = EvaluationSection()
    flywheel: FlywheelConfig | None = None
    stages: tuple[StageName, ...] = STAGE_ORDER
    disk_reserve_gb: float = Field(default=2.0, ge=0.0)

    @field_validator("name")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9\-]{1,62}", value):
            raise ValueError("experiment name must be a lowercase slug (a-z, 0-9, '-')")
        return value

    @model_validator(mode="after")
    def _stages_consistent(self) -> ExperimentConfig:
        enabled = set(self.stages)
        needs_training = {"train", "evaluate_candidate", "gate"} & enabled
        if needs_training and self.training is None:
            raise ValueError(f"stages {sorted(needs_training)} need a training section")
        if "flywheel" in enabled and self.flywheel is None:
            raise ValueError("the flywheel stage needs a flywheel section")
        for stage in self.stages:
            for dependency in stage_dependencies(stage, self):
                if dependency not in enabled:
                    raise ValueError(
                        f"stage {stage!r} depends on {dependency!r}, which is not enabled"
                    )
        return self


def stage_dependencies(stage: StageName, config: ExperimentConfig) -> tuple[StageName, ...]:
    if stage == "train":
        method = config.training.method if config.training else TrainingMethod.SFT
        return (DATASET_STAGE[method],)
    dependencies: dict[str, tuple[StageName, ...]] = {
        "tasks": (),
        "rollouts": ("tasks",),
        "build_sft": ("rollouts",),
        "build_dpo": ("rollouts",),
        "build_grpo": ("tasks", "rollouts"),
        "evaluate_base": ("tasks",),
        "evaluate_candidate": ("tasks", "train"),
        "gate": ("evaluate_base", "evaluate_candidate"),
        "analysis": ("rollouts",),
        "flywheel": ("tasks",),
        "report": (),
    }
    return dependencies[stage]
