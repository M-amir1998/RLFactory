"""Typed YAML schemas for the standalone CLI commands (experiments use ExperimentConfig)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.datasets.builders import PreferenceBuilderConfig, RLBuilderConfig, SFTBuilderConfig
from rlfactory.domain.model import ModelReference
from rlfactory.domain.training import TrainingConfig, TrainingMethod
from rlfactory.evaluation.evaluator import EvaluationConfig
from rlfactory.evaluation.gates import DEFAULT_GATES, GateConfig
from rlfactory.policies.factory import PolicySpec
from rlfactory.rewards.registry import RewardSpec
from rlfactory.rollouts.config import RolloutConfig
from rlfactory.sandbox.factory import SandboxConfig
from rlfactory.training.mock.backend import MockTrainingSettings
from rlfactory.training.trl.backend import TRLBackendSettings
from rlfactory.verifiers.registry import VerifierSpec


class _Job(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RolloutJob(_Job):
    """``rlfactory rollout run CONFIG``."""

    name: str
    suites: tuple[str, ...] = Field(min_length=1)
    max_tasks_per_suite: int | None = Field(default=None, ge=1)
    policy: PolicySpec
    rollouts: RolloutConfig = RolloutConfig()
    sandbox: SandboxConfig = SandboxConfig()
    verifiers: dict[str, VerifierSpec] = Field(default_factory=dict)
    rewards: dict[str, RewardSpec] = Field(default_factory=dict)


class DatasetJob(_Job):
    """``rlfactory dataset build-{sft,dpo,grpo} CONFIG``."""

    name: str
    source_run: str
    """Run ID whose ``rollouts/`` store provides trajectories."""
    train_suites: tuple[str, ...] = ()
    """Needed for build-grpo (prompts come from tasks, statistics from the rollouts)."""
    eval_suites: tuple[str, ...] = ()
    """Held-out suites used for contamination control."""
    exclude_eval_families: bool = False
    sft: SFTBuilderConfig = SFTBuilderConfig()
    dpo: PreferenceBuilderConfig = PreferenceBuilderConfig()
    grpo: RLBuilderConfig = RLBuilderConfig()


class TrainJob(_Job):
    """``rlfactory train CONFIG``."""

    name: str
    backend: Literal["mock", "trl", "verl", "openrlhf"] = "mock"
    method: TrainingMethod = TrainingMethod.SFT
    model: ModelReference
    dataset_dir: str
    config: TrainingConfig = TrainingConfig()
    seed: int = 0
    mock: MockTrainingSettings | None = None
    trl: TRLBackendSettings | None = None


class EvaluateJob(_Job):
    """``rlfactory evaluate CONFIG``."""

    name: str
    label: str = "candidate"
    suites: tuple[str, ...] = Field(min_length=1)
    max_tasks_per_suite: int | None = Field(default=None, ge=1)
    policy: PolicySpec
    evaluation: EvaluationConfig = EvaluationConfig()
    sandbox: SandboxConfig = SandboxConfig()
    verifiers: dict[str, VerifierSpec] = Field(default_factory=dict)
    rewards: dict[str, RewardSpec] = Field(default_factory=dict)
    baseline: str | None = None
    """Path to a previous evaluation JSON to gate against."""
    gates: GateConfig = DEFAULT_GATES
