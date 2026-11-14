"""Framework-agnostic training run description, handles and status."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from rlfactory.domain.common import DomainModel, stable_hash
from rlfactory.domain.model import ModelReference


class TrainingMethod(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    REWARD_MODEL = "reward_model"


class LaunchMode(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class DatasetReference(DomainModel):
    name: str
    path: str
    schema_name: str
    """Canonical RLFactory schema, e.g. ``rlfactory.sft.v1`` or ``trl.grpo.prompt_only``."""
    num_records: int = Field(ge=0)
    fingerprint: str


class LoRASettings(DomainModel):
    enabled: bool = True
    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: str | tuple[str, ...] = "all-linear"
    quantization: Literal["none", "4bit"] = "none"
    """``4bit`` = QLoRA (NF4 + double quantisation); requires bitsandbytes on a CUDA host."""


class RewardFunctionSpec(DomainModel):
    """One reward function exposed to an RL trainer, backed by RLFactory verifiers/rewards."""

    component: Literal["correctness", "format", "length", "tool_cost", "total"]
    weight: float = 1.0


class SFTParams(DomainModel):
    method: Literal["sft"] = "sft"
    max_length: int = Field(default=1024, ge=16)
    packing: bool = False
    completion_only_loss: bool = True


class DPOParams(DomainModel):
    method: Literal["dpo"] = "dpo"
    beta: float = Field(default=0.1, gt=0.0)
    loss_type: tuple[str, ...] = ("sigmoid",)
    max_length: int = Field(default=1024, ge=16)
    precompute_ref_log_probs: bool = False


class GRPOParams(DomainModel):
    method: Literal["grpo"] = "grpo"
    num_generations: int = Field(default=8, ge=2)
    max_completion_length: int = Field(default=512, ge=1)
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    beta: float = Field(default=0.0, ge=0.0)
    loss_type: str = "dapo"
    scale_rewards: Literal["group", "batch", "none"] = "group"
    num_iterations: int = Field(default=1, ge=1)
    epsilon: float = Field(default=0.2, gt=0.0)
    mask_truncated_completions: bool = False
    use_vllm: bool = False
    vllm_mode: Literal["colocate", "server"] = "colocate"
    vllm_server_base_url: str | None = None
    vllm_gpu_memory_utilization: float = Field(default=0.3, gt=0.0, lt=1.0)
    log_completions: bool = True
    reward_functions: tuple[RewardFunctionSpec, ...] = (
        RewardFunctionSpec(component="correctness"),
    )


class RewardModelParams(DomainModel):
    method: Literal["reward_model"] = "reward_model"
    max_length: int = Field(default=1024, ge=16)
    center_rewards_coefficient: float | None = None


MethodParams = Annotated[
    SFTParams | DPOParams | GRPOParams | RewardModelParams, Field(discriminator="method")
]


class TrainingConfig(DomainModel):
    learning_rate: float = Field(default=1e-5, gt=0.0)
    num_train_epochs: float = Field(default=1.0, gt=0.0)
    max_steps: int = -1
    per_device_train_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    gradient_checkpointing: bool = True
    bf16: bool = True
    warmup_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    lr_scheduler_type: str = "cosine"
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int | None = 2
    lora: LoRASettings = LoRASettings()
    method_params: MethodParams = Field(default_factory=SFTParams)
    backend_overrides: dict[str, JsonValue] = Field(default_factory=dict)
    """Escape hatch passed verbatim to the framework config; recorded in lineage."""


class TrainingRun(DomainModel):
    run_id: str
    method: TrainingMethod
    model: ModelReference
    dataset: DatasetReference
    eval_dataset: DatasetReference | None = None
    config: TrainingConfig = TrainingConfig()
    seed: int = 0
    launch_mode: LaunchMode = LaunchMode.LOCAL

    @model_validator(mode="after")
    def _method_consistent(self) -> TrainingRun:
        if self.config.method_params.method != self.method.value:
            raise ValueError(
                f"method {self.method.value!r} does not match method_params "
                f"{self.config.method_params.method!r}"
            )
        return self

    def fingerprint(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude={"run_id"}), length=16)


class PreparedTrainingRun(DomainModel):
    run: TrainingRun
    backend: str
    workdir: str
    command: tuple[str, ...]
    """Launch command, relative to ``workdir``; never contains secrets."""
    files: tuple[str, ...] = ()
    environment: dict[str, str] = Field(default_factory=dict)
    framework_version: str | None = None
    notes: tuple[str, ...] = ()


class TrainingState(StrEnum):
    PREPARED = "prepared"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (TrainingState.SUCCEEDED, TrainingState.FAILED, TrainingState.CANCELLED)


class TrainingHandle(DomainModel):
    handle_id: str
    backend: str
    run_id: str
    workdir: str
    launch_mode: LaunchMode
    remote_host: str | None = None
    started_at: datetime


class TrainingMetricsPoint(DomainModel):
    step: int = Field(ge=0)
    loss: float | None = None
    learning_rate: float | None = None
    tokens_seen: int | None = None
    wall_time_s: float = 0.0
    peak_vram_gb: float | None = None
    extra: dict[str, float] = Field(default_factory=dict)


class CheckpointEvent(DomainModel):
    step: int = Field(ge=0)
    path: str
    created_at: datetime


class TrainingStatus(DomainModel):
    handle_id: str
    state: TrainingState
    step: int = 0
    total_steps: int | None = None
    latest: TrainingMetricsPoint | None = None
    checkpoints: tuple[CheckpointEvent, ...] = ()
    error: str | None = None
    simulated: bool = False


class TrainingSummary(DomainModel):
    """Written to ``training_summary.json`` after a run finishes (or fails)."""

    run_id: str
    backend: str
    method: TrainingMethod
    state: TrainingState
    steps: int
    final_loss: float | None
    wall_time_s: float
    tokens_seen: int | None
    peak_vram_gb: float | None
    checkpoints: tuple[CheckpointEvent, ...]
    framework_version: str | None
    dataset: DatasetReference
    model: ModelReference
    simulated: bool
    resumed_from_step: int | None = None
    error: str | None = None
