"""Trajectory model: messages, actions, observations, verification, rewards and lineage."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue, model_validator

from rlfactory.domain.common import CodeVersion, ComponentRef, DomainModel
from rlfactory.domain.model import GenerationConfig
from rlfactory.domain.reward import RewardResult
from rlfactory.domain.task import TaskDomain, ToolCall
from rlfactory.domain.verification import VerificationResult


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(DomainModel):
    role: Role
    content: str
    name: str | None = None


class TokenUsage(DomainModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    estimated: bool = False
    """True when counts are approximations (e.g. whitespace tokens from the mock policy)."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelAction(DomainModel):
    """Raw model output for one turn. Interpretation (parsing) belongs to the environment."""

    text: str
    finish_reason: str | None = None
    usage: TokenUsage = TokenUsage()
    latency_s: float = Field(default=0.0, ge=0.0)


class ExecutionStatus(StrEnum):
    OK = "ok"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    OUTPUT_LIMIT = "output_limit"
    FILE_LIMIT = "file_limit"
    SANDBOX_ERROR = "sandbox_error"


class UnitTestOutcome(DomainModel):
    index: int
    passed: bool
    error: str | None = None


class ExecutionReport(DomainModel):
    """Summary of executing generated code against hidden tests (stored in trajectories)."""

    status: ExecutionStatus
    tests_passed: int = Field(ge=0)
    tests_total: int = Field(ge=0)
    outcomes: tuple[UnitTestOutcome, ...] = ()
    error_type: str | None = None
    report_found: bool = True
    stderr_tail: str = ""
    duration_s: float = Field(default=0.0, ge=0.0)
    sandbox: str = "unknown"

    @model_validator(mode="after")
    def _counts_consistent(self) -> ExecutionReport:
        if self.tests_passed > self.tests_total:
            raise ValueError("tests_passed cannot exceed tests_total")
        return self


class ToolCallRecord(DomainModel):
    call: ToolCall | None
    result: JsonValue = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.call is not None and self.error is None


class Observation(DomainModel):
    """What the environment returns. Only ``content`` is shown to the policy.

    Typed side-channels (``parsed_answer``, ``execution``, ``tool``) carry the environment's
    interpretation for verifiers and analysis, e.g. hidden unit-test results.
    """

    content: str
    role: Role = Role.USER
    parsed_answer: str | None = None
    parse_error: str | None = None
    execution: ExecutionReport | None = None
    tool: ToolCallRecord | None = None


class StepResult(DomainModel):
    observation: Observation
    done: bool
    truncated: bool = False


class FailureOrigin(StrEnum):
    MODEL = "model"
    INFRASTRUCTURE = "infrastructure"


class FailureKind(StrEnum):
    POLICY_TIMEOUT = "policy_timeout"
    POLICY_ERROR = "policy_error"
    SANDBOX_TIMEOUT = "sandbox_timeout"
    SANDBOX_ERROR = "sandbox_error"
    VERIFIER_ERROR = "verifier_error"
    INVALID_OUTPUT = "invalid_output"
    INFRASTRUCTURE_ERROR = "infrastructure_error"

    @property
    def origin(self) -> FailureOrigin:
        """Model failures are legitimate (negative) training signal; infrastructure ones are not."""
        if self in (FailureKind.INVALID_OUTPUT, FailureKind.SANDBOX_TIMEOUT):
            return FailureOrigin.MODEL
        return FailureOrigin.INFRASTRUCTURE


class FailureRecord(DomainModel):
    kind: FailureKind
    message: str = ""
    attempts: int = Field(default=1, ge=1)
    retryable: bool = False
    """Whether the orchestrator may retry (e.g. HTTP 503 yes, HTTP 400 no)."""

    @property
    def origin(self) -> FailureOrigin:
        return self.kind.origin


class TrajectoryStatus(StrEnum):
    PENDING = "pending"
    """Episode finished but not yet verified/rewarded. Transient: never persisted."""
    COMPLETED = "completed"
    """The episode ran to completion and was verified (the answer may still be wrong)."""
    FAILED = "failed"
    """Infrastructure prevented a valid episode; excluded from training datasets."""


class Lineage(DomainModel):
    run_id: str
    task_version: str
    task_fingerprint: str
    environment: ComponentRef
    verifier: ComponentRef
    reward: ComponentRef
    prompt_template: ComponentRef
    policy_id: str
    model: str | None = None
    checkpoint: str | None = None
    generation: GenerationConfig
    seed: int
    code: CodeVersion


class TrajectoryMetadata(DomainModel):
    domain: TaskDomain
    task_family: str
    sample_index: int = Field(ge=0)
    group_id: str
    """All samples of one task under one rollout configuration (GRPO group / preference pool)."""
    attempts: int = Field(default=1, ge=1)
    started_at: datetime
    finished_at: datetime
    wall_time_s: float = Field(ge=0.0)
    truncated: bool = False


class Trajectory(DomainModel):
    trajectory_id: str
    task_id: str
    policy_id: str
    status: TrajectoryStatus
    messages: tuple[Message, ...]
    actions: tuple[ModelAction, ...]
    observations: tuple[Observation, ...]
    """``observations[0]`` comes from ``reset``; ``observations[i + 1]`` follows ``actions[i]``."""
    verification: VerificationResult | None = None
    rewards: RewardResult | None = None
    failure: FailureRecord | None = None
    lineage: Lineage
    metadata: TrajectoryMetadata

    @model_validator(mode="after")
    def _consistent(self) -> Trajectory:
        if self.observations and len(self.observations) not in (
            len(self.actions),
            len(self.actions) + 1,
        ):
            raise ValueError("observations must align with actions")
        if self.status is TrajectoryStatus.FAILED and self.failure is None:
            raise ValueError("failed trajectories must carry a failure record")
        if self.status is TrajectoryStatus.COMPLETED and self.verification is None:
            raise ValueError("completed trajectories must be verified")
        return self

    def finalized(self, verification: VerificationResult, rewards: RewardResult) -> Trajectory:
        """Return the COMPLETED version of a PENDING trajectory (re-validated)."""
        if self.status is not TrajectoryStatus.PENDING:
            raise ValueError("only pending trajectories can be finalized")
        fields = dict(self)
        fields.update(status=TrajectoryStatus.COMPLETED, verification=verification, rewards=rewards)
        return Trajectory(**fields)

    @property
    def final_action(self) -> ModelAction | None:
        return self.actions[-1] if self.actions else None

    @property
    def final_observation(self) -> Observation | None:
        return self.observations[-1] if len(self.observations) > 1 else None

    @property
    def completion_text(self) -> str:
        final = self.final_action
        return final.text if final is not None else ""

    @property
    def completion_tokens(self) -> int:
        return sum(action.usage.completion_tokens for action in self.actions)

    @property
    def passed(self) -> bool:
        return self.verification is not None and self.verification.passed

    @property
    def reward_total(self) -> float | None:
        return self.rewards.total if self.rewards is not None else None

    @property
    def is_trainable(self) -> bool:
        """Completed, verified and rewarded trajectories are eligible for dataset building."""
        return (
            self.status is TrajectoryStatus.COMPLETED
            and self.verification is not None
            and self.rewards is not None
        )

    def tool_records(self) -> list[ToolCallRecord]:
        return [obs.tool for obs in self.observations if obs.tool is not None]

    def prompt_messages(self) -> tuple[Message, ...]:
        """Messages up to (excluding) the first assistant turn."""
        prompt: list[Message] = []
        for message in self.messages:
            if message.role is Role.ASSISTANT:
                break
            prompt.append(message)
        return tuple(prompt)
