"""Trajectory store protocol, query model, flat summaries and retention policies."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rlfactory.domain.common import stable_hash
from rlfactory.domain.trajectory import Trajectory, TrajectoryStatus


class RetentionPolicy(StrEnum):
    """Which full payloads to keep. Flat summaries are ALWAYS kept for every trajectory."""

    ALL = "all"
    SUCCESSFUL_ONLY = "successful_only"
    FAILED_ONLY = "failed_only"
    SAMPLED = "sampled"


def should_retain(trajectory: Trajectory, policy: RetentionPolicy, sample_rate: float) -> bool:
    if policy is RetentionPolicy.ALL:
        return True
    if policy is RetentionPolicy.SUCCESSFUL_ONLY:
        return trajectory.passed
    if policy is RetentionPolicy.FAILED_ONLY:
        return not trajectory.passed
    bucket = int(stable_hash(trajectory.trajectory_id, length=8), 16) / float(16**8)
    return bucket < sample_rate


class TrajectoryQuery(BaseModel):
    """Filters combined with AND. ``None`` means "do not filter"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str | None = None
    task_id: str | None = None
    task_ids: tuple[str, ...] | None = None
    domain: str | None = None
    environment: str | None = None
    policy_id: str | None = None
    checkpoint: str | None = None
    verifier: str | None = None
    passed: bool | None = None
    status: TrajectoryStatus | None = None
    failure_kind: str | None = None
    min_reward: float | None = None
    max_reward: float | None = None
    seed: int | None = None
    limit: int | None = Field(default=None, ge=1)


class TrajectorySummary(BaseModel):
    """One flat row per trajectory (what the Parquet index stores besides the payload)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trajectory_id: str
    run_id: str
    task_id: str
    task_version: str
    task_family: str
    domain: str
    environment: str
    environment_version: str
    verifier: str
    verifier_version: str
    reward_fn: str
    reward_version: str
    policy_id: str
    model: str | None
    checkpoint: str | None
    prompt_template: str
    seed: int
    sample_index: int
    group_id: str
    status: str
    failure_kind: str | None
    failure_origin: str | None
    attempts: int
    passed: bool | None
    score: float | None
    reason_code: str | None
    reward: float | None
    reward_correctness: float | None
    reward_format: float | None
    reward_length: float | None
    reward_tool_cost: float | None
    reward_safety: float | None
    num_turns: int
    prompt_tokens: int
    completion_tokens: int
    completion_chars: int
    tokens_estimated: bool
    policy_latency_s: float
    wall_time_s: float
    truncated: bool
    generation_fingerprint: str
    git_sha: str | None
    rlfactory_version: str
    started_at: datetime
    finished_at: datetime
    payload_retained: bool

    @field_validator("started_at", "finished_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def summarize(trajectory: Trajectory, *, payload_retained: bool) -> TrajectorySummary:
    lineage = trajectory.lineage
    verification = trajectory.verification
    rewards = trajectory.rewards
    failure = trajectory.failure
    components = rewards.components if rewards is not None else None
    return TrajectorySummary(
        trajectory_id=trajectory.trajectory_id,
        run_id=lineage.run_id,
        task_id=trajectory.task_id,
        task_version=lineage.task_version,
        task_family=trajectory.metadata.task_family,
        domain=trajectory.metadata.domain.value,
        environment=lineage.environment.name,
        environment_version=lineage.environment.version,
        verifier=lineage.verifier.name,
        verifier_version=lineage.verifier.version,
        reward_fn=lineage.reward.name,
        reward_version=lineage.reward.version,
        policy_id=trajectory.policy_id,
        model=lineage.model,
        checkpoint=lineage.checkpoint,
        prompt_template=str(lineage.prompt_template),
        seed=lineage.seed,
        sample_index=trajectory.metadata.sample_index,
        group_id=trajectory.metadata.group_id,
        status=trajectory.status.value,
        failure_kind=failure.kind.value if failure else None,
        failure_origin=failure.origin.value if failure else None,
        attempts=trajectory.metadata.attempts,
        passed=verification.passed if verification else None,
        score=verification.score if verification else None,
        reason_code=verification.reason_code if verification else None,
        reward=rewards.total if rewards else None,
        reward_correctness=components.correctness if components else None,
        reward_format=components.format if components else None,
        reward_length=components.length if components else None,
        reward_tool_cost=components.tool_cost if components else None,
        reward_safety=components.safety if components else None,
        num_turns=len(trajectory.actions),
        prompt_tokens=sum(a.usage.prompt_tokens for a in trajectory.actions),
        completion_tokens=trajectory.completion_tokens,
        completion_chars=sum(len(a.text) for a in trajectory.actions),
        tokens_estimated=any(a.usage.estimated for a in trajectory.actions),
        policy_latency_s=sum(a.latency_s for a in trajectory.actions),
        wall_time_s=trajectory.metadata.wall_time_s,
        truncated=trajectory.metadata.truncated,
        generation_fingerprint=lineage.generation.fingerprint(),
        git_sha=lineage.code.git_sha,
        rlfactory_version=lineage.code.rlfactory_version,
        started_at=trajectory.metadata.started_at,
        finished_at=trajectory.metadata.finished_at,
        payload_retained=payload_retained,
    )


class TrajectoryStore(Protocol):
    def write(self, trajectories: Sequence[Trajectory]) -> None:
        """Buffer (and eventually persist) terminal trajectories. Rejects PENDING ones."""
        ...

    def flush(self) -> None: ...

    def completed_ids(self, run_id: str | None = None) -> set[str]:
        """IDs whose latest record is COMPLETED (used to resume without regenerating work)."""
        ...

    def query(self, query: TrajectoryQuery) -> list[TrajectorySummary]: ...

    def load(self, query: TrajectoryQuery) -> list[Trajectory]:
        """Full trajectories whose payload was retained."""
        ...

    def aggregate(self, group_by: Sequence[str], query: TrajectoryQuery) -> list[dict[str, Any]]:
        """Counts, pass rate and reward statistics per group (latest record per trajectory)."""
        ...
