"""Rollout worker: Task -> Policy -> Environment -> Verifier -> Reward -> Trajectory.

Failure taxonomy (see ``FailureKind``):

* infrastructure failures (policy timeout/error, sandbox error, verifier error, unexpected
  exceptions) produce a FAILED trajectory that carries partial messages for debugging and is
  excluded from datasets; the orchestrator decides whether to retry;
* model failures (unparseable output, generated code timing out) produce a COMPLETED trajectory
  that is verified and rewarded like any other answer, with the failure recorded for analysis.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime

from rlfactory.domain.common import CodeVersion, ComponentRef, stable_seed
from rlfactory.domain.trajectory import (
    ExecutionStatus,
    FailureKind,
    FailureRecord,
    Lineage,
    ModelAction,
    Trajectory,
    TrajectoryMetadata,
    TrajectoryStatus,
)
from rlfactory.domain.verification import VerificationResult
from rlfactory.environments.base import Environment, EnvironmentContext
from rlfactory.environments.registry import EnvironmentRegistry
from rlfactory.errors import PolicyError, PolicyTimeoutError, SandboxError
from rlfactory.policies.base import GenerationRequest, Policy
from rlfactory.rewards.registry import RewardRegistry
from rlfactory.rollouts.config import RolloutConfig
from rlfactory.rollouts.episode import MAX_TURNS, EpisodeRecorder
from rlfactory.rollouts.jobs import RolloutJob
from rlfactory.tasks.prompts import PromptBuilder
from rlfactory.telemetry.metrics import RolloutMetrics
from rlfactory.verifiers.registry import VerifierRegistry


class EpisodeFailureError(Exception):
    def __init__(self, kind: FailureKind, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass(frozen=True)
class RolloutContext:
    run_id: str
    policy: Policy
    environments: EnvironmentRegistry
    environment_context: EnvironmentContext
    verifiers: VerifierRegistry
    rewards: RewardRegistry
    prompts: PromptBuilder
    config: RolloutConfig
    code: CodeVersion
    metrics: RolloutMetrics


def classify_model_failure(trajectory: Trajectory) -> FailureKind | None:
    """Observed model-side failure of a finished episode (not a root-cause diagnosis)."""
    final = trajectory.final_observation
    if final is None:
        return FailureKind.INVALID_OUTPUT
    if final.execution is not None and final.execution.status is ExecutionStatus.TIMEOUT:
        return FailureKind.SANDBOX_TIMEOUT
    if final.parse_error is not None:
        return FailureKind.INVALID_OUTPUT
    if trajectory.metadata.truncated and final.parsed_answer is None:
        return FailureKind.INVALID_OUTPUT
    return None


class RolloutWorker:
    def __init__(self, context: RolloutContext) -> None:
        self.ctx = context

    def lineage(self, job: RolloutJob) -> Lineage:
        ctx = self.ctx
        task = job.task
        verifier = ctx.verifiers.get(ctx.config.verifier_overrides.get(task.domain, task.verifier))
        reward = ctx.rewards.get(ctx.config.reward)
        return Lineage(
            run_id=ctx.run_id,
            task_version=task.version,
            task_fingerprint=task.fingerprint(),
            environment=ctx.environments.spec(task.environment).ref,
            verifier=ComponentRef(name=verifier.name, version=verifier.version),
            reward=ComponentRef(name=reward.name, version=reward.version),
            prompt_template=ctx.prompts.template,
            policy_id=ctx.policy.policy_id,
            model=ctx.policy.model,
            checkpoint=ctx.policy.checkpoint,
            generation=ctx.config.generation,
            seed=job.seed,
            code=ctx.code,
        )

    async def _generate(
        self, job: RolloutJob, recorder: EpisodeRecorder, attempt: int
    ) -> ModelAction:
        ctx = self.ctx
        request = GenerationRequest(
            request_id=f"{job.trajectory_id}-t{recorder.turn}-a{attempt}",
            messages=tuple(recorder.messages),
            config=ctx.config.generation,
            seed=stable_seed(job.seed, recorder.turn) % (2**31 - 1),
            task_id=job.task.task_id,
            sample_index=job.sample_index,
            turn=recorder.turn,
            attempt=attempt,
        )
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                ctx.policy.generate(request), timeout=ctx.config.generation.timeout_s
            )
        except (TimeoutError, PolicyTimeoutError) as exc:
            raise EpisodeFailureError(
                FailureKind.POLICY_TIMEOUT, str(exc) or "timeout", retryable=True
            ) from exc
        except PolicyError as exc:
            raise EpisodeFailureError(
                FailureKind.POLICY_ERROR, str(exc), retryable=exc.retryable
            ) from exc
        finally:
            ctx.metrics.policy_latency.observe(time.perf_counter() - started)
        return ModelAction(
            text=result.text,
            finish_reason=result.finish_reason,
            usage=result.usage,
            latency_s=result.latency_s,
        )

    async def _step(
        self, environment: Environment, action: ModelAction, recorder: EpisodeRecorder
    ) -> None:
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(environment.step, action)
        except SandboxError as exc:
            self.ctx.metrics.sandbox_failures.inc()
            raise EpisodeFailureError(FailureKind.SANDBOX_ERROR, str(exc), retryable=True) from exc
        finally:
            self.ctx.metrics.environment_latency.observe(time.perf_counter() - started)
        recorder.record(action, result)

    def _verify(self, job: RolloutJob, pending: Trajectory) -> VerificationResult:
        ctx = self.ctx
        name = ctx.config.verifier_overrides.get(job.task.domain, job.task.verifier)
        started = time.perf_counter()
        try:
            return ctx.verifiers.get(name).verify(job.task, pending)
        except Exception as exc:
            raise EpisodeFailureError(
                FailureKind.VERIFIER_ERROR, f"{type(exc).__name__}: {exc}", retryable=False
            ) from exc
        finally:
            ctx.metrics.verifier_latency.observe(time.perf_counter() - started)

    def failed_trajectory(
        self,
        job: RolloutJob,
        failure: EpisodeFailureError,
        *,
        attempt: int,
        started_at: datetime | None = None,
        recorder: EpisodeRecorder | None = None,
        lineage: Lineage | None = None,
    ) -> Trajectory:
        """FAILED trajectory (with partial messages when available) for an infra failure."""
        lineage = lineage or self.lineage(job)
        started_at = started_at or datetime.now(UTC)
        record = FailureRecord(
            kind=failure.kind,
            message=str(failure)[:500],
            attempts=attempt,
            retryable=failure.retryable,
        )
        if recorder is not None:
            return recorder.build(
                trajectory_id=job.trajectory_id,
                policy_id=self.ctx.policy.policy_id,
                lineage=lineage,
                sample_index=job.sample_index,
                group_id=job.group_id,
                started_at=started_at,
                attempts=attempt,
                status=TrajectoryStatus.FAILED,
                failure=record,
            )
        finished = datetime.now(UTC)
        return Trajectory(
            trajectory_id=job.trajectory_id,
            task_id=job.task.task_id,
            policy_id=self.ctx.policy.policy_id,
            status=TrajectoryStatus.FAILED,
            messages=(),
            actions=(),
            observations=(),
            failure=record,
            lineage=lineage,
            metadata=TrajectoryMetadata(
                domain=job.task.domain,
                task_family=job.task.family,
                sample_index=job.sample_index,
                group_id=job.group_id,
                attempts=attempt,
                started_at=started_at,
                finished_at=finished,
                wall_time_s=(finished - started_at).total_seconds(),
            ),
        )

    async def run(self, job: RolloutJob, attempt: int = 1) -> Trajectory:
        ctx = self.ctx
        started_at = datetime.now(UTC)
        lineage = self.lineage(job)
        recorder: EpisodeRecorder | None = None
        environment: Environment | None = None
        try:
            try:
                environment = ctx.environments.create(job.task.environment, ctx.environment_context)
                first = environment.reset(job.task)
            except Exception as exc:
                raise EpisodeFailureError(
                    FailureKind.INFRASTRUCTURE_ERROR,
                    f"environment reset failed: {exc}",
                    retryable=False,
                ) from exc
            recorder = EpisodeRecorder(
                job.task, ctx.prompts.initial_messages(job.task, first), first
            )
            while not recorder.done and recorder.turn < MAX_TURNS:
                action = await self._generate(job, recorder, attempt)
                await self._step(environment, action, recorder)
            pending = recorder.build(
                trajectory_id=job.trajectory_id,
                policy_id=ctx.policy.policy_id,
                lineage=lineage,
                sample_index=job.sample_index,
                group_id=job.group_id,
                started_at=started_at,
                attempts=attempt,
                status=TrajectoryStatus.PENDING,
            )
            verification = await asyncio.to_thread(self._verify, job, pending)
            rewards = ctx.rewards.get(ctx.config.reward).compute(job.task, pending, verification)
            trajectory = pending.finalized(verification, rewards)
            model_failure = classify_model_failure(trajectory)
            if model_failure is not None:
                trajectory = trajectory.model_copy(
                    update={"failure": FailureRecord(kind=model_failure, attempts=attempt)}
                )
            ctx.metrics.reward.observe(rewards.total)
            return trajectory
        except EpisodeFailureError as failure:
            return self.failed_trajectory(
                job,
                failure,
                attempt=attempt,
                started_at=started_at,
                recorder=recorder,
                lineage=lineage,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - classify unexpected errors as infrastructure
            unexpected = EpisodeFailureError(
                FailureKind.INFRASTRUCTURE_ERROR,
                f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=3)[-300:]}",
                retryable=True,
            )
            return self.failed_trajectory(
                job,
                unexpected,
                attempt=attempt,
                started_at=started_at,
                recorder=recorder,
                lineage=lineage,
            )
        finally:
            if environment is not None:
                environment.close()
