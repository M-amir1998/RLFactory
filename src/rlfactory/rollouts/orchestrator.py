"""Bounded-concurrency rollout orchestration with retries, timeouts, cancellation and resume.

Concurrency is bounded structurally: exactly ``concurrency`` consumer tasks pull jobs from a queue
of size ``2 x concurrency``; no task is created per job. Completed trajectories are buffered and
flushed to the store in batches, and always flushed on exit, including cancellation (Ctrl+C), so an
interrupted run can resume from its completed trajectory IDs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rlfactory.domain.trajectory import FailureKind, FailureOrigin, Trajectory, TrajectoryStatus
from rlfactory.rollouts.config import RolloutConfig
from rlfactory.rollouts.jobs import RolloutJob
from rlfactory.rollouts.worker import EpisodeFailureError, RolloutWorker
from rlfactory.telemetry.logging import get_logger, log_event
from rlfactory.telemetry.metrics import RolloutMetrics
from rlfactory.trajectories.store import TrajectoryStore

logger = get_logger("rollouts")


class RolloutSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    total_jobs: int
    skipped_existing: int
    attempted: int
    completed: int
    failed: int
    passed: int
    retries: int
    cancelled: bool
    wall_time_s: float
    trajectories_per_s: float | None
    completion_tokens: int
    tokens_per_s: float | None
    tokens_estimated: bool
    mean_reward: float | None
    pass_rate: float | None
    infra_failure_rate: float | None
    verification_rate: float | None
    failures_by_kind: dict[str, int]
    reason_codes: dict[str, int]
    by_domain: dict[str, dict[str, float]]
    started_at: datetime
    finished_at: datetime


class ProgressReporter(Protocol):
    def start(self, total: int, skipped: int) -> None: ...

    def advance(self, trajectory: Trajectory) -> None: ...

    def finish(self, summary: RolloutSummary) -> None: ...


class _Tally:
    def __init__(self) -> None:
        self.completed = 0
        self.failed = 0
        self.passed = 0
        self.retries = 0
        self.tokens = 0
        self.estimated = False
        self.rewards: list[float] = []
        self.failure_kinds: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()
        self.domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.domain_rewards: dict[str, list[float]] = defaultdict(list)

    def add(self, trajectory: Trajectory) -> None:
        domain = trajectory.metadata.domain.value
        self.domain_counts[domain]["total"] += 1
        self.tokens += trajectory.completion_tokens
        self.estimated = self.estimated or any(a.usage.estimated for a in trajectory.actions)
        if trajectory.failure is not None:
            self.failure_kinds[trajectory.failure.kind.value] += 1
        if trajectory.status is TrajectoryStatus.COMPLETED:
            self.completed += 1
            self.domain_counts[domain]["completed"] += 1
            if trajectory.verification is not None:
                self.reasons[trajectory.verification.reason_code] += 1
            if trajectory.passed:
                self.passed += 1
                self.domain_counts[domain]["passed"] += 1
            if trajectory.reward_total is not None:
                self.rewards.append(trajectory.reward_total)
                self.domain_rewards[domain].append(trajectory.reward_total)
        else:
            self.failed += 1
            self.domain_counts[domain]["failed"] += 1


class RolloutOrchestrator:
    def __init__(
        self,
        worker: RolloutWorker,
        store: TrajectoryStore,
        config: RolloutConfig,
        *,
        progress: ProgressReporter | None = None,
        metrics: RolloutMetrics | None = None,
    ) -> None:
        self.worker = worker
        self.store = store
        self.config = config
        self.progress = progress
        self.metrics = metrics or worker.ctx.metrics
        self._buffer: list[Trajectory] = []
        self._write_lock = asyncio.Lock()
        self._tally = _Tally()

    async def _flush(self) -> None:
        async with self._write_lock:
            if self._buffer:
                batch, self._buffer = self._buffer, []
                await asyncio.to_thread(self.store.write, batch)
            await asyncio.to_thread(self.store.flush)

    async def _record(self, trajectory: Trajectory) -> None:
        self._tally.add(trajectory)
        log_event(
            logger,
            "trajectory finished",
            logging.DEBUG,
            run_id=self.worker.ctx.run_id,
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            environment=trajectory.lineage.environment.name,
            policy=trajectory.policy_id,
            verifier=trajectory.lineage.verifier.name,
            status=trajectory.status.value,
            passed=trajectory.passed,
            reward=trajectory.reward_total,
            failure=trajectory.failure.kind.value if trajectory.failure else None,
            latency_s=round(sum(a.latency_s for a in trajectory.actions), 4),
            attempts=trajectory.metadata.attempts,
        )
        status = trajectory.status.value
        self.metrics.trajectories.inc(status=status, domain=trajectory.metadata.domain.value)
        if trajectory.failure is not None:
            self.metrics.failures.inc(
                kind=trajectory.failure.kind.value, origin=trajectory.failure.origin.value
            )
        if self.progress is not None:
            self.progress.advance(trajectory)
        async with self._write_lock:
            self._buffer.append(trajectory)
            should_flush = len(self._buffer) >= self.config.flush_every
        if should_flush:
            await self._flush()

    async def _run_with_retries(self, job: RolloutJob, cancel: asyncio.Event | None) -> Trajectory:
        retry = self.config.retry
        attempt = 1
        while True:
            try:
                trajectory = await asyncio.wait_for(
                    self.worker.run(job, attempt), timeout=self.config.episode_timeout_s
                )
            except TimeoutError:
                trajectory = self._episode_timeout(job, attempt)
            failure = trajectory.failure
            retryable = (
                trajectory.status is TrajectoryStatus.FAILED
                and failure is not None
                and failure.retryable
                and failure.kind in retry.retry_on
                and attempt < retry.max_attempts
                and not (cancel is not None and cancel.is_set())
            )
            if not retryable:
                return trajectory
            if failure is None:  # pragma: no cover - narrowed by `retryable`
                return trajectory
            self._tally.retries += 1
            self.metrics.retries.inc(kind=failure.kind.value)
            log_event(
                logger,
                "retrying rollout",
                trajectory_id=job.trajectory_id,
                task_id=job.task.task_id,
                attempt=attempt,
                failure=failure.kind.value,
            )
            await asyncio.sleep(retry.delay(attempt))
            attempt += 1

    def _episode_timeout(self, job: RolloutJob, attempt: int) -> Trajectory:
        failure = EpisodeFailureError(
            FailureKind.INFRASTRUCTURE_ERROR,
            f"episode exceeded {self.config.episode_timeout_s}s",
            retryable=True,
        )
        return self.worker.failed_trajectory(job, failure, attempt=attempt)

    async def run(
        self, jobs: Sequence[RolloutJob], cancel: asyncio.Event | None = None
    ) -> RolloutSummary:
        started_at = datetime.now(UTC)
        clock = time.perf_counter()
        run_id = self.worker.ctx.run_id
        existing = (
            await asyncio.to_thread(self.store.completed_ids, run_id)
            if self.config.resume
            else set()
        )
        pending = [job for job in jobs if job.trajectory_id not in existing]
        skipped = len(jobs) - len(pending)
        if self.progress is not None:
            self.progress.start(len(pending), skipped)
        log_event(logger, "rollouts starting", run_id=run_id, jobs=len(pending), skipped=skipped)
        queue: asyncio.Queue[RolloutJob | None] = asyncio.Queue(maxsize=2 * self.config.concurrency)
        attempted = 0

        async def producer() -> None:
            for job in pending:
                if cancel is not None and cancel.is_set():
                    break
                await queue.put(job)
            for _ in range(self.config.concurrency):
                await queue.put(None)

        async def consumer() -> None:
            nonlocal attempted
            while True:
                job = await queue.get()
                if job is None:
                    return
                if cancel is not None and cancel.is_set():
                    continue
                attempted += 1
                await self._record(await self._run_with_retries(job, cancel))

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(producer())
                for _ in range(self.config.concurrency):
                    group.create_task(consumer())
        finally:
            await self._flush()
        summary = self._summary(run_id, len(jobs), skipped, attempted, cancel, started_at, clock)
        if self.progress is not None:
            self.progress.finish(summary)
        log_event(
            logger,
            "rollouts finished",
            run_id=run_id,
            completed=summary.completed,
            failed=summary.failed,
            pass_rate=summary.pass_rate,
        )
        return summary

    def _summary(
        self,
        run_id: str,
        total: int,
        skipped: int,
        attempted: int,
        cancel: asyncio.Event | None,
        started_at: datetime,
        clock: float,
    ) -> RolloutSummary:
        t = self._tally
        elapsed = time.perf_counter() - clock
        finished = t.completed + t.failed
        infra = sum(
            count
            for kind, count in t.failure_kinds.items()
            if FailureKind(kind).origin is FailureOrigin.INFRASTRUCTURE
        )
        by_domain: dict[str, dict[str, float]] = {}
        for domain, counts in sorted(t.domain_counts.items()):
            rewards = t.domain_rewards[domain]
            by_domain[domain] = {
                "trajectories": float(counts["total"]),
                "completed": float(counts["completed"]),
                "failed": float(counts["failed"]),
                "pass_rate": counts["passed"] / counts["completed"] if counts["completed"] else 0.0,
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            }
        return RolloutSummary(
            run_id=run_id,
            total_jobs=total,
            skipped_existing=skipped,
            attempted=attempted,
            completed=t.completed,
            failed=t.failed,
            passed=t.passed,
            retries=t.retries,
            cancelled=cancel is not None and cancel.is_set(),
            wall_time_s=round(elapsed, 3),
            trajectories_per_s=round(finished / elapsed, 3) if elapsed > 0 and finished else None,
            completion_tokens=t.tokens,
            tokens_per_s=round(t.tokens / elapsed, 1) if elapsed > 0 and t.tokens else None,
            tokens_estimated=t.estimated,
            mean_reward=sum(t.rewards) / len(t.rewards) if t.rewards else None,
            pass_rate=t.passed / t.completed if t.completed else None,
            infra_failure_rate=infra / finished if finished else None,
            verification_rate=t.completed / finished if finished else None,
            failures_by_kind=dict(sorted(t.failure_kinds.items())),
            reason_codes=dict(sorted(t.reasons.items())),
            by_domain=by_domain,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )


__all__ = ["ProgressReporter", "RolloutOrchestrator", "RolloutSummary"]
