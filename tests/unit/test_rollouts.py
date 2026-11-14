"""Rollout engine: success, timeouts, retries, failure taxonomy, resume, cancellation, concurrency."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import FailureKind, FailureOrigin, TrajectoryStatus
from rlfactory.errors import PolicyError, PolicyTimeoutError
from rlfactory.policies.base import GenerationRequest, GenerationResult
from rlfactory.policies.mock import MockPolicy, MockPolicyProfile
from rlfactory.policies.scripted import ScriptedPolicy
from rlfactory.rollouts.config import RetryPolicy, RolloutConfig
from rlfactory.rollouts.jobs import plan_jobs
from rlfactory.rollouts.orchestrator import RolloutSummary
from rlfactory.rollouts.service import run_rollouts
from rlfactory.rollouts.worker import classify_model_failure
from rlfactory.sandbox.mock import MockSandbox
from rlfactory.trajectories.memory_store import MemoryTrajectoryStore
from rlfactory.trajectories.store import TrajectoryQuery
from rlfactory.verifiers.base import BaseVerifier
from tests.helpers import deps, make_task, tasks_for

FAST_RETRY = RetryPolicy(max_attempts=3, backoff_initial_s=0.0)


def _run(
    tasks: Sequence[Task], policy, config: RolloutConfig, store=None, dependencies=None, cancel=None
) -> RolloutSummary:  # type: ignore[no-untyped-def]
    return asyncio.run(
        run_rollouts(
            run_id="run",
            tasks=tasks,
            policy=policy,
            config=config,
            store=store or MemoryTrajectoryStore(),
            deps=dependencies or deps(),
            cancel=cancel,
        )
    )


def test_successful_rollouts_with_lineage() -> None:
    tasks = tasks_for(TaskDomain.ARITHMETIC)
    store = MemoryTrajectoryStore()
    summary = _run(
        tasks,
        MockPolicy(tasks, MockPolicyProfile(correct_probability=1.0, malformed_probability=0.0)),
        RolloutConfig(samples_per_task=2),
        store,
    )
    assert summary.completed == 8 and summary.failed == 0 and summary.pass_rate == 1.0
    trajectory = store.load(TrajectoryQuery())[0]
    lineage = trajectory.lineage
    assert (
        lineage.run_id == "run"
        and lineage.verifier.name == "numeric"
        and lineage.environment.name == "arithmetic"
    )
    assert (
        lineage.task_fingerprint
        == next(t for t in tasks if t.task_id == trajectory.task_id).fingerprint()
    )
    assert lineage.code.rlfactory_version


def test_job_ids_are_deterministic_and_config_sensitive() -> None:
    tasks = tasks_for(TaskDomain.TRANSFORMATION)
    config = RolloutConfig(samples_per_task=2)
    first = [j.trajectory_id for j in plan_jobs(tasks, config, policy_id="p")]
    assert first == [j.trajectory_id for j in plan_jobs(tasks, config, policy_id="p")]
    assert len(set(first)) == len(first)
    assert first != [j.trajectory_id for j in plan_jobs(tasks, config, policy_id="q")]
    assert first != [
        j.trajectory_id
        for j in plan_jobs(
            tasks, config.model_copy(update={"reward": "correctness_format"}), policy_id="p"
        )
    ]
    groups = {j.group_id for j in plan_jobs(tasks, config, policy_id="p")}
    assert len(groups) == len(tasks)


def test_policy_timeouts_are_retried_then_recorded_as_infrastructure() -> None:
    task = make_task("arithmetic.integer_chain")
    always_timeout = ScriptedPolicy(lambda r: (_ for _ in ()).throw(PolicyTimeoutError()))  # type: ignore[arg-type]
    store = MemoryTrajectoryStore()
    summary = _run(
        [task], always_timeout, RolloutConfig(samples_per_task=1, retry=FAST_RETRY), store
    )
    assert summary.failed == 1 and summary.retries == 2
    trajectory = store.load(TrajectoryQuery())[0]
    assert trajectory.status is TrajectoryStatus.FAILED
    assert trajectory.failure is not None and trajectory.failure.kind is FailureKind.POLICY_TIMEOUT
    assert (
        trajectory.failure.origin is FailureOrigin.INFRASTRUCTURE
        and trajectory.metadata.attempts == 3
    )
    assert not trajectory.is_trainable


def test_transient_failure_recovers_on_retry() -> None:
    task = make_task("arithmetic.integer_chain")
    answer = f"<answer>{task.payload.answer}</answer>"  # type: ignore[union-attr]
    policy = ScriptedPolicy(
        {task.task_id: [answer]}, failures={(task.task_id, 1): PolicyError("503", retryable=True)}
    )
    store = MemoryTrajectoryStore()
    summary = _run([task], policy, RolloutConfig(samples_per_task=1, retry=FAST_RETRY), store)
    assert summary.completed == 1 and summary.retries == 1 and summary.passed == 1
    assert store.load(TrajectoryQuery())[0].metadata.attempts == 2


def test_non_retryable_errors_are_not_retried() -> None:
    task = make_task("arithmetic.integer_chain")
    policy = ScriptedPolicy(
        {}, failures={(task.task_id, 1): PolicyError("HTTP 400", retryable=False)}
    )
    summary = _run([task], policy, RolloutConfig(samples_per_task=1, retry=FAST_RETRY))
    assert summary.failed == 1 and summary.retries == 0
    assert summary.failures_by_kind == {"policy_error": 1}


def test_model_failures_complete_and_are_not_retried() -> None:
    task = make_task("arithmetic.integer_chain")
    store = MemoryTrajectoryStore()
    summary = _run(
        [task],
        ScriptedPolicy({task.task_id: ["no tags here"]}),
        RolloutConfig(samples_per_task=1, retry=FAST_RETRY),
        store,
    )
    assert summary.completed == 1 and summary.retries == 0 and summary.passed == 0
    trajectory = store.load(TrajectoryQuery())[0]
    assert trajectory.failure is not None and trajectory.failure.kind is FailureKind.INVALID_OUTPUT
    assert (
        trajectory.is_trainable and classify_model_failure(trajectory) is FailureKind.INVALID_OUTPUT
    )


def test_sandbox_infrastructure_errors_are_retried() -> None:
    task = make_task("python.sum_of_evens")
    policy = ScriptedPolicy({task.task_id: ["```python\ndef sum_of_evens(v):\n    return 0\n```"]})
    summary = _run(
        [task],
        policy,
        RolloutConfig(samples_per_task=1, retry=FAST_RETRY),
        dependencies=deps(MockSandbox(fail_with="docker down")),
    )
    assert summary.failed == 1 and summary.retries == 2
    assert summary.failures_by_kind == {"sandbox_error": 1}


class _CrashingVerifier(BaseVerifier):
    def verify(self, task, trajectory):  # type: ignore[no-untyped-def]
        raise RuntimeError("verifier bug")


def test_verifier_errors_are_separated_and_not_retried() -> None:
    task = make_task("arithmetic.integer_chain")
    dependencies = deps()
    dependencies.verifiers.register("numeric", _CrashingVerifier("numeric"))
    summary = _run(
        [task],
        ScriptedPolicy({task.task_id: ["<answer>1</answer>"]}),
        RolloutConfig(samples_per_task=1, retry=FAST_RETRY),
        dependencies=dependencies,
    )
    assert summary.failures_by_kind == {"verifier_error": 1} and summary.retries == 0


def test_resume_skips_completed_but_retries_infra_failures() -> None:
    tasks = tasks_for(TaskDomain.TRANSFORMATION)
    store = MemoryTrajectoryStore()
    flaky = MockPolicy(tasks, MockPolicyProfile(error_probability=0.6), seed=2)
    first = _run(
        tasks, flaky, RolloutConfig(samples_per_task=2, retry=RetryPolicy(max_attempts=1)), store
    )
    assert first.failed > 0
    healthy = MockPolicy(tasks, MockPolicyProfile(), seed=2)
    second = _run(tasks, healthy, RolloutConfig(samples_per_task=2), store)
    assert second.skipped_existing == first.completed
    assert second.attempted == first.failed
    third = _run(tasks, healthy, RolloutConfig(samples_per_task=2), store)
    assert third.attempted == 0 and third.skipped_existing == len(tasks) * 2


def test_cancellation_stops_and_flushes() -> None:
    tasks = tasks_for(TaskDomain.ARITHMETIC, per_family=5)
    store = MemoryTrajectoryStore()
    cancel = asyncio.Event()

    class CancellingPolicy(MockPolicy):
        calls = 0

        async def generate(self, request: GenerationRequest) -> GenerationResult:
            CancellingPolicy.calls += 1
            if CancellingPolicy.calls == 3:
                cancel.set()
            return await super().generate(request)

    summary = _run(
        tasks,
        CancellingPolicy(tasks),
        RolloutConfig(samples_per_task=2, concurrency=1),
        store,
        cancel=cancel,
    )
    assert summary.cancelled
    assert 0 < len(store.load(TrajectoryQuery())) < len(tasks) * 2


def test_concurrency_is_bounded() -> None:
    tasks = tasks_for(TaskDomain.ARITHMETIC, per_family=4)

    class CountingPolicy(MockPolicy):
        active = 0
        peak = 0

        async def generate(self, request: GenerationRequest) -> GenerationResult:
            CountingPolicy.active += 1
            CountingPolicy.peak = max(CountingPolicy.peak, CountingPolicy.active)
            await asyncio.sleep(0.01)
            try:
                return await super().generate(request)
            finally:
                CountingPolicy.active -= 1

    _run(tasks, CountingPolicy(tasks), RolloutConfig(samples_per_task=2, concurrency=3))
    assert CountingPolicy.peak <= 3


def test_episode_timeout_is_infrastructure_failure() -> None:
    task = make_task("arithmetic.integer_chain")

    class SlowPolicy(MockPolicy):
        async def generate(self, request: GenerationRequest) -> GenerationResult:
            await asyncio.sleep(5)
            return await super().generate(request)

    summary = _run(
        [task],
        SlowPolicy([task]),
        RolloutConfig(samples_per_task=1, episode_timeout_s=0.2, retry=RetryPolicy(max_attempts=1)),
    )
    assert summary.failures_by_kind == {"infrastructure_error": 1}


def test_unknown_reward_fails_fast() -> None:
    task = make_task("arithmetic.integer_chain")
    with pytest.raises(Exception, match="unknown reward"):
        _run([task], MockPolicy([task]), RolloutConfig(reward="nope"))


def test_multi_step_tool_rollouts() -> None:
    tasks = tasks_for(TaskDomain.TOOL_USE)
    store = MemoryTrajectoryStore()
    _run(
        tasks,
        MockPolicy(tasks, MockPolicyProfile(correct_probability=1.0, malformed_probability=0.0)),
        RolloutConfig(samples_per_task=1),
        store,
    )
    for trajectory in store.load(TrajectoryQuery()):
        assert trajectory.passed and len(trajectory.actions) > 1
        assert len(trajectory.observations) == len(trajectory.actions) + 1
