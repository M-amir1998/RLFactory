"""Shared test helpers: deterministic tasks and fully verified trajectories without a policy."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from random import Random

from rlfactory.domain.common import CodeVersion, ComponentRef, stable_hash
from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import Trajectory, TrajectoryStatus
from rlfactory.policies.oracle import ResponseMode, ResponseOracle
from rlfactory.rollouts.episode import offline_lineage, run_sync_episode
from rlfactory.rollouts.service import RolloutDependencies
from rlfactory.sandbox.base import ResourceLimits, Sandbox
from rlfactory.sandbox.mock import MockSandbox
from rlfactory.tasks.generators.catalog import default_template_registry
from rlfactory.tasks.prompts import PromptBuilder

TEMPLATES = default_template_registry()
CODE = CodeVersion(rlfactory_version="test")


def make_task(family: str, *, seed: int = 0, difficulty: int = 2) -> Task:
    template = TEMPLATES.get(family)
    return template.make_task(template.sample(Random(seed), difficulty))


def tasks_for(domain: TaskDomain, *, per_family: int = 1, seed: int = 0) -> list[Task]:
    return [
        template.make_task(template.sample(Random(seed + i), 2))
        for template in TEMPLATES.by_domain(domain)
        for i in range(per_family)
    ]


def deps(sandbox: Sandbox | None = None, *, wall_time_s: float = 5.0) -> RolloutDependencies:
    return RolloutDependencies.default(
        sandbox=sandbox or MockSandbox(), python_limits=ResourceLimits(wall_time_s=wall_time_s)
    )


def play(
    task: Task,
    responses: Sequence[str],
    *,
    dependencies: RolloutDependencies | None = None,
    verifier: str | None = None,
    reward: str = "binary",
    sample_index: int = 0,
    group_id: str = "group-test",
    run_id: str = "test-run",
) -> Trajectory:
    """Drive the real environment with scripted responses, then verify and reward."""
    d = dependencies or deps()
    environment = d.environments.create(task.environment, d.environment_context)
    try:
        recorder = run_sync_episode(
            environment,
            task,
            d.prompts,
            lambda turn, _messages: responses[min(turn, len(responses) - 1)],
        )
    finally:
        environment.close()
    checker = d.verifiers.get(verifier or task.verifier)
    reward_fn = d.rewards.get(reward)
    pending = recorder.build(
        trajectory_id=f"trj-{stable_hash(task.task_id, sample_index, list(responses), length=16)}",
        policy_id="test-policy",
        lineage=offline_lineage(
            task,
            run_id=run_id,
            environment=d.environments.spec(task.environment).ref,
            verifier=ComponentRef(name=checker.name, version=checker.version),
            reward=ComponentRef(name=reward_fn.name, version=reward_fn.version),
            prompt_template=d.prompts.template,
            policy_id="test-policy",
            code=CODE,
        ),
        sample_index=sample_index,
        group_id=group_id,
        started_at=datetime.now(UTC),
        status=TrajectoryStatus.PENDING,
    )
    verification = checker.verify(task, pending)
    return pending.finalized(verification, reward_fn.compute(task, pending, verification))


def oracle_response(task: Task, mode: ResponseMode, *, turn: int = 0, seed: int = 0) -> str:
    return ResponseOracle().respond(
        task, mode, verbose=False, turn=turn, rng=Random(seed), canonical=True
    )


def oracle_play(
    task: Task,
    mode: ResponseMode,
    *,
    dependencies: RolloutDependencies | None = None,
    **kwargs: object,
) -> Trajectory:
    """Play every turn with the oracle (multi-step tasks included)."""
    responses = [oracle_response(task, mode, turn=turn) for turn in range(32)]
    return play(task, responses, dependencies=dependencies, **kwargs)  # type: ignore[arg-type]


PROMPTS = PromptBuilder()
