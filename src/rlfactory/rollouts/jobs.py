"""Deterministic rollout job planning (idempotent trajectory IDs)."""

from __future__ import annotations

from collections.abc import Iterable

from rlfactory.domain.common import DomainModel, stable_hash, stable_seed
from rlfactory.domain.task import Task
from rlfactory.rollouts.config import RolloutConfig


class RolloutJob(DomainModel):
    trajectory_id: str
    task: Task
    sample_index: int
    seed: int
    group_id: str


def plan_jobs(
    tasks: Iterable[Task],
    config: RolloutConfig,
    *,
    policy_id: str,
    checkpoint: str | None = None,
) -> list[RolloutJob]:
    """One job per (task, sample). IDs depend only on content and configuration.

    The ID covers the task fingerprint, policy and checkpoint, generation config, seed, sample
    index, verifier and reward, so re-running an identical configuration maps onto the same IDs
    (and resume skips them), while any change that alters the outcome yields new IDs.
    """
    generation = config.generation.fingerprint()
    jobs: list[RolloutJob] = []
    for task in tasks:
        verifier = config.verifier_overrides.get(task.domain, task.verifier)
        group_id = "grp-" + stable_hash(
            task.task_id,
            task.fingerprint(),
            policy_id,
            checkpoint,
            generation,
            config.seed,
            verifier,
            length=16,
        )
        for sample in range(config.samples_per_task):
            trajectory_id = "trj-" + stable_hash(
                task.task_id,
                task.fingerprint(),
                policy_id,
                checkpoint,
                generation,
                config.seed,
                sample,
                verifier,
                config.reward,
                length=20,
            )
            jobs.append(
                RolloutJob(
                    trajectory_id=trajectory_id,
                    task=task,
                    sample_index=sample,
                    seed=stable_seed(config.seed, task.task_id, sample) % (2**31 - 1),
                    group_id=group_id,
                )
            )
    return jobs
