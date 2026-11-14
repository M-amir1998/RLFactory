"""Curriculum samplers: uniform, difficulty-balanced, failure-focused and reward-stratified."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from random import Random

from rlfactory.domain.common import stable_seed
from rlfactory.domain.task import Task
from rlfactory.domain.trajectory import Trajectory


class SamplingStrategy(StrEnum):
    UNIFORM = "uniform"
    DIFFICULTY_BALANCED = "difficulty_balanced"
    FAILURE_FOCUSED = "failure_focused"
    REWARD_STRATIFIED = "reward_stratified"


@dataclass(frozen=True)
class TaskStats:
    attempts: int
    passes: int
    mean_reward: float | None

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.passes / self.attempts if self.attempts else 0.0


def task_stats(trajectories: Iterable[Trajectory]) -> dict[str, TaskStats]:
    """Aggregate per-task attempts, passes and mean reward from completed trajectories."""
    attempts: dict[str, int] = defaultdict(int)
    passes: dict[str, int] = defaultdict(int)
    rewards: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        if not trajectory.is_trainable:
            continue
        attempts[trajectory.task_id] += 1
        passes[trajectory.task_id] += int(trajectory.passed)
        if trajectory.reward_total is not None:
            rewards[trajectory.task_id].append(trajectory.reward_total)
    return {
        task_id: TaskStats(
            attempts=count,
            passes=passes[task_id],
            mean_reward=(sum(rewards[task_id]) / len(rewards[task_id]))
            if rewards[task_id]
            else None,
        )
        for task_id, count in attempts.items()
    }


def _round_robin(groups: list[list[Task]], n: int) -> list[Task]:
    chosen: list[Task] = []
    index = 0
    while len(chosen) < n and any(groups):
        group = groups[index % len(groups)]
        if group:
            chosen.append(group.pop())
        index += 1
    return chosen


class CurriculumSampler:
    """Deterministic task sampling without replacement (``n`` is capped at the pool size)."""

    def __init__(
        self,
        strategy: SamplingStrategy,
        *,
        seed: int = 0,
        stats: Mapping[str, TaskStats] | None = None,
        smoothing: float = 1.0,
        reward_buckets: int = 3,
    ) -> None:
        self.strategy = strategy
        self.seed = seed
        self.stats = dict(stats or {})
        self.smoothing = smoothing
        self.reward_buckets = reward_buckets

    def sample(self, tasks: Sequence[Task], n: int) -> list[Task]:
        rng = Random(stable_seed(self.seed, self.strategy.value, len(tasks), n))
        pool = sorted(tasks, key=lambda t: t.task_id)
        n = min(n, len(pool))
        if self.strategy is SamplingStrategy.UNIFORM:
            return rng.sample(pool, n)
        if self.strategy is SamplingStrategy.DIFFICULTY_BALANCED:
            levels: dict[int, list[Task]] = defaultdict(list)
            for task in pool:
                level = task.metadata.difficulty.level if task.metadata.difficulty else 0
                levels[level].append(task)
            groups = [levels[level] for level in sorted(levels)]
            for group in groups:
                rng.shuffle(group)
            return _round_robin(groups, n)
        if self.strategy is SamplingStrategy.FAILURE_FOCUSED:
            return self._failure_focused(pool, n, rng)
        return self._reward_stratified(pool, n, rng)

    def failure_weight(self, task: Task) -> float:
        """Laplace-smoothed failure rate; unseen tasks get the uninformed prior 0.5."""
        stats = self.stats.get(task.task_id)
        s = self.smoothing
        if stats is None:
            return 0.5
        return (stats.attempts - stats.passes + s) / (stats.attempts + 2 * s)

    def _failure_focused(self, pool: list[Task], n: int, rng: Random) -> list[Task]:
        # Efraimidis-Spirakis weighted sampling without replacement: key = u^(1/w).
        keyed = []
        for task in pool:
            weight = max(self.failure_weight(task), 1e-6)
            keyed.append((math.log(rng.random() or 1e-12) / weight, task))
        keyed.sort(key=lambda item: item[0], reverse=True)
        return [task for _, task in keyed[:n]]

    def _reward_stratified(self, pool: list[Task], n: int, rng: Random) -> list[Task]:
        known = [
            t
            for t in pool
            if self.stats.get(t.task_id) and self.stats[t.task_id].mean_reward is not None
        ]
        unknown = [t for t in pool if t not in known]
        known.sort(key=lambda t: self.stats[t.task_id].mean_reward or 0.0)
        buckets: list[list[Task]] = [[] for _ in range(self.reward_buckets)]
        for index, task in enumerate(known):
            buckets[
                min(self.reward_buckets - 1, index * self.reward_buckets // max(1, len(known)))
            ].append(task)
        groups = [bucket for bucket in buckets if bucket] + ([unknown] if unknown else [])
        for group in groups:
            rng.shuffle(group)
        return _round_robin(groups, n)
