"""Evaluation metrics computed from trajectories.

* ``verified_accuracy``: passed / completed. Infrastructure failures are excluded from the
  denominator and reported separately (``infra_failure_rate``) so flaky serving cannot masquerade
  as model regressions, and a gate can bound it.
* ``pass_at_1``: mean over tasks of the per-task pass fraction.
* ``pass_at_k``: the unbiased estimator ``1 - C(n-c, k) / C(n, k)`` (Chen et al., 2021), reported
  only when sampling is stochastic (temperature > 0) and every task has at least ``k`` samples.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from rlfactory.domain.result import MetricsSummary
from rlfactory.domain.task import TaskDomain
from rlfactory.domain.trajectory import FailureKind, Trajectory, TrajectoryStatus
from rlfactory.evaluation.failures import analyze_failures
from rlfactory.rewards.analysis import calibrate_rewards
from rlfactory.rewards.functions import format_ok


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    # Clamp so the interval always contains the point estimate despite float rounding at 0 and 1.
    return (max(0.0, min(p, centre - margin)), min(1.0, max(p, centre + margin)))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one task with ``n`` samples of which ``c`` passed."""
    if not 0 <= c <= n or k < 1 or k > n:
        raise ValueError(f"invalid pass@k inputs n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize_trajectories(
    trajectories: Sequence[Trajectory], *, ks: Sequence[int] = (1,), temperature: float = 0.0
) -> MetricsSummary:
    completed = [t for t in trajectories if t.status is TrajectoryStatus.COMPLETED]
    infra = [t for t in trajectories if t.status is TrajectoryStatus.FAILED]
    passed = sum(1 for t in completed if t.passed)
    per_task: dict[str, list[bool]] = defaultdict(list)
    for trajectory in completed:
        per_task[trajectory.task_id].append(trajectory.passed)
    pass_at_1 = (
        sum(sum(results) / len(results) for results in per_task.values()) / len(per_task)
        if per_task
        else None
    )
    pass_k: dict[int, float] = {}
    if temperature > 0 and per_task:
        for k in sorted(set(ks)):
            if k > 1 and all(len(results) >= k for results in per_task.values()):
                pass_k[k] = sum(pass_at_k(len(r), sum(r), k) for r in per_task.values()) / len(
                    per_task
                )
    tool_records = [record for t in completed for record in t.tool_records()]
    python = [t for t in trajectories if t.metadata.domain is TaskDomain.PYTHON]
    sandbox_errors = sum(
        1 for t in python if t.failure is not None and t.failure.kind is FailureKind.SANDBOX_ERROR
    )
    calibration = calibrate_rewards(completed)
    latencies = [sum(a.latency_s for a in t.actions) for t in completed]
    tokens = [t.completion_tokens for t in completed]
    chars = [sum(len(a.text) for a in t.actions) for t in completed]
    analysis = analyze_failures(trajectories)
    return MetricsSummary(
        num_tasks=len({t.task_id for t in trajectories}),
        num_trajectories=len(trajectories),
        num_completed=len(completed),
        num_infra_failures=len(infra),
        verified_accuracy=passed / len(completed) if completed else None,
        accuracy_ci95=wilson_interval(passed, len(completed)),
        pass_at_1=pass_at_1,
        pass_at_k=pass_k,
        format_compliance=(sum(1 for t in completed if format_ok(t)) / len(completed))
        if completed
        else None,
        tool_success_rate=(sum(1 for r in tool_records if r.succeeded) / len(tool_records))
        if tool_records
        else None,
        sandbox_failure_rate=sandbox_errors / len(python) if python else None,
        infra_failure_rate=len(infra) / len(trajectories) if trajectories else None,
        mean_reward=calibration.mean_reward,
        reward_std=calibration.reward_std,
        reward_correctness_correlation=calibration.point_biserial,
        mean_completion_tokens=sum(tokens) / len(tokens) if tokens else None,
        mean_completion_chars=sum(chars) / len(chars) if chars else None,
        total_completion_tokens=sum(tokens),
        latency_p50_s=_percentile(latencies, 0.5),
        latency_p95_s=_percentile(latencies, 0.95),
        failure_categories=analysis.counts,
    )


def summarize_by_domain(
    trajectories: Sequence[Trajectory], *, ks: Sequence[int] = (1,), temperature: float = 0.0
) -> dict[str, MetricsSummary]:
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.metadata.domain.value].append(trajectory)
    return {
        domain: summarize_trajectories(members, ks=ks, temperature=temperature)
        for domain, members in sorted(groups.items())
    }
