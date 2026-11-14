"""JSONL trajectory store: human-readable, dependency-free, for small runs and debugging."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from rlfactory.domain.trajectory import Trajectory, TrajectoryStatus
from rlfactory.errors import StoreError
from rlfactory.trajectories.store import TrajectoryQuery, TrajectorySummary, summarize


def matches(summary: TrajectorySummary, query: TrajectoryQuery) -> bool:
    checks: list[tuple[object, object]] = [
        (query.run_id, summary.run_id),
        (query.task_id, summary.task_id),
        (query.domain, summary.domain),
        (query.environment, summary.environment),
        (query.policy_id, summary.policy_id),
        (query.checkpoint, summary.checkpoint),
        (query.verifier, summary.verifier),
        (query.failure_kind, summary.failure_kind),
        (query.seed, summary.seed),
        (query.passed, summary.passed),
        (query.status.value if query.status else None, summary.status),
    ]
    if any(expected is not None and expected != actual for expected, actual in checks):
        return False
    if query.task_ids is not None and summary.task_id not in query.task_ids:
        return False
    if query.min_reward is not None and (
        summary.reward is None or summary.reward < query.min_reward
    ):
        return False
    return not (
        query.max_reward is not None
        and (summary.reward is None or summary.reward > query.max_reward)
    )


class JSONLTrajectoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "trajectories.jsonl"
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    def write(self, trajectories: Sequence[Trajectory]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for trajectory in trajectories:
                if trajectory.status is TrajectoryStatus.PENDING:
                    raise StoreError("refusing to persist a pending trajectory")
                handle.write(trajectory.model_dump_json() + "\n")

    def flush(self) -> None:
        return None

    def _latest(self) -> Iterator[Trajectory]:
        if not self.path.exists():
            return
        latest: dict[str, Trajectory] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    trajectory = Trajectory.model_validate_json(line)
                    latest[trajectory.trajectory_id] = trajectory
        yield from (latest[key] for key in sorted(latest))

    def completed_ids(self, run_id: str | None = None) -> set[str]:
        return {
            t.trajectory_id
            for t in self._latest()
            if t.status is TrajectoryStatus.COMPLETED
            and (run_id is None or t.lineage.run_id == run_id)
        }

    def query(self, query: TrajectoryQuery) -> list[TrajectorySummary]:
        rows = [
            s for t in self._latest() if matches(s := summarize(t, payload_retained=True), query)
        ]
        return rows[: query.limit] if query.limit else rows

    def load(self, query: TrajectoryQuery) -> list[Trajectory]:
        rows = [t for t in self._latest() if matches(summarize(t, payload_retained=True), query)]
        return rows[: query.limit] if query.limit else rows

    def aggregate(self, group_by: Sequence[str], query: TrajectoryQuery) -> list[dict[str, Any]]:
        return aggregate_summaries(self.query(query), group_by)


def aggregate_summaries(
    summaries: Sequence[TrajectorySummary], group_by: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[TrajectorySummary]] = defaultdict(list)
    for summary in summaries:
        groups[tuple(getattr(summary, key) for key in group_by)].append(summary)
    results: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: json.dumps(item[0], default=str)):
        completed = [r for r in rows if r.status == TrajectoryStatus.COMPLETED.value]
        rewards = [r.reward for r in rows if r.reward is not None]
        results.append(
            {
                **dict(zip(group_by, key, strict=True)),
                "trajectories": len(rows),
                "completed": len(completed),
                "failed": len(rows) - len(completed),
                "pass_rate": (sum(1 for r in completed if r.passed) / len(completed))
                if completed
                else None,
                "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
            }
        )
    return results
