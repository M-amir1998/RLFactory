"""In-memory trajectory store (tests and ephemeral evaluation runs)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rlfactory.domain.trajectory import Trajectory, TrajectoryStatus
from rlfactory.errors import StoreError
from rlfactory.trajectories.jsonl_store import aggregate_summaries, matches
from rlfactory.trajectories.store import TrajectoryQuery, TrajectorySummary, summarize


class MemoryTrajectoryStore:
    def __init__(self) -> None:
        self._latest: dict[str, Trajectory] = {}

    def write(self, trajectories: Sequence[Trajectory]) -> None:
        for trajectory in trajectories:
            if trajectory.status is TrajectoryStatus.PENDING:
                raise StoreError("refusing to persist a pending trajectory")
            self._latest[trajectory.trajectory_id] = trajectory

    def flush(self) -> None:
        return None

    def completed_ids(self, run_id: str | None = None) -> set[str]:
        return {
            t.trajectory_id
            for t in self._latest.values()
            if t.status is TrajectoryStatus.COMPLETED
            and (run_id is None or t.lineage.run_id == run_id)
        }

    def _ordered(self) -> list[Trajectory]:
        return [self._latest[key] for key in sorted(self._latest)]

    def query(self, query: TrajectoryQuery) -> list[TrajectorySummary]:
        rows = [
            s for t in self._ordered() if matches(s := summarize(t, payload_retained=True), query)
        ]
        return rows[: query.limit] if query.limit else rows

    def load(self, query: TrajectoryQuery) -> list[Trajectory]:
        rows = [t for t in self._ordered() if matches(summarize(t, payload_retained=True), query)]
        return rows[: query.limit] if query.limit else rows

    def aggregate(self, group_by: Sequence[str], query: TrajectoryQuery) -> list[dict[str, Any]]:
        return aggregate_summaries(self.query(query), group_by)
