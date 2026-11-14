"""Local Parquet trajectory store with a DuckDB query layer.

Layout: ``<root>/part-<utc-timestamp>-<uuid>.parquet`` shards, zstd-compressed, each written
atomically (temp file + rename). Every row has flat summary columns for filtering/statistics plus a
``payload`` column holding the full trajectory JSON when the retention policy keeps it.

Records are append-only. A trajectory may be written more than once (an infrastructure failure
followed by a successful resume); all queries use the **latest record per trajectory_id**.
A future object-store backend only needs to provide the same shard listing and append semantics.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from rlfactory.domain.trajectory import Trajectory, TrajectoryStatus
from rlfactory.errors import StoreError
from rlfactory.trajectories.store import (
    RetentionPolicy,
    TrajectoryQuery,
    TrajectorySummary,
    should_retain,
    summarize,
)

# Timestamps are stored as naive UTC: DuckDB needs pytz to return tz-aware values, and the
# summary model re-attaches UTC on read.
_TIMESTAMP = pa.timestamp("us")
_TIMESTAMP_COLUMNS = ("started_at", "finished_at", "written_at")

SCHEMA = pa.schema(
    [
        ("trajectory_id", pa.string()),
        ("run_id", pa.string()),
        ("task_id", pa.string()),
        ("task_version", pa.string()),
        ("task_family", pa.string()),
        ("domain", pa.string()),
        ("environment", pa.string()),
        ("environment_version", pa.string()),
        ("verifier", pa.string()),
        ("verifier_version", pa.string()),
        ("reward_fn", pa.string()),
        ("reward_version", pa.string()),
        ("policy_id", pa.string()),
        ("model", pa.string()),
        ("checkpoint", pa.string()),
        ("prompt_template", pa.string()),
        ("seed", pa.int64()),
        ("sample_index", pa.int64()),
        ("group_id", pa.string()),
        ("status", pa.string()),
        ("failure_kind", pa.string()),
        ("failure_origin", pa.string()),
        ("attempts", pa.int64()),
        ("passed", pa.bool_()),
        ("score", pa.float64()),
        ("reason_code", pa.string()),
        ("reward", pa.float64()),
        ("reward_correctness", pa.float64()),
        ("reward_format", pa.float64()),
        ("reward_length", pa.float64()),
        ("reward_tool_cost", pa.float64()),
        ("reward_safety", pa.float64()),
        ("num_turns", pa.int64()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("completion_chars", pa.int64()),
        ("tokens_estimated", pa.bool_()),
        ("policy_latency_s", pa.float64()),
        ("wall_time_s", pa.float64()),
        ("truncated", pa.bool_()),
        ("generation_fingerprint", pa.string()),
        ("git_sha", pa.string()),
        ("rlfactory_version", pa.string()),
        ("started_at", _TIMESTAMP),
        ("finished_at", _TIMESTAMP),
        ("payload_retained", pa.bool_()),
        ("written_at", _TIMESTAMP),
        ("payload", pa.string()),
    ]
)

SUMMARY_COLUMNS = [name for name in SCHEMA.names if name not in {"payload", "written_at"}]
GROUPABLE_COLUMNS = frozenset(
    {
        "run_id",
        "task_id",
        "task_family",
        "domain",
        "environment",
        "verifier",
        "reward_fn",
        "policy_id",
        "model",
        "checkpoint",
        "status",
        "failure_kind",
        "failure_origin",
        "reason_code",
        "passed",
        "seed",
        "group_id",
    }
)


def _where(query: TrajectoryQuery) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    simple = {
        "run_id": query.run_id,
        "task_id": query.task_id,
        "domain": query.domain,
        "environment": query.environment,
        "policy_id": query.policy_id,
        "checkpoint": query.checkpoint,
        "verifier": query.verifier,
        "failure_kind": query.failure_kind,
        "seed": query.seed,
        "passed": query.passed,
        "status": query.status.value if query.status is not None else None,
    }
    for column, value in simple.items():
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    if query.task_ids is not None:
        if not query.task_ids:
            clauses.append("FALSE")
        else:
            clauses.append(f"task_id IN ({', '.join('?' for _ in query.task_ids)})")
            params.extend(query.task_ids)
    if query.min_reward is not None:
        clauses.append("reward >= ?")
        params.append(query.min_reward)
    if query.max_reward is not None:
        clauses.append("reward <= ?")
        params.append(query.max_reward)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


class LocalParquetTrajectoryStore:
    def __init__(
        self,
        root: Path,
        *,
        retention: RetentionPolicy = RetentionPolicy.ALL,
        sample_rate: float = 0.1,
        batch_size: int = 256,
        compression: str = "zstd",
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be within [0, 1]")
        self.root = root
        self.retention = retention
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.compression = compression
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ writing
    def write(self, trajectories: Sequence[Trajectory]) -> None:
        rows = []
        now = datetime.now(UTC)
        for trajectory in trajectories:
            if trajectory.status is TrajectoryStatus.PENDING:
                raise StoreError(
                    f"refusing to persist pending trajectory {trajectory.trajectory_id}"
                )
            retained = should_retain(trajectory, self.retention, self.sample_rate)
            row = summarize(trajectory, payload_retained=retained).model_dump()
            row["written_at"] = now
            row["payload"] = trajectory.model_dump_json() if retained else None
            for column in _TIMESTAMP_COLUMNS:
                row[column] = row[column].astimezone(UTC).replace(tzinfo=None)
            rows.append(row)
        with self._lock:
            self._buffer.extend(rows)
            if len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=SCHEMA)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        target = self.root / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = target.with_suffix(".parquet.tmp")
        try:
            pq.write_table(table, tmp, compression=self.compression)
            tmp.replace(target)
        except (OSError, pa.ArrowException) as exc:
            raise StoreError(f"failed to write trajectory shard {target}: {exc}") from exc
        self._buffer.clear()

    # ------------------------------------------------------------------ reading
    def shards(self) -> list[Path]:
        return sorted(self.root.glob("part-*.parquet"))

    def _latest_sql(self, columns: str) -> str:
        # Column lists come from the fixed schema; every user-supplied value is a bound parameter.
        return (
            f"SELECT {columns} FROM read_parquet(?, union_by_name=true) "
            "QUALIFY row_number() OVER (PARTITION BY trajectory_id ORDER BY written_at DESC) = 1"
        )

    def _execute(self, sql: str, params: list[Any]) -> list[tuple[Any, ...]]:
        if not self.shards():
            return []
        glob = (self.root / "part-*.parquet").as_posix()
        try:
            with duckdb.connect() as connection:
                return connection.execute(sql, [glob, *params]).fetchall()
        except duckdb.Error as exc:
            raise StoreError(f"trajectory query failed: {exc}") from exc

    def completed_ids(self, run_id: str | None = None) -> set[str]:
        self.flush()
        query = TrajectoryQuery(run_id=run_id, status=TrajectoryStatus.COMPLETED)
        where, params = _where(query)
        sql = f"SELECT trajectory_id FROM ({self._latest_sql('trajectory_id, status, run_id')}){where}"
        return {row[0] for row in self._execute(sql, params)}

    def query(self, query: TrajectoryQuery) -> list[TrajectorySummary]:
        self.flush()
        columns = ", ".join(SUMMARY_COLUMNS)
        where, params = _where(query)
        sql = f"SELECT {columns} FROM ({self._latest_sql(columns + ', written_at')}){where} ORDER BY trajectory_id"
        if query.limit is not None:
            sql += f" LIMIT {int(query.limit)}"
        rows = self._execute(sql, params)
        return [
            TrajectorySummary.model_validate(dict(zip(SUMMARY_COLUMNS, row, strict=True)))
            for row in rows
        ]

    def load(self, query: TrajectoryQuery) -> list[Trajectory]:
        self.flush()
        columns = ", ".join(SUMMARY_COLUMNS)
        where, params = _where(query)
        inner = self._latest_sql(columns + ", written_at, payload")
        sql = (
            f"SELECT payload FROM ({inner}){where} AND payload IS NOT NULL"
            if where
            else (f"SELECT payload FROM ({inner}) WHERE payload IS NOT NULL")
        )
        sql += " ORDER BY trajectory_id"
        if query.limit is not None:
            sql += f" LIMIT {int(query.limit)}"
        return [Trajectory.model_validate_json(row[0]) for row in self._execute(sql, params)]

    def aggregate(self, group_by: Sequence[str], query: TrajectoryQuery) -> list[dict[str, Any]]:
        unknown = set(group_by) - GROUPABLE_COLUMNS
        if unknown:
            raise StoreError(
                f"cannot group by {sorted(unknown)}; allowed: {sorted(GROUPABLE_COLUMNS)}"
            )
        self.flush()
        columns = ", ".join(SUMMARY_COLUMNS)
        where, params = _where(query)
        keys = ", ".join(group_by)
        select_keys = f"{keys}, " if group_by else ""
        sql = (
            f"SELECT {select_keys}count(*) AS trajectories, "
            "count(*) FILTER (WHERE status = 'completed') AS completed, "
            "count(*) FILTER (WHERE status = 'failed') AS failed, "
            "avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END) FILTER (WHERE status = 'completed') AS pass_rate, "
            "avg(reward) AS mean_reward, stddev_pop(reward) AS reward_std, "
            "avg(completion_tokens) AS mean_completion_tokens "
            f"FROM ({self._latest_sql(columns + ', written_at')}){where}"
        )
        if group_by:
            sql += f" GROUP BY {keys} ORDER BY {keys}"
        names = [
            *group_by,
            "trajectories",
            "completed",
            "failed",
            "pass_rate",
            "mean_reward",
            "reward_std",
            "mean_completion_tokens",
        ]
        return [dict(zip(names, row, strict=True)) for row in self._execute(sql, params)]

    def export_rows(self, query: TrajectoryQuery, path: Path, columns: Sequence[str]) -> int:
        """Write selected summary columns of matching rows to a new Parquet file."""
        bad = set(columns) - set(SUMMARY_COLUMNS)
        if bad:
            raise StoreError(f"unknown columns {sorted(bad)}")
        rows = self.query(query)
        table = pa.Table.from_pylist([{c: getattr(r, c) for c in columns} for r in rows])
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression=self.compression)
        return len(rows)
