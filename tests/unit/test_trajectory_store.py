from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from rlfactory.domain.trajectory import FailureKind, FailureRecord, TrajectoryStatus
from rlfactory.errors import StoreError
from rlfactory.trajectories.jsonl_store import JSONLTrajectoryStore
from rlfactory.trajectories.memory_store import MemoryTrajectoryStore
from rlfactory.trajectories.parquet_store import LocalParquetTrajectoryStore
from rlfactory.trajectories.store import RetentionPolicy, TrajectoryQuery, should_retain
from tests.helpers import make_task, play

pytestmark = pytest.mark.store


def _trajectories():  # type: ignore[no-untyped-def]
    arith = make_task("arithmetic.integer_chain", seed=1)
    text = make_task("transformation.reverse_words", seed=1)
    return [
        play(arith, [f"<answer>{arith.payload.answer}</answer>"], sample_index=0),  # type: ignore[union-attr]
        play(arith, ["<answer>-77</answer>"], sample_index=1),
        play(text, [f"<answer>{text.payload.expected_output}</answer>"], sample_index=0),  # type: ignore[union-attr]
    ]


def _failed(base):  # type: ignore[no-untyped-def]
    return base.model_copy(
        update={
            "status": TrajectoryStatus.FAILED,
            "verification": None,
            "rewards": None,
            "failure": FailureRecord(kind=FailureKind.POLICY_TIMEOUT, retryable=True),
        }
    )


@pytest.fixture(params=["parquet", "jsonl", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path):  # type: ignore[no-untyped-def]
    if request.param == "parquet":
        return LocalParquetTrajectoryStore(tmp_path / "p", batch_size=2)
    if request.param == "jsonl":
        return JSONLTrajectoryStore(tmp_path / "j")
    return MemoryTrajectoryStore()


def test_empty_store_queries(store) -> None:  # type: ignore[no-untyped-def]
    assert store.query(TrajectoryQuery()) == []
    assert store.load(TrajectoryQuery()) == []
    assert store.completed_ids() == set()


def test_write_query_load_roundtrip(store) -> None:  # type: ignore[no-untyped-def]
    trajectories = _trajectories()
    store.write(trajectories)
    store.flush()
    assert store.completed_ids("test-run") == {t.trajectory_id for t in trajectories}
    assert store.completed_ids("other-run") == set()
    loaded = {t.trajectory_id: t for t in store.load(TrajectoryQuery())}
    for trajectory in trajectories:
        assert loaded[trajectory.trajectory_id] == trajectory
    assert len(store.query(TrajectoryQuery(domain="arithmetic"))) == 2
    assert len(store.query(TrajectoryQuery(passed=True))) == 2
    assert len(store.query(TrajectoryQuery(min_reward=0.5))) == 2
    assert len(store.query(TrajectoryQuery(max_reward=0.5))) == 1
    assert len(store.query(TrajectoryQuery(task_ids=(trajectories[2].task_id,)))) == 1
    assert (
        len(store.query(TrajectoryQuery(environment="transformation", verifier="exact_match"))) == 1
    )
    assert len(store.query(TrajectoryQuery(limit=1))) == 1


def test_latest_record_wins_for_resume(store) -> None:  # type: ignore[no-untyped-def]
    good = _trajectories()[0]
    store.write([_failed(good)])
    store.flush()
    assert store.completed_ids() == set()
    store.write([good])
    store.flush()
    assert store.completed_ids() == {good.trajectory_id}
    rows = store.query(TrajectoryQuery())
    assert len(rows) == 1 and rows[0].status == "completed"


def test_pending_trajectories_are_rejected(store) -> None:  # type: ignore[no-untyped-def]
    pending = _trajectories()[0].model_copy(update={"status": TrajectoryStatus.PENDING})
    with pytest.raises(StoreError, match="pending"):
        store.write([pending])


def test_aggregate(store) -> None:  # type: ignore[no-untyped-def]
    store.write(_trajectories())
    store.flush()
    rows = {r["domain"]: r for r in store.aggregate(["domain"], TrajectoryQuery())}
    assert rows["arithmetic"]["trajectories"] == 2
    assert rows["arithmetic"]["pass_rate"] == pytest.approx(0.5)
    assert rows["transformation"]["mean_reward"] == pytest.approx(1.0)


def test_parquet_layout_compression_and_summary_columns(tmp_path: Path) -> None:
    store = LocalParquetTrajectoryStore(tmp_path, batch_size=100)
    store.write(_trajectories())
    assert store.shards() == []
    store.flush()
    shards = store.shards()
    assert len(shards) == 1 and not list(tmp_path.glob("*.tmp"))
    metadata = pq.ParquetFile(shards[0]).metadata
    assert metadata.row_group(0).column(0).compression == "ZSTD"
    row = store.query(TrajectoryQuery())[0]
    assert row.git_sha is None and row.rlfactory_version == "test"
    assert row.started_at.tzinfo is not None
    with pytest.raises(StoreError, match="cannot group"):
        store.aggregate(["payload"], TrajectoryQuery())


@pytest.mark.parametrize(
    ("policy", "kept"),
    [
        (RetentionPolicy.ALL, 3),
        (RetentionPolicy.SUCCESSFUL_ONLY, 2),
        (RetentionPolicy.FAILED_ONLY, 1),
    ],
)
def test_retention_keeps_summaries_but_drops_payloads(
    tmp_path: Path, policy: RetentionPolicy, kept: int
) -> None:
    store = LocalParquetTrajectoryStore(tmp_path, retention=policy)
    store.write(_trajectories())
    store.flush()
    assert len(store.query(TrajectoryQuery())) == 3
    assert len(store.load(TrajectoryQuery())) == kept
    assert sum(r.payload_retained for r in store.query(TrajectoryQuery())) == kept


def test_sampled_retention_is_deterministic() -> None:
    trajectory = _trajectories()[0]
    first = should_retain(trajectory, RetentionPolicy.SAMPLED, 0.5)
    assert all(should_retain(trajectory, RetentionPolicy.SAMPLED, 0.5) == first for _ in range(3))
    assert should_retain(trajectory, RetentionPolicy.SAMPLED, 1.0)
    assert not should_retain(trajectory, RetentionPolicy.SAMPLED, 0.0)


def test_export_rows(tmp_path: Path) -> None:
    store = LocalParquetTrajectoryStore(tmp_path / "s")
    store.write(_trajectories())
    count = store.export_rows(
        TrajectoryQuery(), tmp_path / "out.parquet", ["trajectory_id", "reward"]
    )
    assert count == 3 and pq.read_table(tmp_path / "out.parquet").column_names == [
        "trajectory_id",
        "reward",
    ]
    with pytest.raises(StoreError, match="unknown"):
        store.export_rows(TrajectoryQuery(), tmp_path / "x.parquet", ["payload"])
