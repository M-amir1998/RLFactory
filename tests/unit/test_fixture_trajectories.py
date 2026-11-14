"""Committed trajectory fixtures: synthetic content only, and readable by the current schema.

Every ``tests/fixtures/trajectories/*.jsonl`` file was produced by
``scripts/make_fixture_trajectories.py``. Loading each committed version proves that stored
trajectories stay readable when the schema evolves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rlfactory.domain.trajectory import FailureOrigin, Trajectory, TrajectoryStatus
from rlfactory.trajectories.parquet_store import LocalParquetTrajectoryStore
from rlfactory.trajectories.store import TrajectoryQuery

pytestmark = pytest.mark.store

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "trajectories"
FIXTURES = sorted(FIXTURE_DIR.glob("*.jsonl"))
# Machine- or person-specific content must never appear in a public fixture.
PRIVATE = re.compile(r"\b[A-Za-z]:\\\\|/home/|/Users/|[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
DOMAINS = {"arithmetic", "structured_output", "python", "transformation", "tool_use"}


def _load(path: Path) -> list[Trajectory]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Trajectory.model_validate_json(line) for line in lines if line.strip()]


def test_fixtures_are_committed() -> None:
    assert FIXTURES, f"no fixtures in {FIXTURE_DIR}"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_is_readable_and_covers_every_outcome(path: Path) -> None:
    trajectories = _load(path)
    assert len({t.trajectory_id for t in trajectories}) == len(trajectories)
    assert all(Trajectory.model_validate_json(t.model_dump_json()) == t for t in trajectories)
    assert {t.lineage.environment.name for t in trajectories} == DOMAINS
    assert {t.status for t in trajectories} == {TrajectoryStatus.COMPLETED, TrajectoryStatus.FAILED}
    origins = {t.failure.origin for t in trajectories if t.failure is not None}
    assert origins == {FailureOrigin.MODEL, FailureOrigin.INFRASTRUCTURE}
    assert any(t.passed for t in trajectories) and any(len(t.actions) > 1 for t in trajectories)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_semantics_hold(path: Path) -> None:
    for trajectory in _load(path):
        infrastructure = (
            trajectory.failure is not None
            and trajectory.failure.origin is FailureOrigin.INFRASTRUCTURE
        )
        if infrastructure:
            assert trajectory.status is TrajectoryStatus.FAILED
            assert trajectory.verification is None and not trajectory.is_trainable
        else:
            assert trajectory.verification is not None and trajectory.rewards is not None
            assert trajectory.is_trainable
        if trajectory.passed:
            assert trajectory.verification is not None and trajectory.verification.score > 0


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_contains_synthetic_content_only(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not PRIVATE.search(text), PRIVATE.search(text)
    for trajectory in _load(path):
        assert trajectory.lineage.code.git_sha == "0" * 40
        assert trajectory.lineage.model == "rlfactory/mock-policy"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_roundtrips_through_the_parquet_store(path: Path, tmp_path: Path) -> None:
    trajectories = _load(path)
    store = LocalParquetTrajectoryStore(tmp_path)
    store.write(trajectories)
    store.flush()
    summaries = store.query(TrajectoryQuery())
    assert len(summaries) == len(trajectories)
    loaded = {t.trajectory_id: t for t in store.load(TrajectoryQuery())}
    assert loaded == {t.trajectory_id: t for t in trajectories}
    completed = {t.trajectory_id for t in trajectories if t.status is TrajectoryStatus.COMPLETED}
    assert store.completed_ids() == completed
