"""Integration: Task -> MockPolicy -> Environment -> Verifier -> Reward -> Trajectory store,
then stored trajectories -> SFT / DPO / GRPO builders -> dataset validation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rlfactory.datasets.adapters import to_trl_dpo, to_trl_grpo, to_trl_sft
from rlfactory.datasets.builders import (
    PreferenceDatasetBuilder,
    RLPromptDatasetBuilder,
    SFTDatasetBuilder,
)
from rlfactory.datasets.io import write_dataset
from rlfactory.datasets.validation import (
    validate_preference,
    validate_rl,
    validate_sft,
    validate_trl_file,
)
from rlfactory.domain.common import CodeVersion
from rlfactory.domain.task import TaskDomain
from rlfactory.policies.mock import MockPolicy, MockPolicyProfile
from rlfactory.rollouts.config import RetryPolicy, RolloutConfig
from rlfactory.rollouts.service import RolloutDependencies, run_rollouts
from rlfactory.sandbox.base import ResourceLimits
from rlfactory.sandbox.subprocess_sandbox import SubprocessSandbox
from rlfactory.tasks.contamination import ContaminationFilter
from rlfactory.tasks.sampling import task_stats
from rlfactory.trajectories.parquet_store import LocalParquetTrajectoryStore
from rlfactory.trajectories.store import TrajectoryQuery
from tests.helpers import tasks_for

pytestmark = pytest.mark.sandbox


@pytest.fixture(scope="module")
def rollout_store(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[LocalParquetTrajectoryStore, list]:  # type: ignore[type-arg]
    tasks = [task for domain in TaskDomain for task in tasks_for(domain)]
    store = LocalParquetTrajectoryStore(tmp_path_factory.mktemp("store"), batch_size=16)
    deps = RolloutDependencies.default(
        sandbox=SubprocessSandbox(), python_limits=ResourceLimits(wall_time_s=5)
    )
    policy = MockPolicy(
        tasks,
        MockPolicyProfile(
            correct_probability=0.6, timeout_probability=0.05, error_probability=0.05
        ),
        seed=4,
    )
    summary = asyncio.run(
        run_rollouts(
            run_id="integration",
            tasks=tasks,
            policy=policy,
            config=RolloutConfig(
                samples_per_task=4, concurrency=6, retry=RetryPolicy(backoff_initial_s=0.0)
            ),
            store=store,
            deps=deps,
        )
    )
    assert summary.completed + summary.failed == len(tasks) * 4
    return store, tasks


def test_rollouts_are_persisted_with_lineage_and_failure_taxonomy(rollout_store) -> None:  # type: ignore[no-untyped-def]
    store, tasks = rollout_store
    rows = store.query(TrajectoryQuery(run_id="integration"))
    assert len(rows) == len(tasks) * 4
    assert {r.domain for r in rows} == {d.value for d in TaskDomain}
    assert all(r.verifier_version and r.environment_version and r.reward_version for r in rows)
    completed = [r for r in rows if r.status == "completed"]
    assert all(r.passed is not None and r.reward is not None for r in completed)
    model_failures = {r.failure_kind for r in completed if r.failure_kind}
    assert model_failures <= {"invalid_output", "sandbox_timeout"}
    python_rows = [r for r in completed if r.domain == "python"]
    assert python_rows and any(r.passed for r in python_rows)
    by_domain = {row["domain"]: row for row in store.aggregate(["domain"], TrajectoryQuery())}
    assert set(by_domain) == {d.value for d in TaskDomain}


def test_stored_trajectories_build_valid_datasets(rollout_store, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store, tasks = rollout_store
    trajectories = store.load(TrajectoryQuery(run_id="integration"))
    index = {t.task_id: t for t in tasks}
    held_out = ContaminationFilter.from_tasks(tasks[:2])
    sft = SFTDatasetBuilder(contamination=held_out, tasks=index).build(trajectories)
    dpo = PreferenceDatasetBuilder(contamination=held_out, tasks=index).build(trajectories)
    grpo = RLPromptDatasetBuilder(
        environments=__import__(
            "rlfactory.environments.registry", fromlist=["x"]
        ).default_environment_registry(),
        contamination=held_out,
    ).build(tasks, task_stats(trajectories))
    assert sft.records and dpo.records and grpo.records
    excluded_ids = {t.task_id for t in tasks[:2]}
    for records in (sft.records, dpo.records):
        assert not {r.lineage.task_id for r in records} & excluded_ids
    assert not {r.task.task_id for r in grpo.records} & excluded_ids
    assert validate_sft(sft.records, held_out).ok
    assert validate_preference(dpo.records, held_out).ok
    assert validate_rl(grpo.records, held_out).ok
    code = CodeVersion(rlfactory_version="t")
    for kind, records, adapter in (
        ("sft", sft.records, to_trl_sft),
        ("dpo", dpo.records, to_trl_dpo),
        ("grpo", grpo.records, to_trl_grpo),
    ):
        write_dataset(
            tmp_path / kind,
            name=kind,
            kind=kind,
            records=records,
            rows=[adapter(r) for r in records],  # type: ignore[arg-type]
            builder=kind,
            builder_config=None,
            build_report={},
            validation={"ok": True},
            source_run_ids=["integration"],
            contamination_counts={},
            code=code,
        )
        assert validate_trl_file(tmp_path / kind / "train.jsonl", kind).ok  # type: ignore[arg-type]
