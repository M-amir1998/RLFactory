"""Regenerate the committed trajectory fixture (synthetic content only).

Tasks come from RLFactory's own templates and answers from the mock policy (with injected
timeouts and server errors); Python answers run in the subprocess sandbox. No model, dataset,
network access or user data is involved, so the fixture is safe to publish.

Regenerate only when the trajectory schema changes on purpose, under a new file name, and keep the
old file: ``tests/unit/test_fixture_trajectories.py`` loads every committed version to prove that
stored trajectories stay readable.

    uv run python scripts/make_fixture_trajectories.py tests/fixtures/trajectories/synthetic-v1.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from random import Random

from rlfactory.domain.task import Task
from rlfactory.policies.mock import MockPolicy, MockPolicyProfile
from rlfactory.rollouts.config import RetryPolicy, RolloutConfig
from rlfactory.rollouts.service import RolloutDependencies, run_rollouts
from rlfactory.sandbox.base import ResourceLimits
from rlfactory.tasks.generators.catalog import default_template_registry
from rlfactory.trajectories.memory_store import MemoryTrajectoryStore
from rlfactory.trajectories.store import TrajectoryQuery

# A placeholder instead of the real commit keeps regenerated fixtures free of build metadata.
FIXTURE_GIT_SHA = "0" * 40
FAMILIES = (
    "arithmetic.integer_chain",
    "arithmetic.fraction_sum",
    "structured_output.filter_sort",
    "python.sum_of_evens",
    "python.running_max",
    "transformation.reverse_words",
    "tool_use.order_total",
)


def fixture_tasks() -> list[Task]:
    templates = default_template_registry()
    return [
        templates.get(family).make_task(templates.get(family).sample(Random(index), 2))
        for index, family in enumerate(FAMILIES)
    ]


async def generate(output: Path) -> int:
    os.environ["RLFACTORY_GIT_SHA"] = FIXTURE_GIT_SHA
    tasks = fixture_tasks()
    policy = MockPolicy(
        tasks,
        MockPolicyProfile(
            correct_probability=0.5,
            malformed_probability=0.15,
            timeout_probability=0.1,
            error_probability=0.05,
        ),
        seed=7,
        policy_id="mock-fixture",
    )
    store = MemoryTrajectoryStore()
    config = RolloutConfig(
        samples_per_task=2,
        concurrency=2,
        seed=7,
        reward="rlvr_default",
        retry=RetryPolicy(max_attempts=1),
    )
    await run_rollouts(
        run_id="fixture-v1",
        tasks=tasks,
        policy=policy,
        config=config,
        store=store,
        deps=RolloutDependencies.default(python_limits=ResourceLimits(wall_time_s=10)),
    )
    trajectories = sorted(store.load(TrajectoryQuery()), key=lambda t: t.trajectory_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.model_dump_json() + "\n")
    return len(trajectories)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = asyncio.run(generate(args.output))
    print(f"wrote {count} trajectories to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
