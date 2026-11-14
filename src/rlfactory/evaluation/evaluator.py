"""Unified evaluator: the same suite, verifiers and metrics for base, SFT, DPO and GRPO policies."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.domain.common import stable_hash
from rlfactory.domain.model import GenerationConfig
from rlfactory.domain.result import EvaluationResult
from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import Trajectory
from rlfactory.evaluation.metrics import summarize_by_domain, summarize_trajectories
from rlfactory.policies.base import Policy
from rlfactory.rollouts.config import RetryPolicy, RolloutConfig
from rlfactory.rollouts.orchestrator import ProgressReporter, RolloutSummary
from rlfactory.rollouts.service import RolloutDependencies, run_rollouts
from rlfactory.runtime.git import detect_code_version
from rlfactory.trajectories.memory_store import MemoryTrajectoryStore
from rlfactory.trajectories.parquet_store import LocalParquetTrajectoryStore
from rlfactory.trajectories.store import TrajectoryQuery, TrajectoryStore


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    samples_per_task: int = Field(default=1, ge=1)
    generation: GenerationConfig = GenerationConfig(temperature=0.0, max_tokens=512)
    pass_at_k: tuple[int, ...] = (1,)
    seed: int = 0
    concurrency: int = Field(default=4, ge=1)
    reward: str = "binary"
    verifier_overrides: dict[TaskDomain, str] = Field(default_factory=dict)
    retry: RetryPolicy = RetryPolicy()


class Evaluator:
    def __init__(self, deps: RolloutDependencies) -> None:
        self.deps = deps

    async def evaluate(
        self,
        *,
        label: str,
        suite: str,
        policy: Policy,
        tasks: Sequence[Task],
        config: EvaluationConfig,
        run_id: str,
        store_root: Path | None = None,
        progress: ProgressReporter | None = None,
    ) -> tuple[EvaluationResult, list[Trajectory], RolloutSummary]:
        store: TrajectoryStore = (
            LocalParquetTrajectoryStore(store_root)
            if store_root is not None
            else MemoryTrajectoryStore()
        )
        rollout_config = RolloutConfig(
            samples_per_task=config.samples_per_task,
            concurrency=config.concurrency,
            generation=config.generation,
            seed=config.seed,
            reward=config.reward,
            verifier_overrides=config.verifier_overrides,
            retry=config.retry,
        )
        eval_run_id = f"{run_id}-eval-{label}"
        summary = await run_rollouts(
            run_id=eval_run_id,
            tasks=tasks,
            policy=policy,
            config=rollout_config,
            store=store,
            deps=self.deps,
            progress=progress,
        )
        trajectories = store.load(TrajectoryQuery(run_id=eval_run_id))
        temperature = config.generation.temperature
        result = EvaluationResult(
            evaluation_id=f"eval-{stable_hash(eval_run_id, policy.policy_id, policy.checkpoint, length=12)}",
            label=label,
            policy_id=policy.policy_id,
            checkpoint=policy.checkpoint,
            suite=suite,
            num_tasks=len(tasks),
            samples_per_task=config.samples_per_task,
            temperature=temperature,
            overall=summarize_trajectories(
                trajectories, ks=config.pass_at_k, temperature=temperature
            ),
            by_domain=summarize_by_domain(
                trajectories, ks=config.pass_at_k, temperature=temperature
            ),
            simulated=policy.simulated,
            created_at=datetime.now(UTC),
            code=detect_code_version(),
        )
        return result, trajectories, summary
