"""Failure-driven data flywheel.

Each iteration:

1. evaluate the current policy on a **development** split and classify failures;
2. mutate failed tasks deterministically (numbers, constraints, names, lengths, formats, edges);
3. validate, deduplicate and decontaminate mutants against the held-out evaluation suite
   (mutants of evaluation tasks can never enter training);
4. roll out the current policy on the new tasks and keep verified-correct samples
   (rejection-sampling SFT data);
5. train a new checkpoint, re-evaluate on the held-out suite and apply regression gates.

The loop is backend-agnostic: training and checkpoint->policy resolution are injected.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.datasets.adapters import to_trl_sft
from rlfactory.datasets.builders import SFTBuilderConfig, SFTDatasetBuilder
from rlfactory.datasets.io import DatasetCard, write_dataset
from rlfactory.datasets.validation import validate_sft
from rlfactory.domain.model import CheckpointReference
from rlfactory.domain.task import Task
from rlfactory.evaluation.evaluator import EvaluationConfig, Evaluator
from rlfactory.evaluation.failures import analyze_failures
from rlfactory.evaluation.gates import GateConfig, evaluate_gates
from rlfactory.policies.base import Policy
from rlfactory.rollouts.config import RolloutConfig
from rlfactory.rollouts.service import RolloutDependencies, run_rollouts
from rlfactory.runtime.git import detect_code_version
from rlfactory.tasks.contamination import ContaminationFilter
from rlfactory.tasks.mutation import TaskMutator
from rlfactory.tasks.synthesis import SynthesisPipeline
from rlfactory.trajectories.parquet_store import LocalParquetTrajectoryStore
from rlfactory.trajectories.store import TrajectoryQuery


class FlywheelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    iterations: int = Field(default=2, ge=1, le=20)
    mutations_per_failure: int = Field(default=3, ge=1, le=20)
    max_failure_tasks: int = Field(default=30, ge=1)
    samples_per_task: int = Field(default=4, ge=1)
    sft: SFTBuilderConfig = SFTBuilderConfig()


class FlywheelIteration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    iteration: int
    failures_observed: int
    failure_tasks: int
    failure_categories: dict[str, int]
    candidates: int
    tasks_added: int
    rejected_invalid: int
    rejected_contaminated: int
    train_examples: int
    checkpoint: str | None
    eval_accuracy: float | None
    eval_accuracy_ci95: tuple[float, float] | None
    gate_passed: bool | None
    simulated: bool


TrainFn = Callable[[int, Path, DatasetCard], Awaitable[CheckpointReference]]
PolicyFn = Callable[[CheckpointReference | None, Sequence[Task]], Policy]


class FailureDrivenFlywheel:
    def __init__(
        self,
        *,
        config: FlywheelConfig,
        deps: RolloutDependencies,
        mutator: TaskMutator,
        pipeline: SynthesisPipeline,
        evaluator: Evaluator,
        evaluation: EvaluationConfig,
        gates: GateConfig,
        train: TrainFn,
        policy_for: PolicyFn,
        workdir: Path,
        run_id: str,
    ) -> None:
        self.config = config
        self.deps = deps
        self.mutator = mutator
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.evaluation = evaluation
        self.gates = gates
        self.train = train
        self.policy_for = policy_for
        self.workdir = workdir
        self.run_id = run_id

    async def run(
        self, dev_tasks: Sequence[Task], eval_tasks: Sequence[Task]
    ) -> list[FlywheelIteration]:
        known: dict[str, Task] = {t.task_id: t for t in (*dev_tasks, *eval_tasks)}
        checkpoint: CheckpointReference | None = None
        policy = self.policy_for(None, list(known.values()))
        previous, _, _ = await self.evaluator.evaluate(
            label="flywheel-iter0",
            suite="flywheel-eval",
            policy=policy,
            tasks=eval_tasks,
            config=self.evaluation,
            run_id=self.run_id,
        )
        records = [
            FlywheelIteration(
                iteration=0,
                failures_observed=0,
                failure_tasks=0,
                failure_categories={},
                candidates=0,
                tasks_added=0,
                rejected_invalid=0,
                rejected_contaminated=0,
                train_examples=0,
                checkpoint=None,
                eval_accuracy=previous.overall.verified_accuracy,
                eval_accuracy_ci95=previous.overall.accuracy_ci95,
                gate_passed=None,
                simulated=policy.simulated,
            )
        ]
        contamination = ContaminationFilter.from_tasks(eval_tasks)
        for iteration in range(1, self.config.iterations + 1):
            directory = self.workdir / f"iter-{iteration:02d}"
            _, dev_trajectories, _ = await self.evaluator.evaluate(
                label=f"flywheel-dev-{iteration}",
                suite="flywheel-dev",
                policy=policy,
                tasks=dev_tasks,
                config=self.evaluation,
                run_id=self.run_id,
            )
            analysis = analyze_failures(dev_trajectories)
            failed_ids = analysis.failed_task_ids()[: self.config.max_failure_tasks]
            candidates = [
                mutant
                for task_id in failed_ids
                for mutant in self.mutator.mutate_many(
                    known[task_id], self.config.mutations_per_failure
                )
            ]
            accepted, report = self.pipeline.filter(candidates)
            known.update({t.task_id: t for t in accepted})
            policy = self.policy_for(checkpoint, list(known.values()))
            examples = 0
            card: DatasetCard | None = None
            if accepted:
                store = LocalParquetTrajectoryStore(directory / "rollouts")
                rollout_run = f"{self.run_id}-flywheel-{iteration}"
                await run_rollouts(
                    run_id=rollout_run,
                    tasks=accepted,
                    policy=policy,
                    config=RolloutConfig(
                        samples_per_task=self.config.samples_per_task, seed=iteration
                    ),
                    store=store,
                    deps=self.deps,
                )
                result = SFTDatasetBuilder(
                    self.config.sft, contamination=contamination, tasks=known
                ).build(store.load(TrajectoryQuery(run_id=rollout_run)))
                validation = validate_sft(result.records, contamination)
                examples = len(result.records)
                if examples and validation.ok:
                    card = write_dataset(
                        directory / "dataset",
                        name=f"flywheel-{iteration}",
                        kind="sft",
                        records=result.records,
                        rows=[to_trl_sft(r) for r in result.records],
                        builder="SFTDatasetBuilder",
                        builder_config=self.config.sft,
                        build_report=result.report(),
                        validation=validation.to_dict(),
                        source_run_ids=[rollout_run],
                        contamination_counts={
                            k: len(v) for k, v in contamination.to_dict().items()
                        },
                        code=detect_code_version(),
                    )
            if card is not None:
                checkpoint = await self.train(iteration, directory / "dataset", card)
                policy = self.policy_for(checkpoint, list(known.values()))
            evaluation, _, _ = await self.evaluator.evaluate(
                label=f"flywheel-iter{iteration}",
                suite="flywheel-eval",
                policy=policy,
                tasks=eval_tasks,
                config=self.evaluation,
                run_id=self.run_id,
            )
            gate = evaluate_gates(previous, evaluation, self.gates)
            records.append(
                FlywheelIteration(
                    iteration=iteration,
                    failures_observed=len(analysis.observations),
                    failure_tasks=len(failed_ids),
                    failure_categories=analysis.counts,
                    candidates=len(candidates),
                    tasks_added=len(accepted),
                    rejected_invalid=len(report.invalid),
                    rejected_contaminated=len(report.contaminated),
                    train_examples=examples,
                    checkpoint=checkpoint.checkpoint_id if checkpoint else None,
                    eval_accuracy=evaluation.overall.verified_accuracy,
                    eval_accuracy_ci95=evaluation.overall.accuracy_ci95,
                    gate_passed=gate.passed,
                    simulated=policy.simulated,
                )
            )
            previous = evaluation
        return records
