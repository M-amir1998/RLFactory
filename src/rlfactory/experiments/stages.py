"""Stage implementations. Each stage reads its inputs from the run directory and writes outputs back."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rlfactory.config.loader import read_json_model, write_json
from rlfactory.datasets.adapters import (
    DatasetKind,
    to_trl_dpo,
    to_trl_grpo,
    to_trl_reward,
    to_trl_sft,
)
from rlfactory.datasets.builders import (
    PreferenceDatasetBuilder,
    RLPromptDatasetBuilder,
    SFTDatasetBuilder,
)
from rlfactory.datasets.io import DatasetCard, read_card, write_dataset
from rlfactory.datasets.reward_model import RewardModelDatasetBuilder
from rlfactory.datasets.validation import validate_preference, validate_rl, validate_sft
from rlfactory.domain.model import CheckpointReference
from rlfactory.domain.result import EvaluationResult
from rlfactory.domain.task import Task
from rlfactory.domain.training import (
    LaunchMode,
    SFTParams,
    TrainingMethod,
    TrainingRun,
    TrainingState,
)
from rlfactory.environments.base import EnvironmentContext
from rlfactory.errors import ExperimentError
from rlfactory.evaluation.coverage import verifier_coverage
from rlfactory.evaluation.evaluator import Evaluator
from rlfactory.evaluation.failures import analyze_failures, write_failures_parquet
from rlfactory.evaluation.gates import evaluate_gates
from rlfactory.evaluation.promotion import PromotionRegistry
from rlfactory.experiments.context import RunContext, StageBlockedError, StageOutput
from rlfactory.flywheel.loop import FailureDrivenFlywheel
from rlfactory.policies.base import Policy
from rlfactory.policies.factory import MockPolicySpec, build_policy
from rlfactory.policies.mock import MockPolicy, MockPolicyProfile
from rlfactory.rewards.analysis import RewardHackingAnalyzer, calibrate_rewards
from rlfactory.rollouts.service import run_rollouts
from rlfactory.runtime.disk import estimate_rollout_bytes
from rlfactory.tasks.contamination import ContaminationFilter, deduplicate
from rlfactory.tasks.io import save_tasks
from rlfactory.tasks.mutation import TaskMutator
from rlfactory.tasks.registry import TaskRegistry
from rlfactory.tasks.sampling import task_stats
from rlfactory.tasks.splits import Split, split_tasks
from rlfactory.tasks.synthesis import SynthesisPipeline
from rlfactory.tasks.validation import TaskValidator
from rlfactory.telemetry.metrics import RolloutMetrics
from rlfactory.training.base import SUMMARY_FILE, save_prepared
from rlfactory.training.mock.backend import (
    MockTrainingBackend,
    MockTrainingSettings,
    load_mock_checkpoint,
)
from rlfactory.training.trl.backend import TRLTrainingBackend
from rlfactory.trajectories.store import TrajectoryQuery

StageFn = Callable[[RunContext], Awaitable[StageOutput]]


# ------------------------------------------------------------------------------------- tasks


def _validate_all(
    ctx: RunContext, tasks: Sequence[Task]
) -> tuple[list[Task], dict[str, list[str]]]:
    deps = ctx.deps
    validator = TaskValidator(
        deps.environments,
        deps.verifiers,
        EnvironmentContext(
            sandbox=deps.environment_context.sandbox, python_limits=ctx.config.sandbox.limits
        ),
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(validator.validate, tasks))
    valid = [task for task, outcome in zip(tasks, outcomes, strict=True) if outcome.valid]
    invalid = {o.task_id: list(o.reasons) for o in outcomes if not o.valid}
    return valid, invalid


async def stage_tasks(ctx: RunContext) -> StageOutput:
    section = ctx.config.tasks
    registry = TaskRegistry(ctx.settings.config_dir, ctx.templates)

    def load(names: Sequence[str]) -> list[Task]:
        tasks: list[Task] = []
        for name in names:
            suite = list(registry.load(name).tasks)
            tasks.extend(
                suite[: section.max_tasks_per_suite] if section.max_tasks_per_suite else suite
            )
        return deduplicate(tasks)[0]

    train, evaluation = load(section.train_suites), load(section.eval_suites)
    invalid: dict[str, list[str]] = {}
    if section.validate_tasks:
        train, invalid_train = _validate_all(ctx, train)
        evaluation, invalid_eval = _validate_all(ctx, evaluation)
        invalid = {**invalid_train, **invalid_eval}
    contamination = ContaminationFilter.from_tasks(
        evaluation, exclude_families=section.exclude_eval_families
    )
    clean = [t for t in train if contamination.reason(t) is None]
    removed = Counter(reason for t in train if (reason := contamination.reason(t)) is not None)
    validation: list[Task] = []
    if section.split is not None:
        parts = split_tasks(clean, section.split)
        clean, validation = parts[Split.TRAIN], parts[Split.VALIDATION]
    for name, tasks in (("train", clean), ("validation", validation), ("evaluation", evaluation)):
        save_tasks(tasks, ctx.path("tasks", f"{name}.jsonl"))
    write_json(contamination.to_dict(), ctx.path("tasks", "contamination.json"))
    coverage = verifier_coverage(
        [*clean, *evaluation], ctx.deps.verifiers, ctx.config.rollouts.verifier_overrides
    )
    summary: dict[str, Any] = {
        "train": len(clean),
        "validation": len(validation),
        "evaluation": len(evaluation),
        "invalid": len(invalid),
        "contamination_removed": dict(removed),
        "train_by_domain": dict(Counter(t.domain.value for t in clean)),
        "evaluation_by_domain": dict(Counter(t.domain.value for t in evaluation)),
        "verifier_coverage": coverage.counts,
        "suites": {
            name: registry.load(name).ref().model_dump()
            for name in (*section.train_suites, *section.eval_suites)
        },
    }
    write_json({**summary, "invalid_reasons": invalid}, ctx.path("tasks", "summary.json"))
    return StageOutput(
        outputs={"tasks_dir": str(ctx.path("tasks"))}, summary=json.loads(json.dumps(summary))
    )


# ---------------------------------------------------------------------------------- rollouts


async def stage_rollouts(ctx: RunContext) -> StageOutput:
    tasks = ctx.tasks("train")
    config = ctx.config.rollouts
    ctx.disk_guard.ensure(
        ctx.run_dir, estimate_rollout_bytes(len(tasks) * config.samples_per_task), "rollouts"
    )
    policy = build_policy(ctx.config.policy, tasks)
    metrics = RolloutMetrics()
    try:
        summary = await run_rollouts(
            run_id=ctx.run_id,
            tasks=tasks,
            policy=policy,
            config=config,
            store=ctx.store,
            deps=ctx.deps,
            progress=ctx.progress("rollouts"),
            metrics=metrics,
        )
    finally:
        await policy.aclose()
    write_json(summary, ctx.path("rollout_summary.json"))
    ctx.path("metrics.prom").write_text(metrics.registry.to_prometheus(), encoding="utf-8")
    return StageOutput(
        outputs={
            "store": str(ctx.store.root),
            "summary": str(ctx.path("rollout_summary.json")),
            "metrics": str(ctx.path("metrics.prom")),
        },
        summary={
            "completed": summary.completed,
            "failed": summary.failed,
            "pass_rate": summary.pass_rate,
            "mean_reward": summary.mean_reward,
            "retries": summary.retries,
            "trajectories_per_s": summary.trajectories_per_s,
        },
    )


# ---------------------------------------------------------------------------------- datasets


def _write(
    ctx: RunContext,
    kind: DatasetKind,
    records: Sequence[Any],
    rows: list[dict[str, Any]],
    builder: str,
    builder_config: Any,
    report: dict[str, Any],
    validation: dict[str, Any],
) -> StageOutput:
    if not validation.get("ok", False):
        raise ExperimentError(f"{kind} dataset failed validation: {validation.get('errors')}")
    directory = ctx.path("datasets", kind)
    contamination = ctx.contamination()
    card = write_dataset(
        directory,
        name=f"{ctx.run_id}-{kind}",
        kind=kind,
        records=records,
        rows=rows,
        builder=builder,
        builder_config=builder_config,
        build_report=report,
        validation=validation,
        source_run_ids=[ctx.run_id],
        contamination_counts={k: len(v) for k, v in contamination.to_dict().items()},
        code=ctx.code,
        disk_guard=ctx.disk_guard,
    )
    return StageOutput(
        outputs={"dataset_dir": str(directory)},
        summary={"records": card.num_records, "fingerprint": card.fingerprint, **report},
    )


def _trajectories(ctx: RunContext) -> list[Any]:
    return ctx.store.load(TrajectoryQuery(run_id=ctx.run_id))


async def stage_build_sft(ctx: RunContext) -> StageOutput:
    config = ctx.config.datasets.sft
    result = SFTDatasetBuilder(
        config, contamination=ctx.contamination(), tasks=ctx.task_index()
    ).build(_trajectories(ctx))
    report = validate_sft(result.records, ctx.contamination())
    return _write(
        ctx,
        "sft",
        result.records,
        [to_trl_sft(r) for r in result.records],
        "SFTDatasetBuilder",
        config,
        result.report(),
        report.to_dict(),
    )


async def stage_build_dpo(ctx: RunContext) -> StageOutput:
    config = ctx.config.datasets.dpo
    result = PreferenceDatasetBuilder(
        config, contamination=ctx.contamination(), tasks=ctx.task_index()
    ).build(_trajectories(ctx))
    report = validate_preference(result.records, ctx.contamination())
    output = _write(
        ctx,
        "dpo",
        result.records,
        [to_trl_dpo(r) for r in result.records],
        "PreferenceDatasetBuilder",
        config,
        result.report(),
        report.to_dict(),
    )
    if (
        ctx.config.training is not None
        and ctx.config.training.method is TrainingMethod.REWARD_MODEL
    ):
        # Reward-model pairs must have a verified winner (chosen passed, rejected failed).
        verified = RewardModelDatasetBuilder().build(result.records)
        _write(
            ctx,
            "reward_model",
            verified,
            [to_trl_reward(r) for r in verified],
            "RewardModelDatasetBuilder",
            config,
            {**result.report(), "verified_winner_pairs": len(verified)},
            validate_preference(verified, ctx.contamination()).to_dict(),
        )
    return output


async def stage_build_grpo(ctx: RunContext) -> StageOutput:
    config = ctx.config.datasets.grpo
    builder = RLPromptDatasetBuilder(
        config,
        environments=ctx.deps.environments,
        prompts=ctx.deps.prompts,
        contamination=ctx.contamination(),
    )
    result = builder.build(ctx.tasks("train"), task_stats(_trajectories(ctx)))
    report = validate_rl(result.records, ctx.contamination())
    return _write(
        ctx,
        "grpo",
        result.records,
        [to_trl_grpo(r) for r in result.records],
        "RLPromptDatasetBuilder",
        config,
        result.report(),
        report.to_dict(),
    )


# ---------------------------------------------------------------------------------- training


def _mock_settings(ctx: RunContext) -> MockTrainingSettings:
    training = ctx.config.training
    settings = (training.mock if training and training.mock else None) or MockTrainingSettings()
    if isinstance(ctx.config.policy, MockPolicySpec):
        settings = settings.model_copy(update={"base_profile": ctx.config.policy.profile})
    return settings


def _dataset_dir(ctx: RunContext, method: TrainingMethod) -> Path:
    kind = {
        TrainingMethod.SFT: "sft",
        TrainingMethod.DPO: "dpo",
        TrainingMethod.GRPO: "grpo",
        TrainingMethod.REWARD_MODEL: "reward_model",
    }[method]
    return ctx.path("datasets", kind)


async def stage_train(ctx: RunContext) -> StageOutput:
    training = ctx.config.training
    if training is None:  # pragma: no cover - guarded by config validation
        raise ExperimentError("no training section")
    directory = _dataset_dir(ctx, training.method)
    card: DatasetCard = read_card(directory)
    run = TrainingRun(
        run_id=f"{ctx.run_id}-{training.method.value}",
        method=training.method,
        model=training.model,
        dataset=card.reference(directory),
        config=training.config,
        seed=ctx.config.seed,
        launch_mode=LaunchMode.REMOTE if training.remote_host else LaunchMode.LOCAL,
    )
    root = ctx.path("training")
    if training.backend == "mock":
        backend = MockTrainingBackend(root, _mock_settings(ctx))
        prepared = save_prepared(backend.prepare(run))
        status = backend.inspect(backend.launch(prepared))
        if status.state is not TrainingState.SUCCEEDED or not status.checkpoints:
            raise ExperimentError(f"mock training did not succeed: {status.error}")
        checkpoint = status.checkpoints[-1].path
    elif training.backend == "trl" and not training.remote_host:
        trl_backend = TRLTrainingBackend(root, training.trl)
        prepared = save_prepared(trl_backend.prepare(run))
        trl_backend.launch(prepared)
        checkpoint = str(Path(prepared.workdir) / "outputs" / "final")
    else:
        workdir = root / run.run_id
        summary_path = workdir / SUMMARY_FILE
        if (
            summary_path.exists()
            and json.loads(summary_path.read_text(encoding="utf-8")).get("state") == "succeeded"
        ):
            checkpoint = str(workdir / "outputs" / "final")
        else:
            from rlfactory.training.base import default_backend_registry

            prepared = save_prepared(
                default_backend_registry().create(training.backend, root).prepare(run)
            )
            raise StageBlockedError(
                f"prepared {training.backend} run at {prepared.workdir}. Remote GPU training is never "
                f"launched implicitly: run `rlfactory remote launch {prepared.workdir} --host "
                f"{training.remote_host or '<host>'}`, then `rlfactory remote fetch`, then resume."
            )
    summary_file = Path(root / run.run_id / SUMMARY_FILE)
    if summary_file.exists():
        ctx.path("training_summary.json").write_text(
            summary_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return StageOutput(
        outputs={"checkpoint": checkpoint, "workdir": str(root / run.run_id)},
        summary={
            "backend": training.backend,
            "method": training.method.value,
            "checkpoint": checkpoint,
        },
    )


# -------------------------------------------------------------------------------- evaluation


def candidate_policy(ctx: RunContext, checkpoint_path: str, tasks: Sequence[Task]) -> Policy:
    training = ctx.config.training
    if training is not None and training.backend == "mock":
        checkpoint = load_mock_checkpoint(Path(checkpoint_path))
        base = ctx.config.policy
        seed = base.seed if isinstance(base, MockPolicySpec) else 0
        return MockPolicy(
            tasks,
            checkpoint.simulated_policy_profile,
            policy_id=f"{base.policy_id}-{training.method.value}",
            seed=seed,
            checkpoint=checkpoint.metadata.checkpoint.checkpoint_id,
        )
    if ctx.config.evaluation.candidate_policy is None:
        raise ExperimentError(
            "evaluating a real checkpoint needs evaluation.candidate_policy, e.g. a vLLM server "
            "started with --enable-lora serving the trained adapter"
        )
    return build_policy(ctx.config.evaluation.candidate_policy, tasks)


async def _evaluate(ctx: RunContext, label: str, policy: Policy) -> StageOutput:
    tasks = ctx.tasks("evaluation")
    evaluator = Evaluator(ctx.deps)
    try:
        result, trajectories, _ = await evaluator.evaluate(
            label=label,
            suite="+".join(ctx.config.tasks.eval_suites),
            policy=policy,
            tasks=tasks,
            config=ctx.config.evaluation,
            run_id=ctx.run_id,
            store_root=ctx.path("evaluation", label, "trajectories"),
            progress=ctx.progress(f"evaluate:{label}"),
        )
    finally:
        await policy.aclose()
    path = ctx.path("evaluation", f"{label}.json")
    write_json(result, path)
    write_failures_parquet(
        analyze_failures(trajectories), ctx.path("evaluation", label, "failures.parquet")
    )
    return StageOutput(
        outputs={"result": str(path)},
        summary={
            "verified_accuracy": result.overall.verified_accuracy,
            "format_compliance": result.overall.format_compliance,
            "simulated": result.simulated,
        },
    )


async def stage_evaluate_base(ctx: RunContext) -> StageOutput:
    return await _evaluate(ctx, "base", build_policy(ctx.config.policy, ctx.tasks("evaluation")))


async def stage_evaluate_candidate(ctx: RunContext) -> StageOutput:
    checkpoint = ctx.output("train", "checkpoint")
    return await _evaluate(
        ctx, "candidate", candidate_policy(ctx, checkpoint, ctx.tasks("evaluation"))
    )


async def stage_gate(ctx: RunContext) -> StageOutput:
    base = read_json_model(ctx.path("evaluation", "base.json"), EvaluationResult)
    candidate = read_json_model(ctx.path("evaluation", "candidate.json"), EvaluationResult)
    report = evaluate_gates(base, candidate, ctx.config.evaluation.gates)
    write_json(report, ctx.path("gate_report.json"))
    registry = PromotionRegistry(ctx.settings.results_dir / "model_registry.json")
    checkpoint_id = candidate.checkpoint or f"{ctx.run_id}-candidate"
    if registry.get(checkpoint_id) is None:
        registry.register_candidate(checkpoint_id, simulated=candidate.simulated)
    current = registry.get(checkpoint_id)
    decision = (
        registry.decide(checkpoint_id, report, evaluation_id=candidate.evaluation_id)
        if current is not None and current.state.value == "candidate"
        else current
    )
    write_json(
        {
            "baseline": base.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "gate": report.model_dump(mode="json"),
            "promotion": decision.model_dump(mode="json") if decision else None,
        },
        ctx.path("evaluation.json"),
    )
    return StageOutput(
        outputs={
            "gate_report": str(ctx.path("gate_report.json")),
            "evaluation": str(ctx.path("evaluation.json")),
        },
        summary={"passed": report.passed, "promotion": decision.state.value if decision else None},
    )


# ---------------------------------------------------------------------------------- analysis


async def stage_analysis(ctx: RunContext) -> StageOutput:
    trajectories = _trajectories(ctx)
    analysis = analyze_failures(trajectories)
    write_failures_parquet(analysis, ctx.path("failures.parquet"))
    hacking = RewardHackingAnalyzer().analyze(trajectories)
    calibration = calibrate_rewards(trajectories)
    payload = {
        "calibration": calibration.__dict__,
        "hacking": {
            "examined": hacking.examined,
            "counts": hacking.counts,
            "caveat": hacking.caveat,
            "examples": [f.__dict__ for f in hacking.flags[:20]],
        },
        "failures": {"counts": analysis.counts, "by_domain": analysis.by_domain},
    }
    write_json(json.loads(json.dumps(payload, default=str)), ctx.path("reward_analysis.json"))
    return StageOutput(
        outputs={
            "failures": str(ctx.path("failures.parquet")),
            "reward_analysis": str(ctx.path("reward_analysis.json")),
        },
        summary=json.loads(
            json.dumps(
                {
                    "failure_counts": analysis.counts,
                    "hacking_flags": hacking.counts,
                    "reward_correctness_correlation": calibration.point_biserial,
                }
            )
        ),
    )


# ---------------------------------------------------------------------------------- flywheel


async def stage_flywheel(ctx: RunContext) -> StageOutput:
    config = ctx.config.flywheel
    training = ctx.config.training
    if config is None:  # pragma: no cover - guarded by config validation
        raise ExperimentError("no flywheel section")
    if (
        training is None
        or training.backend != "mock"
        or not isinstance(ctx.config.policy, MockPolicySpec)
    ):
        raise ExperimentError(
            "the automated flywheel stage currently requires the mock backend and policy"
        )
    base_spec = ctx.config.policy
    eval_tasks = ctx.tasks("evaluation")
    dev_tasks = ctx.tasks("validation") or ctx.tasks("train")
    profiles: dict[str, MockPolicyProfile] = {}
    settings = _mock_settings(ctx)

    def policy_for(checkpoint: CheckpointReference | None, tasks: Sequence[Task]) -> Policy:
        profile = profiles[checkpoint.checkpoint_id] if checkpoint else base_spec.profile
        return MockPolicy(
            tasks,
            profile,
            policy_id=f"{base_spec.policy_id}-flywheel" if checkpoint else base_spec.policy_id,
            seed=base_spec.seed,
            checkpoint=checkpoint.checkpoint_id if checkpoint else None,
        )

    current: dict[str, MockPolicyProfile] = {"profile": base_spec.profile}

    # The flywheel trains on rejection-sampled SFT data regardless of the main experiment's method.
    sft_config = (
        training.config
        if training.method is TrainingMethod.SFT
        else training.config.model_copy(update={"method_params": SFTParams()})
    )

    async def train(iteration: int, dataset_dir: Path, card: DatasetCard) -> CheckpointReference:
        backend = MockTrainingBackend(
            ctx.path("flywheel", "training"),
            settings.model_copy(update={"base_profile": current["profile"]}),
        )
        run = TrainingRun(
            run_id=f"{ctx.run_id}-flywheel-{iteration}",
            method=TrainingMethod.SFT,
            model=training.model,
            dataset=card.reference(dataset_dir),
            config=sft_config,
            seed=ctx.config.seed + iteration,
        )
        status = backend.inspect(backend.launch(backend.prepare(run)))
        checkpoint = load_mock_checkpoint(Path(status.checkpoints[-1].path))
        reference = checkpoint.metadata.checkpoint
        profiles[reference.checkpoint_id] = checkpoint.simulated_policy_profile
        current["profile"] = checkpoint.simulated_policy_profile
        return reference

    flywheel = FailureDrivenFlywheel(
        config=config,
        deps=ctx.deps,
        mutator=TaskMutator(ctx.templates, seed=ctx.config.seed),
        pipeline=SynthesisPipeline(
            ctx.templates,
            TaskValidator(ctx.deps.environments, ctx.deps.verifiers, ctx.deps.environment_context),
            ContaminationFilter.from_tasks(eval_tasks),
        ),
        evaluator=Evaluator(ctx.deps),
        evaluation=ctx.config.evaluation,
        gates=ctx.config.evaluation.gates,
        train=train,
        policy_for=policy_for,
        workdir=ctx.path("flywheel"),
        run_id=ctx.run_id,
    )
    iterations = await flywheel.run(dev_tasks, eval_tasks)
    write_json([i.model_dump(mode="json") for i in iterations], ctx.path("flywheel.json"))
    return StageOutput(
        outputs={"flywheel": str(ctx.path("flywheel.json"))},
        summary={
            "iterations": len(iterations) - 1,
            "eval_accuracy": [i.eval_accuracy for i in iterations],
        },
    )


# ------------------------------------------------------------------------------------ report


async def stage_report(ctx: RunContext) -> StageOutput:
    from rlfactory.reporting.report import generate_report

    path = generate_report(ctx.run_dir)
    return StageOutput(outputs={"report": str(path)}, summary={"report": str(path)})


STAGES: dict[str, StageFn] = {
    "tasks": stage_tasks,
    "rollouts": stage_rollouts,
    "build_sft": stage_build_sft,
    "build_dpo": stage_build_dpo,
    "build_grpo": stage_build_grpo,
    "train": stage_train,
    "evaluate_base": stage_evaluate_base,
    "evaluate_candidate": stage_evaluate_candidate,
    "gate": stage_gate,
    "analysis": stage_analysis,
    "flywheel": stage_flywheel,
    "report": stage_report,
}
