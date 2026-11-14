"""RLFactory command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from rlfactory import __version__
from rlfactory.config.loader import dump_yaml, load_model, read_json_model, write_json
from rlfactory.config.settings import Settings
from rlfactory.datasets.adapters import DatasetKind
from rlfactory.domain.common import stable_hash
from rlfactory.domain.result import EvaluationResult
from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.training import TrainingRun
from rlfactory.errors import RLFactoryError
from rlfactory.telemetry.logging import configure_logging

app = typer.Typer(
    name="rlfactory",
    help="Verifier-driven LLM post-training infrastructure.",
    no_args_is_help=True,
    add_completion=False,
)
tasks_app = typer.Typer(help="Task suites: list, inspect, synthesize.", no_args_is_help=True)
environments_app = typer.Typer(help="Environments: list and self-check.", no_args_is_help=True)
rollout_app = typer.Typer(help="Rollout generation.", no_args_is_help=True)
trajectories_app = typer.Typer(help="Trajectory store queries.", no_args_is_help=True)
dataset_app = typer.Typer(help="Dataset construction and validation.", no_args_is_help=True)
experiment_app = typer.Typer(help="Experiment DAGs: plan, run, resume.", no_args_is_help=True)
remote_app = typer.Typer(
    help="Remote GPU hosts over SSH (never provisions infrastructure).", no_args_is_help=True
)
for sub, name in (
    (tasks_app, "tasks"),
    (environments_app, "environments"),
    (rollout_app, "rollout"),
    (trajectories_app, "trajectories"),
    (dataset_app, "dataset"),
    (experiment_app, "experiment"),
    (remote_app, "remote"),
):
    app.add_typer(sub, name=name)

console = Console()
_STATUS_STYLE = {"ok": "green", "info": "cyan", "warn": "yellow", "missing": "dim"}


def _settings(ctx: typer.Context) -> Settings:
    settings = ctx.obj
    if not isinstance(settings, Settings):  # pragma: no cover - set by the callback
        settings = Settings.from_env()
    return settings


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


@app.callback()
def main_callback(
    ctx: typer.Context,
    config_dir: Annotated[
        Path | None, typer.Option(help="Config directory (default ./configs)")
    ] = None,
    results_dir: Annotated[
        Path | None, typer.Option(help="Results directory (default ./results)")
    ] = None,
    log_format: Annotated[str | None, typer.Option(help="text or json")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    settings = Settings.from_env()
    updates: dict[str, Any] = {}
    if config_dir is not None:
        updates["config_dir"] = config_dir
    if results_dir is not None:
        updates["results_dir"] = results_dir
    if log_format is not None:
        updates["log_format"] = log_format
    if verbose:
        updates["log_level"] = "DEBUG"
    settings = settings.model_copy(update=updates)
    configure_logging(settings.log_level if verbose else "WARNING", settings.log_format)
    ctx.obj = settings


@app.command()
def version() -> None:
    """Print the RLFactory version."""
    console.print(__version__)


@app.command()
def doctor(
    ctx: typer.Context,
    no_docker: Annotated[bool, typer.Option("--no-docker", help="Skip Docker checks")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Report Python, CPU, RAM, disk, Docker, ML stack and GPU availability."""
    from rlfactory.runtime.doctor import results_writable, run_checks, summarize

    settings = _settings(ctx)
    checks = run_checks(settings, include_docker=not no_docker)
    if json_output:
        console.print_json(json.dumps([check.__dict__ for check in checks]))
        return
    table = Table(title="rlfactory doctor")
    for column in ("group", "check", "status", "detail"):
        table.add_column(column)
    for check in checks:
        style = _STATUS_STYLE[check.status]
        table.add_row(check.group, check.name, f"[{style}]{check.status}[/{style}]", check.detail)
    console.print(table)
    writable = results_writable(settings.results_dir)
    console.print(
        f"results dir writable: {'yes' if writable else 'NO'}  |  summary: {summarize(checks)}"
    )
    console.print("[dim]No local GPU is expected; GPU training runs on a remote host.[/dim]")


# ------------------------------------------------------------------------------------- tasks


@tasks_app.command("list")
def tasks_list(ctx: typer.Context) -> None:
    """List configured task suites."""
    from rlfactory.tasks.registry import TaskRegistry

    registry = TaskRegistry(_settings(ctx).config_dir)
    table = Table(title="task suites")
    for column in ("suite", "version", "tasks", "domains", "description"):
        table.add_column(column)
    for name in registry.names():
        suite = registry.load(name)
        domains = ", ".join(f"{d.value}:{len(t)}" for d, t in sorted(suite.by_domain().items()))
        table.add_row(name, suite.version, str(len(suite)), domains, suite.description)
    console.print(table)


@tasks_app.command("inspect")
def tasks_inspect(
    ctx: typer.Context,
    suite: str,
    limit: Annotated[int, typer.Option(help="Example tasks to show")] = 5,
    show_payload: Annotated[bool, typer.Option("--show-payload")] = False,
) -> None:
    """Show suite composition, difficulty and example tasks."""
    from collections import Counter

    from rlfactory.tasks.registry import TaskRegistry

    task_suite = TaskRegistry(_settings(ctx).config_dir).load(suite)
    ref = task_suite.ref()
    console.print(
        f"[bold]{ref.name}[/bold] v{ref.version}: {ref.num_tasks} tasks, fingerprint {ref.fingerprint}"
    )
    families = Counter(t.family for t in task_suite.tasks)
    levels = Counter(t.metadata.difficulty.level for t in task_suite.tasks if t.metadata.difficulty)
    console.print("families: " + ", ".join(f"{k}={v}" for k, v in sorted(families.items())))
    console.print("difficulty levels: " + ", ".join(f"L{k}={v}" for k, v in sorted(levels.items())))
    for task in task_suite.tasks[:limit]:
        console.rule(f"{task.task_id} [{task.family}] verifier={task.verifier}")
        console.print(task.prompt)
        if show_payload:
            console.print_json(task.payload.model_dump_json())


@tasks_app.command("synthesize")
def tasks_synthesize(
    ctx: typer.Context,
    suite: str,
    out: Annotated[Path, typer.Option(help="Output JSONL path")],
    exclude_suite: Annotated[
        list[str] | None, typer.Option(help="Held-out suites to decontaminate against")
    ] = None,
    validate: Annotated[bool, typer.Option("--validate/--no-validate")] = True,
) -> None:
    """Generate -> validate -> deduplicate -> decontaminate a suite and write it as JSONL."""
    from rlfactory.environments.base import EnvironmentContext
    from rlfactory.environments.registry import default_environment_registry
    from rlfactory.sandbox.subprocess_sandbox import SubprocessSandbox
    from rlfactory.tasks.contamination import ContaminationFilter
    from rlfactory.tasks.io import save_tasks
    from rlfactory.tasks.registry import TaskRegistry
    from rlfactory.tasks.synthesis import GeneratedSource, SynthesisPipeline
    from rlfactory.tasks.validation import TaskValidator
    from rlfactory.verifiers.registry import VerifierRegistry

    registry = TaskRegistry(_settings(ctx).config_dir)
    config = registry.config(suite)
    sources = [s for s in config.sources if isinstance(s, GeneratedSource)]
    held_out = [t for name in exclude_suite or [] for t in registry.load(name).tasks]
    validator = (
        TaskValidator(
            default_environment_registry(),
            VerifierRegistry(),
            EnvironmentContext(sandbox=SubprocessSandbox()),
        )
        if validate
        else None
    )
    pipeline = SynthesisPipeline(
        registry.templates, validator, ContaminationFilter.from_tasks(held_out)
    )
    tasks, report = pipeline.run(sources)
    count = save_tasks(tasks, out)
    write_json(report.to_dict(), out.with_suffix(".report.json"))
    console.print(
        f"wrote {count} tasks to {out}; report: {json.dumps({k: v for k, v in report.to_dict().items() if k != 'invalid_reasons'})}"
    )


# ------------------------------------------------------------------------------ environments


@environments_app.command("list")
def environments_list() -> None:
    """List registered environments."""
    from rlfactory.environments.registry import default_environment_registry

    table = Table(title="environments")
    for column in ("name", "version", "domain", "multi-step", "sandbox", "description"):
        table.add_column(column)
    for spec in default_environment_registry().specs():
        table.add_row(
            spec.name,
            spec.version,
            spec.domain.value,
            str(spec.multi_step),
            str(spec.requires_sandbox),
            spec.description,
        )
    console.print(table)


@environments_app.command("check")
def environments_check(
    name: str,
    samples: Annotated[int, typer.Option(help="Generated tasks to validate")] = 4,
    sandbox: Annotated[str, typer.Option(help="subprocess | docker")] = "subprocess",
) -> None:
    """Self-check: reference solutions must pass and known-wrong answers must fail."""
    from rlfactory.environments.base import EnvironmentContext
    from rlfactory.environments.registry import default_environment_registry
    from rlfactory.sandbox.factory import SandboxConfig, create_sandbox
    from rlfactory.tasks.generators.catalog import default_template_registry
    from rlfactory.tasks.synthesis import GeneratedSource, generate_from_source
    from rlfactory.tasks.validation import TaskValidator
    from rlfactory.verifiers.registry import VerifierRegistry

    environments = default_environment_registry()
    spec = environments.spec(name)
    box = create_sandbox(SandboxConfig(kind=sandbox))  # type: ignore[arg-type]
    caps = box.capabilities()
    console.print(
        f"sandbox {caps.name}: network isolation={caps.network_isolation}, memory={caps.memory_limit}"
    )
    tasks = generate_from_source(
        GeneratedSource(domain=spec.domain, count=samples, seed=13), default_template_registry()
    )
    validator = TaskValidator(environments, VerifierRegistry(), EnvironmentContext(sandbox=box))
    table = Table(title=f"environment check: {name}")
    for column in ("task", "family", "valid", "reasons"):
        table.add_column(column)
    failures = 0
    for task in tasks:
        outcome = validator.validate(task)
        failures += int(not outcome.valid)
        table.add_row(
            task.task_id,
            task.family,
            "yes" if outcome.valid else "[red]NO[/red]",
            "; ".join(outcome.reasons),
        )
    console.print(table)
    if failures:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------------- rollouts


def _load_suite_tasks(settings: Settings, suites: tuple[str, ...], cap: int | None) -> list[Task]:
    from rlfactory.tasks.contamination import deduplicate
    from rlfactory.tasks.registry import TaskRegistry

    registry = TaskRegistry(settings.config_dir)
    tasks: list[Task] = []
    for name in suites:
        suite = list(registry.load(name).tasks)
        tasks.extend(suite[:cap] if cap else suite)
    return deduplicate(tasks)[0]


def _run_rollout_job(settings: Settings, job_path: Path, run_id: str) -> None:
    from rlfactory.cli.jobs import RolloutJob
    from rlfactory.cli.progress import RichRolloutProgress
    from rlfactory.policies.factory import build_policy
    from rlfactory.rollouts.service import RolloutDependencies, run_rollouts
    from rlfactory.runtime.disk import DiskGuard, estimate_rollout_bytes
    from rlfactory.sandbox.factory import create_sandbox
    from rlfactory.trajectories.parquet_store import LocalParquetTrajectoryStore

    job = load_model(job_path, RolloutJob)
    tasks = _load_suite_tasks(settings, job.suites, job.max_tasks_per_suite)
    run_dir = settings.run_dir(run_id)
    DiskGuard(settings.disk_reserve_gb).ensure(
        run_dir,
        estimate_rollout_bytes(len(tasks) * job.rollouts.samples_per_task),
        "rollout generation",
    )
    store = LocalParquetTrajectoryStore(
        run_dir / "rollouts",
        retention=job.rollouts.retention,
        sample_rate=job.rollouts.retention_sample_rate,
    )
    deps = RolloutDependencies.default(
        sandbox=create_sandbox(job.sandbox),
        python_limits=job.sandbox.limits,
        verifier_specs=job.verifiers,
        reward_specs=job.rewards,
    )
    policy = build_policy(job.policy, tasks)

    async def go() -> Any:
        try:
            return await run_rollouts(
                run_id=run_id,
                tasks=tasks,
                policy=policy,
                config=job.rollouts,
                store=store,
                deps=deps,
                progress=RichRolloutProgress(console, f"rollouts {run_id}"),
            )
        finally:
            await policy.aclose()

    summary = asyncio.run(go())
    write_json(summary, run_dir / "rollout_summary.json")
    console.print(f"trajectories: {store.root}  summary: {run_dir / 'rollout_summary.json'}")


@rollout_app.command("run")
def rollout_run(
    ctx: typer.Context,
    config: Path,
    run_id: Annotated[str | None, typer.Option(help="Explicit run ID (default: derived)")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the job plan without generating anything")
    ] = False,
) -> None:
    """Generate rollouts for a rollout job config."""
    from rlfactory.cli.jobs import RolloutJob

    settings = _settings(ctx)
    job = load_model(config, RolloutJob)
    tasks = _load_suite_tasks(settings, job.suites, job.max_tasks_per_suite)
    total = len(tasks) * job.rollouts.samples_per_task
    run_id = run_id or f"{job.name}-{stable_hash(job.model_dump(mode='json'), length=8)}"
    console.print(
        f"run {run_id}: {len(tasks)} tasks x {job.rollouts.samples_per_task} samples = {total} trajectories, policy {job.policy.kind}"
    )
    if dry_run:
        console.print("[dim]dry run: nothing executed[/dim]")
        return
    run_dir = settings.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(job, run_dir / "rollout_job.yaml")
    _run_rollout_job(settings, run_dir / "rollout_job.yaml", run_id)


@rollout_app.command("resume")
def rollout_resume(ctx: typer.Context, run_id: str) -> None:
    """Resume an interrupted rollout run (completed trajectory IDs are skipped)."""
    settings = _settings(ctx)
    job_path = settings.run_dir(run_id) / "rollout_job.yaml"
    if not job_path.exists():
        _fail(f"{job_path} not found")
    _run_rollout_job(settings, job_path, run_id)


# ------------------------------------------------------------------------------ trajectories


def _store(settings: Settings, run_id: str) -> Any:
    from rlfactory.trajectories.parquet_store import LocalParquetTrajectoryStore

    root = settings.run_dir(run_id) / "rollouts"
    if not root.exists():
        _fail(f"no trajectory store at {root}")
    return LocalParquetTrajectoryStore(root)


@trajectories_app.command("inspect")
def trajectories_inspect(ctx: typer.Context, run_id: str) -> None:
    """Aggregate statistics of a run's trajectories."""
    from rlfactory.trajectories.store import TrajectoryQuery

    store = _store(_settings(ctx), run_id)
    for group in (["domain"], ["status", "failure_kind"], ["reason_code"]):
        rows = store.aggregate(group, TrajectoryQuery())
        table = Table(title=f"by {', '.join(group)}")
        for column in [*group, "trajectories", "completed", "failed", "pass_rate", "mean_reward"]:
            table.add_column(column)
        for row in rows:
            table.add_row(
                *[
                    str(row[c]) if not isinstance(row[c], float) else f"{row[c]:.3f}"
                    for c in [*group, "trajectories", "completed", "failed"]
                ],
                *(
                    f"{row[k]:.3f}" if isinstance(row[k], float) else "-"
                    for k in ("pass_rate", "mean_reward")
                ),
            )
        console.print(table)


@trajectories_app.command("query")
def trajectories_query(
    ctx: typer.Context,
    run_id: str,
    task_id: Annotated[str | None, typer.Option()] = None,
    domain: Annotated[str | None, typer.Option()] = None,
    environment: Annotated[str | None, typer.Option()] = None,
    policy_id: Annotated[str | None, typer.Option()] = None,
    checkpoint: Annotated[str | None, typer.Option()] = None,
    verifier: Annotated[str | None, typer.Option()] = None,
    passed: Annotated[bool | None, typer.Option("--passed/--failed", show_default=False)] = None,
    min_reward: Annotated[float | None, typer.Option()] = None,
    max_reward: Annotated[float | None, typer.Option()] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option()] = 20,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Filter trajectories by task, environment, policy, checkpoint, verifier result, reward, seed."""
    from rlfactory.trajectories.store import TrajectoryQuery

    query = TrajectoryQuery(
        run_id=None,
        task_id=task_id,
        domain=domain,
        environment=environment,
        policy_id=policy_id,
        checkpoint=checkpoint,
        verifier=verifier,
        passed=passed,
        min_reward=min_reward,
        max_reward=max_reward,
        seed=seed,
        limit=limit,
    )
    rows = _store(_settings(ctx), run_id).query(query)
    if json_output:
        console.print_json(json.dumps([r.model_dump(mode="json") for r in rows]))
        return
    table = Table(title=f"{len(rows)} trajectories")
    for column in (
        "trajectory_id",
        "task_id",
        "domain",
        "status",
        "passed",
        "reward",
        "reason",
        "turns",
    ):
        table.add_column(column)
    for r in rows:
        table.add_row(
            r.trajectory_id,
            r.task_id,
            r.domain,
            r.status,
            str(r.passed),
            f"{r.reward:.3f}" if r.reward is not None else "-",
            r.reason_code or r.failure_kind or "-",
            str(r.num_turns),
        )
    console.print(table)


@trajectories_app.command("show")
def trajectories_show(ctx: typer.Context, run_id: str, trajectory_id: str) -> None:
    """Print one trajectory's conversation, verification and reward breakdown."""
    from rlfactory.trajectories.store import TrajectoryQuery

    store = _store(_settings(ctx), run_id)
    matches = [t for t in store.load(TrajectoryQuery()) if t.trajectory_id == trajectory_id]
    if not matches:
        _fail(f"{trajectory_id} not found (or its payload was not retained)")
    trajectory = matches[0]
    for message in trajectory.messages:
        console.rule(message.role.value)
        console.print(message.content, markup=False)
    console.rule("verification")
    console.print_json(
        trajectory.verification.model_dump_json() if trajectory.verification else "null"
    )
    console.rule("reward")
    console.print_json(trajectory.rewards.model_dump_json() if trajectory.rewards else "null")


# ---------------------------------------------------------------------------------- datasets


def _build_dataset(settings: Settings, config: Path, kind: DatasetKind) -> None:
    from rlfactory.cli.jobs import DatasetJob
    from rlfactory.datasets.adapters import to_trl_dpo, to_trl_grpo, to_trl_sft
    from rlfactory.datasets.builders import (
        PreferenceDatasetBuilder,
        RLPromptDatasetBuilder,
        SFTDatasetBuilder,
    )
    from rlfactory.datasets.io import write_dataset
    from rlfactory.datasets.validation import validate_preference, validate_rl, validate_sft
    from rlfactory.environments.registry import default_environment_registry
    from rlfactory.runtime.disk import DiskGuard
    from rlfactory.runtime.git import detect_code_version
    from rlfactory.tasks.contamination import ContaminationFilter
    from rlfactory.tasks.sampling import task_stats
    from rlfactory.trajectories.store import TrajectoryQuery

    job = load_model(config, DatasetJob)
    held_out = _load_suite_tasks(settings, job.eval_suites, None) if job.eval_suites else []
    contamination = ContaminationFilter.from_tasks(
        held_out, exclude_families=job.exclude_eval_families
    )
    train_tasks = _load_suite_tasks(settings, job.train_suites, None) if job.train_suites else []
    index = {t.task_id: t for t in [*train_tasks, *held_out]}
    trajectories = _store(settings, job.source_run).load(TrajectoryQuery(run_id=job.source_run))
    records: list[BaseModel]
    rows: list[dict[str, Any]]
    builder_config: BaseModel
    if kind == "sft":
        sft = SFTDatasetBuilder(job.sft, contamination=contamination, tasks=index).build(
            trajectories
        )
        records, rows, builder_config = (
            list(sft.records),
            [to_trl_sft(r) for r in sft.records],
            job.sft,
        )
        report, build_report = validate_sft(sft.records, contamination), sft.report()
    elif kind == "dpo":
        dpo = PreferenceDatasetBuilder(job.dpo, contamination=contamination, tasks=index).build(
            trajectories
        )
        records, rows, builder_config = (
            list(dpo.records),
            [to_trl_dpo(r) for r in dpo.records],
            job.dpo,
        )
        report, build_report = validate_preference(dpo.records, contamination), dpo.report()
    else:
        if not train_tasks:
            _fail("build-grpo needs train_suites")
        rl = RLPromptDatasetBuilder(
            job.grpo, environments=default_environment_registry(), contamination=contamination
        ).build(train_tasks, task_stats(trajectories))
        records, rows, builder_config = (
            list(rl.records),
            [to_trl_grpo(r) for r in rl.records],
            job.grpo,
        )
        report, build_report = validate_rl(rl.records, contamination), rl.report()
    if not report.ok:
        _fail(f"dataset failed validation: {report.errors[:5]}")
    directory = settings.results_dir / "datasets" / f"{job.name}-{kind}"
    card = write_dataset(
        directory,
        name=f"{job.name}-{kind}",
        kind=kind,
        records=records,
        rows=rows,
        builder=type(builder_config).__name__,
        builder_config=builder_config,
        build_report=build_report,
        validation=report.to_dict(),
        source_run_ids=[job.source_run],
        contamination_counts={k: len(v) for k, v in contamination.to_dict().items()},
        code=detect_code_version(),
        disk_guard=DiskGuard(settings.disk_reserve_gb),
    )
    console.print(
        f"{card.kind}: {card.num_records} records -> {directory} (schema {card.schema_name})"
    )
    console.print(f"build report: {json.dumps(build_report)}")


@dataset_app.command("build-sft")
def dataset_build_sft(ctx: typer.Context, config: Path) -> None:
    """Build an SFT dataset from successful trajectories."""
    _build_dataset(_settings(ctx), config, "sft")


@dataset_app.command("build-dpo")
def dataset_build_dpo(ctx: typer.Context, config: Path) -> None:
    """Build a preference (DPO) dataset from rollout groups."""
    _build_dataset(_settings(ctx), config, "dpo")


@dataset_app.command("build-grpo")
def dataset_build_grpo(ctx: typer.Context, config: Path) -> None:
    """Build a prompt-only GRPO/RLVR dataset with verifier-backed rewards."""
    _build_dataset(_settings(ctx), config, "grpo")


@dataset_app.command("validate")
def dataset_validate(
    dataset: Path,
    kind: Annotated[
        str | None, typer.Option(help="sft | dpo | grpo | reward_model (default: from card)")
    ] = None,
) -> None:
    """Validate a TRL-formatted dataset (directory with a card, or a JSONL file with --kind)."""
    from rlfactory.datasets.io import TRAIN_FILE, read_card
    from rlfactory.datasets.validation import validate_trl_file

    path = dataset / TRAIN_FILE if dataset.is_dir() else dataset
    resolved = kind or (read_card(dataset).kind if dataset.is_dir() else None)
    if resolved not in ("sft", "dpo", "grpo", "reward_model"):
        _fail("pass --kind or a dataset directory containing dataset_card.json")
    report = validate_trl_file(path, resolved)  # type: ignore[arg-type]
    console.print_json(json.dumps(report.to_dict()))
    if not report.ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------------- training


def _memory_estimate(run: TrainingRun) -> Any:
    from rlfactory.runtime.hf_cache import find_cached_model
    from rlfactory.training.memory import KNOWN_SHAPES, ModelShape, estimate_training_memory

    shape = KNOWN_SHAPES.get(run.model.name)
    cached = find_cached_model(run.model.name)
    if shape is None and cached is not None:
        configs = sorted(cached.path.glob("snapshots/*/config.json"))
        if configs:
            shape = ModelShape.from_hf_config(json.loads(configs[-1].read_text(encoding="utf-8")))
    if shape is None:
        return None
    params = run.config.method_params
    sequence = (
        getattr(params, "max_length", None) or getattr(params, "max_completion_length", 512) + 512
    )
    return estimate_training_memory(
        shape,
        method=run.method,
        lora=run.config.lora,
        batch_size=run.config.per_device_train_batch_size,
        sequence_length=int(sequence),
        gradient_checkpointing=run.config.gradient_checkpointing,
        num_generations=getattr(params, "num_generations", 1),
    )


@app.command()
def train(
    ctx: typer.Context,
    config: Path,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Prepare and estimate only")] = False,
) -> None:
    """Prepare (and for local backends, launch) a training run."""
    from rlfactory.cli.jobs import TrainJob
    from rlfactory.datasets.io import read_card
    from rlfactory.training.base import default_backend_registry, save_prepared
    from rlfactory.training.mock.backend import MockTrainingBackend
    from rlfactory.training.trl.backend import TRLTrainingBackend

    settings = _settings(ctx)
    job = load_model(config, TrainJob)
    directory = Path(job.dataset_dir)
    card = read_card(directory)
    run = TrainingRun(
        run_id=job.name,
        method=job.method,
        model=job.model,
        dataset=card.reference(directory),
        config=job.config,
        seed=job.seed,
    )
    root = settings.results_dir / "training"
    backend: Any
    if job.backend == "mock":
        backend = MockTrainingBackend(root, job.mock)
    elif job.backend == "trl":
        backend = TRLTrainingBackend(root, job.trl)
    else:
        backend = default_backend_registry().create(job.backend, root)
    prepared = save_prepared(backend.prepare(run))
    console.print(f"prepared {prepared.backend} run in {prepared.workdir}")
    console.print(f"command: {' '.join(prepared.command)}")
    for note in prepared.notes:
        console.print(f"[dim]- {note}[/dim]")
    estimate = _memory_estimate(run)
    if estimate is not None:
        console.print(
            f"GPU memory estimate (heuristic): ~{estimate.total_gb:.1f} GB "
            f"({estimate.trainable_parameters_m:.1f}M trainable params)"
        )
    if dry_run:
        return
    handle = backend.launch(prepared)
    status = backend.inspect(handle)
    console.print(
        f"state: {status.state.value}, step {status.step}"
        + (" (simulated)" if status.simulated else "")
    )


# -------------------------------------------------------------------------------- evaluation


@app.command()
def evaluate(
    ctx: typer.Context,
    config: Path,
    run_id: Annotated[str | None, typer.Option()] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show suites and sample counts without evaluating")
    ] = False,
) -> None:
    """Evaluate a policy on held-out suites; optionally gate against a baseline evaluation."""
    from rlfactory.cli.jobs import EvaluateJob
    from rlfactory.cli.progress import RichRolloutProgress
    from rlfactory.evaluation.evaluator import Evaluator
    from rlfactory.evaluation.gates import evaluate_gates
    from rlfactory.policies.factory import build_policy
    from rlfactory.rollouts.service import RolloutDependencies
    from rlfactory.sandbox.factory import create_sandbox

    settings = _settings(ctx)
    job = load_model(config, EvaluateJob)
    tasks = _load_suite_tasks(settings, job.suites, job.max_tasks_per_suite)
    run_id = run_id or f"{job.name}-{stable_hash(job.model_dump(mode='json'), length=8)}"
    if dry_run:
        console.print(
            f"evaluation {run_id}: {len(tasks)} tasks x {job.evaluation.samples_per_task} samples, "
            f"policy {job.policy.kind}, baseline {job.baseline or '-'} [dim](dry run)[/dim]"
        )
        return
    run_dir = settings.run_dir(run_id)
    deps = RolloutDependencies.default(
        sandbox=create_sandbox(job.sandbox),
        python_limits=job.sandbox.limits,
        verifier_specs=job.verifiers,
        reward_specs=job.rewards,
    )
    policy = build_policy(job.policy, tasks)

    async def go() -> Any:
        try:
            return await Evaluator(deps).evaluate(
                label=job.label,
                suite="+".join(job.suites),
                policy=policy,
                tasks=tasks,
                config=job.evaluation,
                run_id=run_id,
                store_root=run_dir / "evaluation" / job.label / "trajectories",
                progress=RichRolloutProgress(console, f"evaluate {job.label}"),
            )
        finally:
            await policy.aclose()

    result, _, _ = asyncio.run(go())
    path = run_dir / "evaluation" / f"{job.label}.json"
    write_json(result, path)
    o = result.overall
    console.print(
        f"{job.label}: verified accuracy {o.verified_accuracy} (CI {o.accuracy_ci95}), "
        f"format {o.format_compliance}, infra failures {o.infra_failure_rate} -> {path}"
        + ("  [yellow](simulated)[/yellow]" if result.simulated else "")
    )
    if job.baseline:
        baseline = read_json_model(Path(job.baseline), EvaluationResult)
        report = evaluate_gates(baseline, result, job.gates)
        write_json(report, run_dir / "gate_report.json")
        for check in report.checks:
            console.print(
                f"{'PASS' if check.passed else 'FAIL'}  {check.rule.name}: {check.detail}"
            )
        console.print(f"gate: {'PASSED' if report.passed else 'FAILED'}")
        if not report.passed:
            raise typer.Exit(code=2)


# ------------------------------------------------------------------------------- experiments


def _print_plan(plan: Any) -> None:
    table = Table(title=f"experiment plan: {plan.experiment}")
    for column in ("stage", "depends on", "description"):
        table.add_column(column)
    for stage in plan.stages:
        table.add_row(stage.name, ", ".join(stage.depends_on) or "-", stage.description)
    console.print(table)
    console.print(
        f"tasks: {plan.train_tasks} train / {plan.eval_tasks} eval | rollout trajectories: "
        f"{plan.rollout_trajectories} | evaluation trajectories: {plan.evaluation_trajectories}\n"
        f"datasets: {', '.join(plan.datasets) or '-'} | training: {plan.training_backend or '-'} "
        f"({plan.training_method or '-'}) | est. disk: {plan.estimated_disk_mb} MB | simulated: {plan.simulated}\n"
        f"external requirements: {'; '.join(plan.external_requirements)}"
    )


def _runner(settings: Settings) -> Any:
    from rlfactory.cli.progress import RichRolloutProgress
    from rlfactory.experiments.runner import ExperimentRunner

    return ExperimentRunner(settings, progress=lambda stage: RichRolloutProgress(console, stage))


def _print_records(records: list[Any], run_dir: Path) -> None:
    table = Table(title=f"run {run_dir.name}")
    for column in ("stage", "status", "seconds", "summary"):
        table.add_column(column)
    for record in records:
        status = record.status + (" (cached)" if record.cached else "")
        summary = (
            json.dumps(record.summary)[:110]
            if record.status == "completed"
            else (record.error or "")[:200]
        )
        table.add_row(record.name, status, f"{record.duration_s:.1f}", summary)
    console.print(table)
    report = run_dir / "report.md"
    if report.exists():
        console.print(f"report: {report}")


@experiment_app.command("plan")
def experiment_plan(ctx: typer.Context, config: Path) -> None:
    """Show stages, task/rollout counts, backend and external requirements without executing."""
    from rlfactory.experiments.config import ExperimentConfig

    _print_plan(_runner(_settings(ctx)).plan(load_model(config, ExperimentConfig)))


def _stage_list(values: list[str] | None) -> list[str] | None:
    return [v for value in values for v in value.split(",") if v] if values else None


@experiment_app.command("run")
def experiment_run(
    ctx: typer.Context,
    config: Path,
    run_id: Annotated[str | None, typer.Option()] = None,
    only: Annotated[list[str] | None, typer.Option(help="Run only these stages")] = None,
    until: Annotated[str | None, typer.Option(help="Stop after this stage")] = None,
    force: Annotated[
        list[str] | None, typer.Option(help="Re-run these stages even if cached")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the stage plan without executing it")
    ] = False,
) -> None:
    """Create a run directory and execute the experiment DAG."""
    from rlfactory.experiments.config import ExperimentConfig

    settings = _settings(ctx)
    experiment = load_model(config, ExperimentConfig)
    runner = _runner(settings)
    if dry_run:
        _print_plan(runner.plan(experiment))
        return
    run_dir = runner.create_run(experiment, run_id)
    console.print(f"run directory: {run_dir}")
    records = asyncio.run(
        runner.run(run_dir, only=_stage_list(only), until=until, force=_stage_list(force) or ())
    )
    _print_records(records, run_dir)


@experiment_app.command("resume")
def experiment_resume(
    ctx: typer.Context,
    run_id: str,
    only: Annotated[list[str] | None, typer.Option()] = None,
    until: Annotated[str | None, typer.Option()] = None,
    force: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    """Resume an experiment: completed stages with unchanged fingerprints are skipped."""
    settings = _settings(ctx)
    run_dir = settings.run_dir(run_id)
    if not (run_dir / "config.yaml").exists():
        _fail(f"no experiment run at {run_dir}")
    records = asyncio.run(
        _runner(settings).run(
            run_dir, only=_stage_list(only), until=until, force=_stage_list(force) or ()
        )
    )
    _print_records(records, run_dir)


@app.command()
def report(ctx: typer.Context, run_id: str) -> None:
    """(Re)generate report.md and figures for a run."""
    from rlfactory.reporting.report import generate_report

    run_dir = _settings(ctx).run_dir(run_id)
    if not run_dir.exists():
        _fail(f"{run_dir} not found")
    console.print(f"report: {generate_report(run_dir)}")


# ------------------------------------------------------------------------------------ remote


def _remote(host_config: Path, dry_run: bool) -> tuple[Any, Any]:
    from rlfactory.remote.commands import DryRunCommandRunner, LocalCommandRunner
    from rlfactory.remote.host import load_remote_host
    from rlfactory.remote.runner import RemoteTrainingRunner
    from rlfactory.remote.ssh import SSHClient

    host = load_remote_host(host_config)
    runner: Any = DryRunCommandRunner() if dry_run else LocalCommandRunner()
    return RemoteTrainingRunner(host, SSHClient(host, runner)), runner


def _print_dry_run(runner: Any) -> None:
    from rlfactory.remote.commands import DryRunCommandRunner

    if isinstance(runner, DryRunCommandRunner):
        console.print("[dim]dry run - commands that would be executed:[/dim]")
        for command in runner.commands:
            console.print("  " + " ".join(command), markup=False)


@remote_app.command("check")
def remote_check(
    host: Path,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the ssh command instead of running it")
    ] = False,
) -> None:
    """Query GPUs on a configured host (read-only)."""
    remote, runner = _remote(host, dry_run)
    gpus = remote.check_gpu()
    _print_dry_run(runner)
    for gpu in gpus:
        console.print(f"{gpu.name}: {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB used")
    if not gpus and not dry_run:
        _fail("no GPU visible (or host unreachable)")


@remote_app.command("launch")
def remote_launch(
    workdir: Path,
    host: Annotated[Path, typer.Option(help="configs/remote/<name>.yaml")],
    sync_code: Annotated[bool, typer.Option("--sync-code/--no-sync-code")] = True,
    prepare_env: Annotated[bool, typer.Option("--prepare-env/--no-prepare-env")] = True,
    smoke: Annotated[bool, typer.Option("--smoke/--skip-smoke")] = True,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print every ssh/rsync command; launch nothing")
    ] = False,
) -> None:
    """Preflight, sync, smoke-test and launch a prepared run on an existing GPU host."""
    from rlfactory.training.base import load_prepared

    prepared = load_prepared(workdir)
    remote, runner = _remote(host, dry_run)
    estimate = _memory_estimate(prepared.run)
    if not dry_run and not yes:
        typer.confirm(
            f"Launch {prepared.backend} run {prepared.run.run_id} on {remote.host.target}? "
            "This uses the remote machine's GPUs (and any cost it incurs).",
            abort=True,
        )
    report = remote.preflight(prepared, estimate)
    for note in report.notes:
        console.print(f"[dim]{note}[/dim]")
    if not report.ok and not dry_run:
        _fail("preflight failed: " + "; ".join(report.problems))
    if sync_code:
        remote.sync_code(Path.cwd())
    if prepare_env:
        remote.prepare_environment()
    remote.sync_run(prepared)
    if smoke and prepared.backend == "trl":
        remote.smoke(prepared)
    handle = remote.launch(prepared)
    _print_dry_run(runner)
    if dry_run:
        # Never write (or overwrite) remote_handle.json here: it may track a real, running job.
        console.print("dry run: nothing was launched and no handle was written")
        return
    write_json(handle, workdir / "remote_handle.json")
    console.print(f"launched: {handle.handle_id} on {handle.remote_host} ({handle.workdir})")


@remote_app.command("status")
def remote_status(workdir: Path, host: Annotated[Path, typer.Option()]) -> None:
    """Poll a remote run's status files."""
    from rlfactory.domain.training import TrainingHandle

    handle = read_json_model(workdir / "remote_handle.json", TrainingHandle)
    remote, _ = _remote(host, False)
    status = remote.status(handle)
    console.print(
        f"{handle.handle_id}: {status.state.value}, step {status.step}"
        + (f", error: {status.error}" if status.error else "")
    )


@remote_app.command("fetch")
def remote_fetch(workdir: Path, host: Annotated[Path, typer.Option()]) -> None:
    """Download lightweight results (never weights) into the local run directory."""
    from rlfactory.domain.training import TrainingHandle

    handle = read_json_model(workdir / "remote_handle.json", TrainingHandle)
    remote, _ = _remote(host, False)
    fetched = remote.fetch_results(handle, workdir)
    console.print(f"fetched {len(fetched)} files into {workdir}: {', '.join(fetched)}")


models_app = typer.Typer(
    help="Model checkpoints: download planning (never downloads).", no_args_is_help=True
)
external_app = typer.Typer(help="Optional external benchmark harnesses.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(external_app, name="external")


@models_app.command("plan")
def models_plan(
    ctx: typer.Context,
    model: list[str],
    expected_size_gb: Annotated[
        list[float] | None, typer.Option(help="Expected size per model (GB)")
    ] = None,
    allow_download: Annotated[bool, typer.Option("--allow-download")] = False,
) -> None:
    """Show model, expected size, free disk and cache path before any download; refuse risky plans."""
    from rlfactory.domain.model import ModelReference
    from rlfactory.runtime.disk import GIB, DiskGuard
    from rlfactory.runtime.hf_cache import CheckpointGuard

    sizes = {
        name: int(size * GIB) for name, size in zip(model, expected_size_gb or [], strict=False)
    }
    guard = CheckpointGuard(DiskGuard(_settings(ctx).disk_reserve_gb))
    plans = guard.plan(
        [ModelReference(name=name) for name in model], sizes, allow_download=allow_download
    )
    for plan in plans:
        console.print(plan.describe())
    if not all(plan.allowed for plan in plans):
        raise typer.Exit(code=1)


@tasks_app.command("sample")
def tasks_sample(
    ctx: typer.Context,
    suite: str,
    n: Annotated[int, typer.Option(help="Number of tasks to sample")],
    out: Annotated[Path, typer.Option(help="Output JSONL path")],
    strategy: Annotated[
        str,
        typer.Option(help="uniform | difficulty_balanced | failure_focused | reward_stratified"),
    ] = "uniform",
    stats_run: Annotated[
        str | None, typer.Option(help="Run ID whose trajectories provide pass/reward statistics")
    ] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Curriculum sampling of a suite (failure-focused / reward-stratified use a run's statistics)."""
    from rlfactory.tasks.io import save_tasks
    from rlfactory.tasks.registry import TaskRegistry
    from rlfactory.tasks.sampling import CurriculumSampler, SamplingStrategy, task_stats
    from rlfactory.trajectories.store import TrajectoryQuery

    settings = _settings(ctx)
    tasks = list(TaskRegistry(settings.config_dir).load(suite).tasks)
    stats = task_stats(_store(settings, stats_run).load(TrajectoryQuery())) if stats_run else {}
    sampler = CurriculumSampler(SamplingStrategy(strategy), seed=seed, stats=stats)
    sampled = sampler.sample(tasks, n)
    save_tasks(sampled, out)
    console.print(f"sampled {len(sampled)} of {len(tasks)} tasks ({strategy}) -> {out}")


@external_app.command("lm-eval")
def external_lm_eval(
    model: Annotated[str, typer.Option(help="Served model name or HF repo id")],
    base_url: Annotated[str | None, typer.Option(help="OpenAI-compatible server root")] = None,
    tasks: Annotated[str, typer.Option(help="Comma-separated lm-eval task names")] = "gsm8k",
    backend: Annotated[
        str, typer.Option(help="local-chat-completions | local-completions | hf")
    ] = "local-chat-completions",
    limit: Annotated[int | None, typer.Option()] = None,
    execute: Annotated[
        bool, typer.Option("--execute", help="Run it (default: print the command only)")
    ] = False,
) -> None:
    """Build (and optionally run) an lm-evaluation-harness command; never part of CI."""
    import shlex
    import shutil
    import subprocess

    from rlfactory.evaluation.external import LMEvalHarnessConfig, build_lm_eval_command

    config = LMEvalHarnessConfig.model_validate(
        {
            "model": model,
            "base_url": base_url,
            "tasks": tasks.split(","),
            "model_backend": backend,
            "limit": limit,
        }
    )
    command = build_lm_eval_command(config)
    console.print(shlex.join(command), markup=False)
    if execute:
        if shutil.which("lm_eval") is None:
            _fail("lm_eval is not installed in this environment")
        raise typer.Exit(code=subprocess.run(command, check=False).returncode)  # noqa: S603


def main() -> None:
    try:
        app()
    except RLFactoryError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise SystemExit(1) from exc


__all__ = ["TaskDomain", "app", "main"]
