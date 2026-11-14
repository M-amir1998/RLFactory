"""Experiment runner: DAG execution with partial runs, conservative stage caching and resume.

A stage is skipped as *cached* only when its previous record completed and its fingerprint is
unchanged. The fingerprint covers the stage's own config slice, the fingerprints of the stages it
depends on, and a hash of RLFactory's source files (so any code change invalidates caches). The
report stage always re-runs.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Collection
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from rlfactory.config.loader import dump_yaml, load_model, read_json_model, write_json
from rlfactory.config.settings import Settings
from rlfactory.domain.common import ComponentRef, stable_hash
from rlfactory.domain.experiment import RunManifest
from rlfactory.errors import ExperimentError
from rlfactory.experiments.config import (
    STAGE_ORDER,
    ExperimentConfig,
    StageName,
    stage_dependencies,
)
from rlfactory.experiments.context import ProgressFactory, RunContext, StageBlockedError
from rlfactory.experiments.stages import STAGES
from rlfactory.policies.factory import MockPolicySpec, describe_policy
from rlfactory.rewards.registry import RewardRegistry
from rlfactory.runtime.git import detect_code_version
from rlfactory.tasks.registry import TaskRegistry
from rlfactory.telemetry.logging import get_logger, log_context, log_event
from rlfactory.telemetry.system import ResourceMonitor, collect_hardware_info
from rlfactory.verifiers.registry import VerifierRegistry

logger = get_logger("experiments")
ALWAYS_RUN: frozenset[str] = frozenset({"report"})


class StageRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    status: Literal["completed", "failed", "blocked"]
    fingerprint: str
    started_at: datetime
    finished_at: datetime
    duration_s: float
    cached: bool = False
    outputs: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, JsonValue] = Field(default_factory=dict)
    resources: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class StagePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    depends_on: tuple[str, ...]
    description: str


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment: str
    stages: tuple[StagePlan, ...]
    train_tasks: int
    eval_tasks: int
    rollout_trajectories: int
    evaluation_trajectories: int
    datasets: tuple[str, ...]
    training_backend: str | None
    training_method: str | None
    external_requirements: tuple[str, ...]
    estimated_disk_mb: float
    simulated: bool


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash of every RLFactory source file: stage caches are invalidated by any code change."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _slice(config: ExperimentConfig, stage: str) -> Any:
    """Config sections that determine a stage's outputs (beyond its dependencies)."""
    scoring = {"sandbox": config.sandbox, "verifiers": config.verifiers, "rewards": config.rewards}
    slices: dict[str, Any] = {
        "tasks": {"tasks": config.tasks, **scoring},
        "rollouts": {"policy": config.policy, "rollouts": config.rollouts, **scoring},
        "build_sft": config.datasets.sft,
        "build_dpo": {"dpo": config.datasets.dpo, "training": config.training},
        "build_grpo": config.datasets.grpo,
        "train": config.training,
        "evaluate_base": {"policy": config.policy, "evaluation": config.evaluation, **scoring},
        "evaluate_candidate": {"evaluation": config.evaluation, "policy": config.policy, **scoring},
        "gate": config.evaluation.gates,
        "analysis": {},
        "flywheel": {
            "flywheel": config.flywheel,
            "policy": config.policy,
            "evaluation": config.evaluation,
            "training": config.training,
            **scoring,
        },
        "report": {},
    }
    return slices[stage]


_DESCRIPTIONS = {
    "tasks": "materialise, validate, decontaminate and split task suites",
    "rollouts": "generate verified, rewarded trajectories with the base policy",
    "build_sft": "build SFT prompt-completion data from successful trajectories",
    "build_dpo": "build preference pairs within rollout groups",
    "build_grpo": "build prompt-only RLVR data (saturated tasks dropped)",
    "train": "prepare and run training through the configured backend",
    "evaluate_base": "evaluate the base policy on the held-out suite",
    "evaluate_candidate": "evaluate the trained checkpoint on the same suite",
    "gate": "apply regression gates and record the promotion decision",
    "analysis": "failure analysis, reward calibration and reward-hacking flags",
    "flywheel": "failure-driven mutation -> rollouts -> training -> re-evaluation loop",
    "report": "render report.md and figures",
}


class ExperimentRunner:
    def __init__(self, settings: Settings, *, progress: ProgressFactory | None = None) -> None:
        self.settings = settings
        self.progress = progress

    # ------------------------------------------------------------------ planning
    def plan(self, config: ExperimentConfig) -> ExperimentPlan:
        registry = TaskRegistry(self.settings.config_dir)
        cap = config.tasks.max_tasks_per_suite

        def count(names: tuple[str, ...]) -> int:
            return sum(
                min(len(registry.load(n)), cap) if cap else len(registry.load(n)) for n in names
            )

        train, evaluation = count(config.tasks.train_suites), count(config.tasks.eval_suites)
        rollouts = train * config.rollouts.samples_per_task if "rollouts" in config.stages else 0
        evaluations = sum(
            evaluation * config.evaluation.samples_per_task
            for stage in ("evaluate_base", "evaluate_candidate")
            if stage in config.stages
        )
        requirements: list[str] = []
        if config.policy.kind not in ("mock",):
            requirements.append(
                f"model server or local model for policy kind '{config.policy.kind}'"
            )
        if config.sandbox.kind == "docker":
            requirements.append("Docker daemon and sandbox image")
        training = config.training
        if training is not None and training.backend != "mock":
            requirements.append(
                f"{training.backend} on a GPU host (remote: {training.remote_host or 'not set'})"
            )
        simulated = isinstance(config.policy, MockPolicySpec) or (
            training is not None and training.backend == "mock"
        )
        return ExperimentPlan(
            experiment=config.name,
            stages=tuple(
                StagePlan(
                    name=s, depends_on=stage_dependencies(s, config), description=_DESCRIPTIONS[s]
                )
                for s in STAGE_ORDER
                if s in config.stages
            ),
            train_tasks=train,
            eval_tasks=evaluation,
            rollout_trajectories=rollouts,
            evaluation_trajectories=evaluations,
            datasets=tuple(
                s.removeprefix("build_") for s in config.stages if s.startswith("build_")
            ),
            training_backend=training.backend if training else None,
            training_method=training.method.value if training else None,
            external_requirements=tuple(requirements) or ("none (CPU-only, offline)",),
            estimated_disk_mb=round((rollouts + evaluations) * 8_000 / 1e6 + 1.0, 1),
            simulated=simulated,
        )

    # ------------------------------------------------------------------ run creation
    def new_run_id(self, config: ExperimentConfig) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"{config.name}-{stamp}-{stable_hash(config.model_dump(mode='json'), length=6)}"

    def create_run(self, config: ExperimentConfig, run_id: str | None = None) -> Path:
        run_id = run_id or self.new_run_id(config)
        run_dir = self.settings.run_dir(run_id)
        if (run_dir / "config.yaml").exists():
            raise ExperimentError(f"run {run_id} already exists; use `experiment resume {run_id}`")
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(config, run_dir / "config.yaml")
        write_json(self.manifest(config, run_id), run_dir / "manifest.json")
        return run_dir

    def manifest(self, config: ExperimentConfig, run_id: str) -> RunManifest:
        registry = TaskRegistry(self.settings.config_dir)
        suites = [
            registry.load(name) for name in (*config.tasks.train_suites, *config.tasks.eval_suites)
        ]
        tasks = [task for suite in suites for task in suite.tasks]
        verifiers = VerifierRegistry(config.verifiers)
        verifier_names = sorted(
            {config.rollouts.verifier_overrides.get(t.domain, t.verifier) for t in tasks}
        )
        environments = sorted({t.environment for t in tasks})
        from rlfactory.environments.registry import default_environment_registry

        env_registry = default_environment_registry()
        reward = RewardRegistry(config.rewards).get(config.rollouts.reward)
        training = config.training
        return RunManifest(
            run_id=run_id,
            experiment=config.name,
            created_at=datetime.now(UTC),
            code=detect_code_version(),
            seed=config.seed,
            task_suites=tuple(s.ref() for s in suites),
            environments=tuple(env_registry.spec(name).ref for name in environments),
            verifiers=tuple(
                ComponentRef(name=v.name, version=v.version)
                for v in (verifiers.get(n) for n in verifier_names)
            ),
            rewards=(ComponentRef(name=reward.name, version=reward.version),),
            policy=describe_policy(config.policy),
            generation=config.rollouts.generation,
            training_backend=training.backend if training else None,
            hardware=collect_hardware_info(),
            config_fingerprint=stable_hash(config.model_dump(mode="json"), length=16),
            simulated=self.plan(config).simulated,
        )

    # ------------------------------------------------------------------ execution
    @staticmethod
    def _record_path(run_dir: Path, stage: str) -> Path:
        return run_dir / "stages" / f"{stage}.json"

    def _load_record(self, run_dir: Path, stage: str) -> StageRecord | None:
        path = self._record_path(run_dir, stage)
        return read_json_model(path, StageRecord) if path.exists() else None

    def load_config(self, run_dir: Path) -> ExperimentConfig:
        return load_model(run_dir / "config.yaml", ExperimentConfig)

    async def run(
        self,
        run_dir: Path,
        *,
        only: Collection[str] | None = None,
        until: str | None = None,
        force: Collection[str] = (),
    ) -> list[StageRecord]:
        config = self.load_config(run_dir)
        unknown = (set(only or ()) | set(force) | ({until} if until else set())) - set(STAGE_ORDER)
        if unknown:
            raise ExperimentError(f"unknown stages: {sorted(unknown)}")
        ctx = RunContext(
            run_id=run_dir.name,
            run_dir=run_dir,
            config=config,
            settings=self.settings,
            code=detect_code_version(),
            progress=self.progress or (lambda _stage: None),
        )
        fingerprints: dict[str, str] = {}
        records: list[StageRecord] = []
        for stage in STAGE_ORDER:
            if stage not in config.stages:
                continue
            dependencies = stage_dependencies(stage, config)
            previous = self._load_record(run_dir, stage)
            fingerprint = stable_hash(
                stage,
                _slice(config, stage),
                [fingerprints.get(d, "missing") for d in dependencies],
                code_fingerprint(),
            )
            selected = only is None or stage in only
            if not selected:
                if previous is not None and previous.status == "completed":
                    fingerprints[stage] = previous.fingerprint
                    ctx.outputs[stage] = previous.outputs
                if until == stage:
                    break
                continue
            missing = [d for d in dependencies if d not in fingerprints]
            if missing:
                raise ExperimentError(f"stage {stage!r} needs completed stages {missing}")
            cacheable = stage not in ALWAYS_RUN and stage not in force
            if (
                cacheable
                and previous is not None
                and previous.status == "completed"
                and previous.fingerprint == fingerprint
            ):
                records.append(previous.model_copy(update={"cached": True}))
                fingerprints[stage] = fingerprint
                ctx.outputs[stage] = previous.outputs
                log_event(logger, "stage cached", stage=stage, run_id=ctx.run_id)
            else:
                record = await self._execute(ctx, stage, fingerprint)
                records.append(record)
                if record.status != "completed":
                    if record.status == "blocked":
                        log_event(logger, "stage blocked", stage=stage, detail=record.error)
                        break
                    raise ExperimentError(f"stage {stage!r} failed: {record.error}")
                fingerprints[stage] = fingerprint
                ctx.outputs[stage] = record.outputs
            if until == stage:
                break
        return records

    async def _execute(self, ctx: RunContext, stage: StageName, fingerprint: str) -> StageRecord:
        started = datetime.now(UTC)
        monitor = ResourceMonitor(ctx.run_dir)
        status: Literal["completed", "failed", "blocked"] = "completed"
        error: str | None = None
        outputs: dict[str, str] = {}
        summary: dict[str, JsonValue] = {}
        with log_context(run_id=ctx.run_id, stage=stage), monitor:
            log_event(logger, "stage started")
            try:
                result = await STAGES[stage](ctx)
                outputs, summary = result.outputs, result.summary
            except StageBlockedError as exc:
                status, error = "blocked", str(exc)
            except Exception as exc:  # noqa: BLE001 - recorded on the stage, then re-raised by run()
                status = "failed"
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"[:4000]
        finished = datetime.now(UTC)
        record = StageRecord(
            name=stage,
            status=status,
            fingerprint=fingerprint,
            started_at=started,
            finished_at=finished,
            duration_s=round((finished - started).total_seconds(), 3),
            outputs=outputs,
            summary=json.loads(json.dumps(summary, default=str)),
            resources=monitor.usage.as_dict() if monitor.usage else {},
            error=error,
        )
        write_json(record, self._record_path(ctx.run_dir, stage))
        log_event(
            logger, "stage finished", stage=stage, status=status, duration_s=record.duration_s
        )
        return record
