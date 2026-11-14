"""Mock training backend.

It simulates training state, a decaying loss curve, a cosine learning-rate schedule, periodic
checkpoint events, injected failures and resume-from-checkpoint. It trains **nothing**.

So that evaluation and regression gates see a *changed* policy, the final (fake) checkpoint carries
a ``simulated_policy_profile``: the base mock-policy profile adjusted by a documented, configurable
function of how many training examples each domain received. These numbers are a simulation knob,
not evidence; every artefact derived from them is labelled ``simulated``.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from random import Random

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.config.loader import read_json_model, write_json
from rlfactory.domain.common import stable_seed
from rlfactory.domain.model import CheckpointMetadata, CheckpointReference
from rlfactory.domain.task import TaskDomain
from rlfactory.domain.training import (
    CheckpointEvent,
    LaunchMode,
    PreparedTrainingRun,
    TrainingHandle,
    TrainingMethod,
    TrainingMetricsPoint,
    TrainingRun,
    TrainingState,
    TrainingStatus,
    TrainingSummary,
)
from rlfactory.policies.mock import MockPolicyProfile
from rlfactory.runtime.git import detect_code_version
from rlfactory.training.base import (
    RUN_CONFIG_FILE,
    SUMMARY_FILE,
    append_metrics,
    read_status,
    status_from_workdir,
    write_status,
)

CHECKPOINT_FILE = "checkpoint.json"
STATE_FILE = "mock_state.json"


class MockTrainingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    simulate_failure_at_step: int | None = Field(default=None, ge=1)
    sleep_per_step_s: float = Field(default=0.0, ge=0.0)
    base_profile: MockPolicyProfile = MockPolicyProfile()
    max_gain: float = Field(default=0.25, ge=0.0, le=1.0)
    saturation_examples: int = Field(default=50, ge=1)
    method_gain: dict[TrainingMethod, float] = Field(
        default_factory=lambda: {
            TrainingMethod.SFT: 1.0,
            TrainingMethod.DPO: 0.8,
            TrainingMethod.GRPO: 1.2,
            TrainingMethod.REWARD_MODEL: 0.0,
        }
    )


class MockCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: CheckpointMetadata
    simulated_policy_profile: MockPolicyProfile


class _State(BaseModel):
    step: int = 0
    failure_injected: bool = False


def _domain_counts(dataset_path: Path) -> Counter[str]:
    records = dataset_path.parent / "records.jsonl"
    counts: Counter[str] = Counter()
    if not records.exists():
        return counts
    for line in records.read_text(encoding="utf-8").splitlines():
        if line.strip():
            lineage = json.loads(line).get("lineage", {})
            counts[str(lineage.get("domain", "unknown"))] += 1
    return counts


def simulated_profile(run: TrainingRun, settings: MockTrainingSettings) -> MockPolicyProfile:
    """Base profile + saturating, per-domain gain. A simulation knob, not a learning model."""
    base = settings.base_profile
    method_gain = settings.method_gain.get(run.method, 0.0)
    counts = _domain_counts(Path(run.dataset.path))
    total = sum(counts.values()) or run.dataset.num_records
    scale = math.log1p(settings.saturation_examples)
    domain_p: dict[TaskDomain, float] = {}
    for domain in TaskDomain:
        n = counts.get(domain.value, 0) if counts else total / len(TaskDomain)
        gain = settings.max_gain * method_gain * min(1.0, math.log1p(n) / scale)
        p = base.p_correct(domain)
        domain_p[domain] = round(min(0.98, p + gain * (1.0 - p)), 4)
    malformed = base.malformed_probability * (
        1.0 - 0.5 * method_gain * min(1.0, total / settings.saturation_examples)
    )
    return base.model_copy(
        update={
            "domain_correct_probability": domain_p,
            "malformed_probability": round(max(0.0, malformed), 4),
        }
    )


def load_mock_checkpoint(path: Path) -> MockCheckpoint:
    return read_json_model(path / CHECKPOINT_FILE if path.is_dir() else path, MockCheckpoint)


class MockTrainingBackend:
    name = "mock"

    def __init__(self, workdir_root: Path, settings: MockTrainingSettings | None = None) -> None:
        self.workdir_root = workdir_root
        self.settings = settings or MockTrainingSettings()

    @staticmethod
    def total_steps(run: TrainingRun) -> int:
        config = run.config
        if config.max_steps > 0:
            return config.max_steps
        per_step = config.per_device_train_batch_size * config.gradient_accumulation_steps
        return max(
            1, math.ceil(run.dataset.num_records / per_step) * math.ceil(config.num_train_epochs)
        )

    def prepare(self, run: TrainingRun) -> PreparedTrainingRun:
        workdir = self.workdir_root / run.run_id
        workdir.mkdir(parents=True, exist_ok=True)
        write_json(run, workdir / RUN_CONFIG_FILE)
        return PreparedTrainingRun(
            run=run,
            backend=self.name,
            workdir=str(workdir),
            command=("<in-process mock training>",),
            files=(RUN_CONFIG_FILE,),
            framework_version=None,
            notes=("Simulated backend: no model weights are trained or written.",),
        )

    def _curve(self, run: TrainingRun, step: int, total: int) -> tuple[float, float]:
        rng = Random(stable_seed(run.run_id, run.seed, step))
        loss = 0.35 + 1.6 * math.exp(-3.0 * step / max(1, total)) + rng.uniform(-0.03, 0.03)
        warmup = max(1, int(run.config.warmup_ratio * total))
        if step < warmup:
            lr = run.config.learning_rate * step / warmup
        else:
            progress = (step - warmup) / max(1, total - warmup)
            lr = run.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        return round(loss, 5), lr

    def _checkpoint(self, prepared: PreparedTrainingRun, step: int, final: bool) -> CheckpointEvent:
        run = prepared.run
        directory = Path(prepared.workdir) / "checkpoints" / f"step-{step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        reference = CheckpointReference(
            checkpoint_id=f"{run.run_id}-step{step}",
            base_model=run.model,
            adapter_uri=str(directory),
            method=run.method.value,
            training_run_id=run.run_id,
            step=step,
            simulated=True,
        )
        metadata = CheckpointMetadata(
            checkpoint=reference,
            training_method=run.method.value,
            dataset=run.dataset.name,
            dataset_fingerprint=run.dataset.fingerprint,
            git_sha=detect_code_version().git_sha,
            training_config=run.config.model_dump(mode="json"),
            framework="rlfactory-mock",
            framework_version=None,
            metrics={"final": 1.0 if final else 0.0},
            created_at=datetime.now(UTC),
        )
        profile = simulated_profile(run, self.settings) if final else self.settings.base_profile
        write_json(
            MockCheckpoint(metadata=metadata, simulated_policy_profile=profile),
            directory / CHECKPOINT_FILE,
        )
        return CheckpointEvent(step=step, path=str(directory), created_at=datetime.now(UTC))

    def launch(self, run: PreparedTrainingRun, *, resume: bool = True) -> TrainingHandle:
        prepared = run
        workdir = Path(prepared.workdir)
        handle = TrainingHandle(
            handle_id=f"mock-{prepared.run.run_id}",
            backend=self.name,
            run_id=prepared.run.run_id,
            workdir=str(workdir),
            launch_mode=LaunchMode.LOCAL,
            started_at=datetime.now(UTC),
        )
        state_path = workdir / STATE_FILE
        state = read_json_model(state_path, _State) if resume and state_path.exists() else _State()
        previous = read_status(workdir) if resume else None
        checkpoints = list(previous.checkpoints) if previous is not None else []
        total = self.total_steps(prepared.run)
        save_every = prepared.run.config.save_steps
        log_every = prepared.run.config.logging_steps
        started = time.perf_counter()
        resumed_from = state.step if state.step > 0 else None
        latest: TrainingMetricsPoint | None = None
        for step in range(state.step + 1, total + 1):
            if self.settings.simulate_failure_at_step == step and not state.failure_injected:
                state = _State(step=state.step, failure_injected=True)
                write_json(state, state_path)
                failed = TrainingStatus(
                    handle_id=handle.handle_id,
                    state=TrainingState.FAILED,
                    step=step,
                    total_steps=total,
                    latest=latest,
                    checkpoints=tuple(checkpoints),
                    error=f"simulated failure at step {step}",
                    simulated=True,
                )
                write_status(workdir, failed)
                self._write_summary(prepared, failed, started, resumed_from)
                return handle
            loss, lr = self._curve(prepared.run, step, total)
            latest = TrainingMetricsPoint(
                step=step, loss=loss, learning_rate=lr, wall_time_s=time.perf_counter() - started
            )
            if step % log_every == 0 or step == total:
                append_metrics(workdir, latest)
            if step % save_every == 0 or step == total:
                checkpoints.append(self._checkpoint(prepared, step, final=step == total))
                state = _State(step=step, failure_injected=state.failure_injected)
                write_json(state, state_path)
            if self.settings.sleep_per_step_s:
                time.sleep(self.settings.sleep_per_step_s)
        done = TrainingStatus(
            handle_id=handle.handle_id,
            state=TrainingState.SUCCEEDED,
            step=total,
            total_steps=total,
            latest=latest,
            checkpoints=tuple(checkpoints),
            simulated=True,
        )
        write_status(workdir, done)
        self._write_summary(prepared, done, started, resumed_from)
        return handle

    def _write_summary(
        self,
        prepared: PreparedTrainingRun,
        status: TrainingStatus,
        started: float,
        resumed_from: int | None,
    ) -> None:
        summary = TrainingSummary(
            run_id=prepared.run.run_id,
            backend=self.name,
            method=prepared.run.method,
            state=status.state,
            steps=status.step,
            final_loss=status.latest.loss if status.latest else None,
            wall_time_s=round(time.perf_counter() - started, 4),
            tokens_seen=None,
            peak_vram_gb=None,
            checkpoints=status.checkpoints,
            framework_version=None,
            dataset=prepared.run.dataset,
            model=prepared.run.model,
            simulated=True,
            resumed_from_step=resumed_from,
            error=status.error,
        )
        write_json(summary, Path(prepared.workdir) / SUMMARY_FILE)

    def inspect(self, handle: TrainingHandle) -> TrainingStatus:
        return status_from_workdir(handle)
