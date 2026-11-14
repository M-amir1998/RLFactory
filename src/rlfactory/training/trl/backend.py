"""TRL training backend: prepares self-contained run directories and launches them."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.config.loader import write_json
from rlfactory.domain.training import (
    GRPOParams,
    LaunchMode,
    PreparedTrainingRun,
    TrainingHandle,
    TrainingRun,
    TrainingStatus,
)
from rlfactory.errors import BackendUnavailableError, ConfigError, TrainingBackendError
from rlfactory.training.base import (
    LAUNCH_SCRIPT,
    RUN_CONFIG_FILE,
    render_launch_script,
    status_from_workdir,
)
from rlfactory.training.trl.config_mapping import (
    CONFIG_CLASSES,
    EXPECTED_SCHEMAS,
    TARGET_TRL_VERSION,
    TRAINER_CLASSES,
    build_config_kwargs,
    check_grpo_batching,
)
from rlfactory.training.trl.spec import TRLRunSpec

ENTRYPOINT_MODULE = "rlfactory.training.trl.entrypoint"


class TRLBackendSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    num_processes: int = Field(default=1, ge=1)
    mixed_precision: str = "bf16"
    cpu_smoke: bool = False
    """Tiny local CPU run (float32, no bf16, no vLLM). For wiring checks, never for results."""
    allow_unpinned_revision: bool = False


def _installed_trl() -> str | None:
    try:
        return importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError:
        return None


class TRLTrainingBackend:
    name = "trl"

    def __init__(self, workdir_root: Path, settings: TRLBackendSettings | None = None) -> None:
        self.workdir_root = workdir_root
        self.settings = settings or TRLBackendSettings()

    def _copy_dataset(self, source: Path, target_dir: Path) -> str:
        if not source.exists():
            raise ConfigError(f"dataset file {source} does not exist")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / source.name)
        for sidecar in ("records.jsonl", "dataset_card.json"):
            if (source.parent / sidecar).exists():
                shutil.copy2(source.parent / sidecar, target_dir / sidecar)
        return (target_dir / source.name).relative_to(target_dir.parent.parent).as_posix()

    def prepare(self, run: TrainingRun) -> PreparedTrainingRun:
        expected = EXPECTED_SCHEMAS[run.method]
        if run.dataset.schema_name != expected:
            raise ConfigError(
                f"{run.method.value} needs a {expected} dataset, got {run.dataset.schema_name}"
            )
        if (
            run.launch_mode is LaunchMode.REMOTE
            and not run.model.is_pinned
            and not self.settings.allow_unpinned_revision
        ):
            raise TrainingBackendError(
                f"model {run.model.name} has no pinned revision; pin a commit SHA before a remote run"
            )
        check_grpo_batching(run, self.settings.num_processes)
        workdir = self.workdir_root / run.run_id
        workdir.mkdir(parents=True, exist_ok=True)
        train_file = self._copy_dataset(Path(run.dataset.path), workdir / "data" / "train")
        eval_file = (
            self._copy_dataset(Path(run.eval_dataset.path), workdir / "data" / "eval")
            if run.eval_dataset is not None
            else None
        )
        params = run.config.method_params
        spec = TRLRunSpec(
            run=run,
            trainer_class=TRAINER_CLASSES[run.method],
            config_class=CONFIG_CLASSES[run.method],
            config_kwargs=build_config_kwargs(
                run, output_dir="outputs", cpu=self.settings.cpu_smoke
            ),
            train_file=train_file,
            eval_file=eval_file,
            reward_functions=params.reward_functions if isinstance(params, GRPOParams) else (),
            cpu=self.settings.cpu_smoke,
            target_trl_version=TARGET_TRL_VERSION,
        )
        write_json(spec, workdir / RUN_CONFIG_FILE)
        if self.settings.cpu_smoke:
            command: tuple[str, ...] = (
                "python",
                "-m",
                ENTRYPOINT_MODULE,
                "--run-config",
                RUN_CONFIG_FILE,
            )
        else:
            command = (
                "accelerate",
                "launch",
                "--num_processes",
                str(self.settings.num_processes),
                "--mixed_precision",
                self.settings.mixed_precision,
                "-m",
                ENTRYPOINT_MODULE,
                "--run-config",
                RUN_CONFIG_FILE,
            )
        environment = {"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
        (workdir / LAUNCH_SCRIPT).write_text(
            render_launch_script(command, environment), encoding="utf-8", newline="\n"
        )
        notes = [
            f"target TRL {TARGET_TRL_VERSION}; installed: {_installed_trl() or 'not installed'}"
        ]
        if not run.model.is_pinned:
            notes.append("model revision is not pinned")
        return PreparedTrainingRun(
            run=run,
            backend=self.name,
            workdir=str(workdir),
            command=("bash", LAUNCH_SCRIPT),
            files=(RUN_CONFIG_FILE, LAUNCH_SCRIPT, train_file, *([eval_file] if eval_file else [])),
            environment=environment,
            framework_version=_installed_trl(),
            notes=tuple(notes),
        )

    def launch(self, run: PreparedTrainingRun) -> TrainingHandle:
        """Local launch (blocking). Remote launches go through ``rlfactory.remote``."""
        if run.run.launch_mode is LaunchMode.REMOTE:
            raise TrainingBackendError(
                "remote TRL runs are launched via `rlfactory train --remote`"
            )
        if importlib.util.find_spec("trl") is None:
            raise BackendUnavailableError(
                "TRL is not installed in this environment; install the 'trl' extra or launch remotely"
            )
        workdir = Path(run.workdir)
        handle = TrainingHandle(
            handle_id=f"trl-{run.run.run_id}",
            backend=self.name,
            run_id=run.run.run_id,
            workdir=str(workdir),
            launch_mode=LaunchMode.LOCAL,
            started_at=datetime.now(UTC),
        )
        with (workdir / "train.log").open("ab") as log:
            completed = subprocess.run(
                [sys.executable, "-m", ENTRYPOINT_MODULE, "--run-config", RUN_CONFIG_FILE],
                cwd=workdir,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            status = self.inspect(handle)
            raise TrainingBackendError(
                f"TRL run failed (exit {completed.returncode}): {status.error}"
            )
        return handle

    def inspect(self, handle: TrainingHandle) -> TrainingStatus:
        return status_from_workdir(handle)
