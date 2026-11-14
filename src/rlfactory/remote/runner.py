"""Remote training workflow: inspect GPU -> preflight -> prepare env -> sync -> smoke -> launch ->
monitor -> fetch lightweight results. It works against an existing SSH-reachable machine and never
creates, starts or pays for cloud resources."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rlfactory.config.loader import write_json
from rlfactory.domain.training import (
    LaunchMode,
    PreparedTrainingRun,
    TrainingHandle,
    TrainingState,
    TrainingStatus,
)
from rlfactory.errors import RemoteExecutionError
from rlfactory.remote.commands import CommandResult
from rlfactory.remote.host import RemoteHost
from rlfactory.remote.ssh import SSHClient
from rlfactory.telemetry.system import GPUInfo, parse_nvidia_smi
from rlfactory.training.base import (
    EXIT_STATUS_FILE,
    METRICS_FILE,
    RUN_CONFIG_FILE,
    STATUS_FILE,
    SUMMARY_FILE,
)
from rlfactory.training.memory import MemoryEstimate

LIGHTWEIGHT_RESULTS = (STATUS_FILE, METRICS_FILE, SUMMARY_FILE, EXIT_STATUS_FILE, "launch.out")
CODE_PATHS = ("src", "configs", "scripts", "pyproject.toml", "uv.lock", "README.md", "LICENSE")
# Only the job is backgrounded, with stdin/stdout/stderr detached from the SSH channel: the cd and
# activation chain runs in the foreground, the session closes as soon as `echo $!` prints the
# job's PID, and nohup keeps the job alive after logout. Backgrounding the whole `&&` chain
# instead keeps the channel open under bash until the job ends, so the launch would time out
# while training was in fact running.
LAUNCH_COMMAND = "{ nohup bash launch.sh > launch.out 2>&1 < /dev/null & echo $!; }"


@dataclass
class PreflightReport:
    ok: bool
    gpus: list[GPUInfo]
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class RemoteTrainingRunner:
    def __init__(self, host: RemoteHost, ssh: SSHClient) -> None:
        self.host = host
        self.ssh = ssh

    def remote_dir(self, run_id: str) -> str:
        return f"{self.host.workspace.rstrip('/')}/{run_id}"

    @staticmethod
    def _require(result: CommandResult, what: str) -> CommandResult:
        if not result.ok:
            raise RemoteExecutionError(
                f"{what} failed (exit {result.returncode}): {result.stderr.strip()[:400]}"
            )
        return result

    def check_gpu(self) -> list[GPUInfo]:
        result = self.ssh.run(
            "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits",
            timeout=60,
        )
        return parse_nvidia_smi(result.stdout) if result.ok else []

    def preflight(
        self, prepared: PreparedTrainingRun, estimate: MemoryEstimate | None
    ) -> PreflightReport:
        gpus = self.check_gpu()
        report = PreflightReport(ok=True, gpus=gpus)
        if not gpus:
            report.problems.append("no NVIDIA GPU visible via nvidia-smi on the remote host")
        if not prepared.run.model.is_pinned:
            report.problems.append(f"model {prepared.run.model.name} revision is not pinned")
        if estimate is not None and gpus:
            free = min(g.memory_total_gb - g.memory_used_gb for g in gpus)
            report.notes.append(
                f"estimated {estimate.total_gb:.1f} GB vs {free:.1f} GB free per GPU"
            )
            if not estimate.fits(free):
                report.problems.append(
                    f"memory estimate {estimate.total_gb:.1f} GB exceeds 90% of free GPU memory ({free:.1f} GB)"
                )
        report.notes.append("estimates are heuristic; the smoke step is the real check")
        report.ok = not report.problems
        return report

    def prepare_environment(self) -> list[CommandResult]:
        return [
            self._require(
                self.ssh.run(
                    self.ssh.in_dir(self.host.code_dir, command, activate=False), timeout=3600
                ),
                f"setup command {command!r}",
            )
            for command in self.host.environment.setup_commands
        ]

    def sync_code(self, repo_root: Path) -> list[CommandResult]:
        results = []
        for relative in CODE_PATHS:
            local = repo_root / relative
            if local.is_dir():
                results.append(
                    self._require(
                        self.ssh.upload(local, f"{self.host.code_dir}/{relative}"),
                        f"sync {relative}",
                    )
                )
            elif local.is_file():
                self._require(
                    self.ssh.run(f"mkdir -p {shlex.quote(self.host.code_dir)}", timeout=60), "mkdir"
                )
                results.append(
                    self._require(
                        self.ssh.runner.run(
                            [
                                "scp",
                                "-q",
                                "-P",
                                str(self.host.port),
                                "-o",
                                "BatchMode=yes",
                                local.as_posix(),
                                f"{self.host.target}:{self.host.code_dir}/{relative}",
                            ],
                            timeout=600,
                        ),
                        f"sync {relative}",
                    )
                )
        return results

    def sync_run(self, prepared: PreparedTrainingRun) -> str:
        remote = self.remote_dir(prepared.run.run_id)
        self._require(self.ssh.upload(Path(prepared.workdir), remote), "sync run directory")
        return remote

    def smoke(
        self, prepared: PreparedTrainingRun, *, max_steps: int = 2, timeout_s: float = 1800
    ) -> CommandResult:
        """Run a few optimizer steps on the real GPU before committing to a full run (TRL)."""
        workdir = Path(prepared.workdir)
        spec = json.loads((workdir / RUN_CONFIG_FILE).read_text(encoding="utf-8"))
        spec["config_kwargs"].update(
            max_steps=max_steps, save_steps=10**6, output_dir="smoke_outputs"
        )
        smoke_name = "run_config.smoke.json"
        write_json(spec, workdir / smoke_name)
        remote = self.sync_run(prepared)
        command = f"python -m rlfactory.training.trl.entrypoint --run-config {smoke_name}"
        return self._require(
            self.ssh.run(self.ssh.in_dir(remote, command), timeout=timeout_s), "smoke step"
        )

    def launch(self, prepared: PreparedTrainingRun) -> TrainingHandle:
        remote = self.remote_dir(prepared.run.run_id)
        result = self._require(
            self.ssh.run(self.ssh.in_dir(remote, LAUNCH_COMMAND), timeout=120),
            "launch",
        )
        pid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "unknown"
        return TrainingHandle(
            handle_id=f"{prepared.backend}-{prepared.run.run_id}-pid{pid}",
            backend=prepared.backend,
            run_id=prepared.run.run_id,
            workdir=remote,
            launch_mode=LaunchMode.REMOTE,
            remote_host=self.host.name,
            started_at=datetime.now(UTC),
        )

    def status(self, handle: TrainingHandle) -> TrainingStatus:
        remote = handle.workdir
        result = self.ssh.run(
            f"cd {shlex.quote(remote)} && cat {STATUS_FILE} 2>/dev/null; echo '@@'; "
            f"cat {EXIT_STATUS_FILE} 2>/dev/null; echo '@@'; tail -n 5 train.log 2>/dev/null",
            timeout=60,
        )
        if not result.ok:
            return TrainingStatus(
                handle_id=handle.handle_id, state=TrainingState.RUNNING, error="host unreachable"
            )
        status_text, exit_text, _tail = [*result.stdout.split("@@"), "", "", ""][:3]
        if status_text.strip():
            try:
                return TrainingStatus.model_validate_json(status_text.strip())
            except ValueError:
                pass
        if exit_text.strip():
            code = int(json.loads(exit_text).get("exit_code", 1))
            return TrainingStatus(
                handle_id=handle.handle_id,
                state=TrainingState.SUCCEEDED if code == 0 else TrainingState.FAILED,
                error=None if code == 0 else f"exit code {code}",
            )
        return TrainingStatus(handle_id=handle.handle_id, state=TrainingState.RUNNING)

    def fetch_results(
        self, handle: TrainingHandle, local_dir: Path, *, include_adapter_config: bool = True
    ) -> list[str]:
        """Download small result files only. Weights stay on the GPU host unless fetched manually."""
        fetched: list[str] = []
        names = list(LIGHTWEIGHT_RESULTS)
        if include_adapter_config:
            names += [
                "outputs/final/rlfactory_checkpoint.json",
                "outputs/final/adapter_config.json",
            ]
        for name in names:
            result = self.ssh.download(f"{handle.workdir}/{name}", local_dir / Path(name).parent)
            if result.ok:
                fetched.append(name)
        return fetched
