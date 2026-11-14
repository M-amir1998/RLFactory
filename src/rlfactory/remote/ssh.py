"""Thin SSH/SCP/rsync wrapper. Remote command strings quote every variable part."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from rlfactory.remote.commands import CommandResult, CommandRunner
from rlfactory.remote.host import RemoteHost

SYNC_EXCLUDES = (
    ".git",
    ".venv",
    ".venv-*",
    "results",
    "__pycache__",
    "*.safetensors",
    "*.bin",
    "*.pt",
)


class SSHClient:
    def __init__(self, host: RemoteHost, runner: CommandRunner) -> None:
        self.host = host
        self.runner = runner

    def run(self, remote_command: str, *, timeout: float | None = 600.0) -> CommandResult:
        return self.runner.run(
            ["ssh", *self.host.ssh_options(), self.host.target, remote_command], timeout=timeout
        )

    def in_dir(self, directory: str, command: str, *, activate: bool = True) -> str:
        """Shell command that activates the environment *from the code checkout*, then runs
        ``command`` inside ``directory``. Every step is chained with ``&&``, so a failed ``cd`` or
        activation aborts instead of running the command in the wrong place or interpreter."""
        env = self.host.environment
        parts: list[str] = []
        if activate and env.activate:
            parts += [f"cd {shlex.quote(self.host.code_dir)}", env.activate]
        parts.append(f"cd {shlex.quote(directory)}")
        parts += [
            f"export {key}={shlex.quote(value)}" for key, value in sorted(env.env_vars.items())
        ]
        parts.append(command)
        return " && ".join(parts)

    def upload(
        self, local: Path, remote_dir: str, *, timeout: float | None = 1800.0
    ) -> CommandResult:
        mkdir = self.run(f"mkdir -p {shlex.quote(remote_dir)}", timeout=60)
        if not mkdir.ok:
            return mkdir
        source = local.as_posix().rstrip("/") + "/"
        if shutil.which("rsync"):
            excludes = [arg for pattern in SYNC_EXCLUDES for arg in ("--exclude", pattern)]
            ssh = shlex.join(["ssh", *self.host.ssh_options()])
            return self.runner.run(
                [
                    "rsync",
                    "-az",
                    "--delete",
                    *excludes,
                    "-e",
                    ssh,
                    source,
                    f"{self.host.target}:{remote_dir.rstrip('/')}/",
                ],
                timeout=timeout,
            )
        return self.runner.run(
            [
                "scp",
                "-r",
                "-q",
                "-P",
                str(self.host.port),
                "-o",
                "BatchMode=yes",
                f"{local.as_posix().rstrip('/')}/.",
                f"{self.host.target}:{remote_dir}",
            ],
            timeout=timeout,
        )

    def download(
        self, remote_path: str, local_dir: Path, *, timeout: float | None = 600.0
    ) -> CommandResult:
        local_dir.mkdir(parents=True, exist_ok=True)
        return self.runner.run(
            [
                "scp",
                "-q",
                "-P",
                str(self.host.port),
                "-o",
                "BatchMode=yes",
                f"{self.host.target}:{remote_path}",
                local_dir.as_posix(),
            ],
            timeout=timeout,
        )
