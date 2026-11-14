"""Docker sandbox: stronger (but still limited) isolation for untrusted generated code.

Each execution runs in a fresh container with: no network, a non-root numeric user, a read-only
root filesystem, a read-only bind mount of a temporary work directory, a small ``noexec`` tmpfs,
dropped capabilities, ``no-new-privileges``, and memory/CPU/PID/file-size limits. The Docker
socket is never mounted into the container. Container isolation shares the host kernel, so a
kernel or runtime vulnerability can still break out; see ``docs/security/sandbox.md``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from rlfactory.domain.trajectory import ExecutionStatus
from rlfactory.errors import SandboxError
from rlfactory.sandbox.base import (
    ExecutionRequest,
    ExecutionResult,
    SandboxCapabilities,
    Stopwatch,
    read_capped,
    write_request_files,
)

DEFAULT_IMAGE = "python:3.12-slim"
_DOCKER_ERROR_CODES = {125, 126, 127}
_OOM_EXIT_CODE = 137


class DockerSandbox:
    name = "docker"

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        docker_binary: str = "docker",
        cpus: float = 1.0,
        user: str = "65534:65534",
        startup_allowance_s: float = 15.0,
        poll_interval_s: float = 0.02,
    ) -> None:
        self.image = image
        self.docker = shutil.which(docker_binary) or docker_binary
        self.cpus = cpus
        self.user = user
        self.startup_allowance_s = startup_allowance_s
        self.poll_interval_s = poll_interval_s

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            name=self.name,
            process_isolation=True,
            filesystem_isolation="container_read_only",
            network_isolation=True,
            memory_limit="enforced",
            cpu_limit="enforced",
            notes=(
                "Containers share the host kernel; this is defence in depth, not a VM boundary.",
                "Wall-clock limits include container start-up allowance.",
            ),
        )

    def _docker(self, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.docker, *args], capture_output=True, text=True, timeout=timeout, check=False
        )

    def available(self) -> bool:
        try:
            return (
                self._docker("info", "--format", "{{.ServerVersion}}", timeout=15).returncode == 0
            )
        except (OSError, subprocess.SubprocessError):
            return False

    def image_present(self) -> bool:
        try:
            return self._docker("image", "inspect", self.image, timeout=15).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _command(self, request: ExecutionRequest, workdir: Path, container: str) -> list[str]:
        limits = request.limits
        return [
            self.docker,
            "run",
            "--rm",
            "-i",
            # Docker's init as PID 1: the program is not init, so default-action signals such as
            # SIGXCPU apply normally, and zombie processes from the test harness are reaped.
            "--init",
            "--name",
            container,
            "--network",
            "none",
            "--user",
            self.user,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",  # noqa: S108 - path inside the container
            "--memory",
            f"{limits.memory_mb}m",
            "--memory-swap",
            f"{limits.memory_mb}m",
            "--cpus",
            str(self.cpus),
            "--pids-limit",
            str(limits.max_processes),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--ulimit",
            f"fsize={limits.max_file_bytes}:{limits.max_file_bytes}",
            "--ulimit",
            # soft < hard: CPU exhaustion arrives as SIGXCPU (exit 152 -> timeout), not SIGKILL.
            f"cpu={limits.cpu_time_s}:{limits.cpu_time_s + 1}",
            "--mount",
            f"type=bind,source={workdir},target=/sandbox,readonly",
            "--workdir",
            "/sandbox",
            "--env",
            "PYTHONHASHSEED=0",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "HOME=/tmp",
            self.image,
            "python",
            "-B",
            "-S",
            "-P",
            "-X",
            "utf8",
            request.entrypoint,
        ]

    def _kill(self, container: str, process: subprocess.Popen[bytes]) -> None:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            self._docker("kill", container, timeout=20)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        limits = request.limits
        container = f"rlf-sbx-{uuid.uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory(prefix="rlf-docker-") as tmp:
            root = Path(tmp)
            workdir = root / "work"
            workdir.mkdir()
            write_request_files(request, workdir)
            if os.name == "posix":
                # The container user is unprivileged; make inputs world-readable (never writable).
                workdir.chmod(0o755)
                for path in workdir.rglob("*"):
                    path.chmod(0o755 if path.is_dir() else 0o644)
            stdout_path, stderr_path = root / "stdout.bin", root / "stderr.bin"
            clock = Stopwatch()
            deadline = limits.wall_time_s + self.startup_allowance_s
            forced: ExecutionStatus | None = None
            with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
                try:
                    process = subprocess.Popen(
                        self._command(request, workdir, container),
                        stdin=subprocess.PIPE,
                        stdout=out,
                        stderr=err,
                    )
                except OSError as exc:
                    raise SandboxError(f"failed to invoke docker: {exc}") from exc
                feeder = threading.Thread(
                    target=self._feed, args=(process, request.stdin.encode("utf-8")), daemon=True
                )
                feeder.start()
                while True:
                    try:
                        process.wait(timeout=self.poll_interval_s)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if clock.elapsed() > deadline:
                        forced = ExecutionStatus.TIMEOUT
                    elif any(
                        p.stat().st_size > limits.max_output_bytes
                        for p in (stdout_path, stderr_path)
                    ):
                        forced = ExecutionStatus.OUTPUT_LIMIT
                    if forced is not None:
                        self._kill(container, process)
                        break
                feeder.join(timeout=1.0)
            duration = clock.elapsed()
            stdout, stdout_truncated = read_capped(stdout_path, limits.max_output_bytes)
            stderr, stderr_truncated = read_capped(stderr_path, limits.max_output_bytes)
        exit_code = process.returncode
        if forced is None and exit_code in _DOCKER_ERROR_CODES:
            raise SandboxError(f"docker failed (exit {exit_code}): {stderr.strip()[:500]}")
        if forced is not None:
            status = forced
        elif exit_code == 0:
            status = ExecutionStatus.OK
        elif exit_code == _OOM_EXIT_CODE or "MemoryError" in stderr:
            status = ExecutionStatus.MEMORY_EXCEEDED
        elif exit_code == 153:  # 128 + SIGXFSZ (25)
            status = ExecutionStatus.FILE_LIMIT
        elif exit_code == 152:  # 128 + SIGXCPU (24)
            status = ExecutionStatus.TIMEOUT
        else:
            status = ExecutionStatus.RUNTIME_ERROR
        return ExecutionResult(
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            duration_s=duration,
            sandbox=self.name,
            detail=f"image={self.image}",
        )

    @staticmethod
    def _feed(process: subprocess.Popen[bytes], data: bytes) -> None:
        stream = process.stdin
        if stream is None:
            return
        try:
            stream.write(data)
            stream.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
