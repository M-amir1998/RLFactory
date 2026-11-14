"""Subprocess sandbox: a separate interpreter process with limits. NOT a security boundary.

What it does:
* runs the program in a fresh temporary directory with a minimal, secret-free environment;
* starts Python with ``-S`` (no site-packages: stdlib only), ``-P`` and ``-B``;
* enforces wall-clock time, output size and per-file size on every platform via a watchdog;
* on POSIX additionally applies ``RLIMIT_CPU``, ``RLIMIT_AS``, ``RLIMIT_FSIZE`` and
  ``RLIMIT_CORE`` through a tiny trusted bootstrap (no ``preexec_fn``, which is unsafe in
  multi-threaded parents) and kills the whole process group on timeout;
* on Windows enforces memory by polling the process tree's RSS (best effort).

What it does NOT do: block network access, hide the host filesystem, drop privileges, or contain
a deliberately malicious program. Use :class:`DockerSandbox` for untrusted workloads.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import psutil

from rlfactory.domain.trajectory import ExecutionStatus
from rlfactory.errors import SandboxError
from rlfactory.sandbox.base import (
    ExecutionRequest,
    ExecutionResult,
    ResourceLimits,
    SandboxCapabilities,
    Stopwatch,
    read_capped,
    write_request_files,
)

_IS_POSIX = os.name == "posix"
_INTERPRETER_FLAGS = ("-B", "-S", "-P", "-X", "utf8")

# Trusted bootstrap executed before the untrusted program (POSIX only): lower resource limits,
# then replace itself with the real interpreter so limits apply to the generated program.
_POSIX_BOOTSTRAP = """
import json, os, resource, sys
limits = json.loads(sys.argv[1])
def lower(res, soft, hard):
    try:
        _, current = resource.getrlimit(res)
        if current != resource.RLIM_INFINITY:
            soft, hard = min(soft, current), min(hard, current)
        resource.setrlimit(res, (soft, hard))
    except (ValueError, OSError):
        pass
# CPU: soft < hard so exhaustion arrives as SIGXCPU (classified as a timeout), not SIGKILL.
lower(resource.RLIMIT_CPU, limits["cpu"], limits["cpu"] + 1)
lower(resource.RLIMIT_AS, limits["memory"], limits["memory"])
lower(resource.RLIMIT_FSIZE, limits["fsize"], limits["fsize"])
lower(resource.RLIMIT_CORE, 0, 0)
os.execv(sys.argv[2], sys.argv[2:])
"""


def _feed_stdin(process: subprocess.Popen[bytes], data: bytes) -> None:
    stream = process.stdin
    if stream is None:
        return
    try:
        if data:
            stream.write(data)
        stream.close()
    except (BrokenPipeError, OSError, ValueError):
        pass


def _tree_rss(process: psutil.Process) -> int:
    try:
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
    except psutil.Error:
        return 0
    return total


def _largest_file(directory: Path) -> int:
    largest = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            try:
                largest = max(largest, (Path(root) / name).stat().st_size)
            except OSError:
                continue
    return largest


class SubprocessSandbox:
    name = "subprocess"

    def __init__(self, python_executable: str | None = None, poll_interval_s: float = 0.01) -> None:
        self.python = python_executable or sys.executable
        self.poll_interval_s = poll_interval_s

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            name=self.name,
            process_isolation=True,
            filesystem_isolation="temp_dir",
            network_isolation=False,
            memory_limit="enforced" if _IS_POSIX else "best_effort",
            cpu_limit="enforced" if _IS_POSIX else "best_effort",
            notes=(
                (
                    "Not a security boundary: the program runs as the current user and can "
                    "read the host filesystem and use the network."
                ),
                "Python starts with -S (stdlib only) and a minimal environment without secrets.",
            ),
        )

    def _command(self, request: ExecutionRequest) -> list[str]:
        program = [self.python, *_INTERPRETER_FLAGS, request.entrypoint]
        if not _IS_POSIX:
            return program
        limits = request.limits
        payload = json.dumps(
            {
                "cpu": limits.cpu_time_s,
                "memory": limits.memory_mb * 1024 * 1024,
                "fsize": limits.max_file_bytes,
            }
        )
        return [self.python, "-B", "-S", "-c", _POSIX_BOOTSTRAP, payload, *program]

    @staticmethod
    def _environment(workdir: Path) -> dict[str, str]:
        tmp = workdir / ".tmp"
        tmp.mkdir(exist_ok=True)
        env = {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "HOME": str(workdir),
            "TMPDIR": str(tmp),
            "TEMP": str(tmp),
            "TMP": str(tmp),
            "LANG": "C.UTF-8",
        }
        if os.name == "nt":
            # Required by the Windows CRT / socket / random initialisation in child Pythons.
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        return env

    def _kill(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.kill()
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, psutil.Error, OSError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()

    def _watch(
        self,
        process: subprocess.Popen[bytes],
        limits: ResourceLimits,
        streams: tuple[Path, Path],
        workdir: Path,
        clock: Stopwatch,
    ) -> tuple[ExecutionStatus | None, int]:
        try:
            handle = psutil.Process(process.pid)
        except psutil.Error:
            handle = None
        memory_limit = limits.memory_mb * 1024 * 1024
        peak_rss = 0
        polls = 0
        while True:
            try:
                process.wait(timeout=self.poll_interval_s)
                return None, peak_rss
            except subprocess.TimeoutExpired:
                pass
            polls += 1
            if clock.elapsed() > limits.wall_time_s:
                self._kill(process)
                return ExecutionStatus.TIMEOUT, peak_rss
            if handle is not None:
                rss = _tree_rss(handle)
                peak_rss = max(peak_rss, rss)
                if not _IS_POSIX and rss > memory_limit:
                    self._kill(process)
                    return ExecutionStatus.MEMORY_EXCEEDED, peak_rss
            if any(
                path.exists() and path.stat().st_size > limits.max_output_bytes for path in streams
            ):
                self._kill(process)
                return ExecutionStatus.OUTPUT_LIMIT, peak_rss
            if polls % 10 == 0 and _largest_file(workdir) > limits.max_file_bytes:
                self._kill(process)
                return ExecutionStatus.FILE_LIMIT, peak_rss

    @staticmethod
    def _classify_exit(exit_code: int | None, stderr: str) -> ExecutionStatus:
        if exit_code == 0:
            return ExecutionStatus.OK
        if sys.platform != "win32" and exit_code is not None and exit_code < 0:
            signum = -exit_code
            if signum == signal.SIGXCPU:
                return ExecutionStatus.TIMEOUT
            if signum == signal.SIGXFSZ:
                return ExecutionStatus.FILE_LIMIT
            if signum == signal.SIGKILL:
                return ExecutionStatus.MEMORY_EXCEEDED
        if "MemoryError" in stderr:
            return ExecutionStatus.MEMORY_EXCEEDED
        return ExecutionStatus.RUNTIME_ERROR

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        limits = request.limits
        with tempfile.TemporaryDirectory(prefix="rlf-sbx-") as tmp:
            root = Path(tmp)
            workdir = root / "work"
            workdir.mkdir()
            write_request_files(request, workdir)
            stdout_path, stderr_path = root / "stdout.bin", root / "stderr.bin"
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            clock = Stopwatch()
            with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
                try:
                    process = subprocess.Popen(
                        self._command(request),
                        cwd=workdir,
                        env=self._environment(workdir),
                        stdin=subprocess.PIPE,
                        stdout=out,
                        stderr=err,
                        start_new_session=_IS_POSIX,
                        creationflags=creationflags,
                    )
                except OSError as exc:
                    raise SandboxError(f"failed to start sandboxed interpreter: {exc}") from exc
                feeder = threading.Thread(
                    target=_feed_stdin,
                    args=(process, request.stdin.encode("utf-8")),
                    daemon=True,
                )
                feeder.start()
                forced_status, peak_rss = self._watch(
                    process, limits, (stdout_path, stderr_path), workdir, clock
                )
                feeder.join(timeout=1.0)
            duration = clock.elapsed()
            stdout, stdout_truncated = read_capped(stdout_path, limits.max_output_bytes)
            stderr, stderr_truncated = read_capped(stderr_path, limits.max_output_bytes)
        exit_code = process.returncode
        status = forced_status or self._classify_exit(exit_code, stderr)
        return ExecutionResult(
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            duration_s=duration,
            peak_memory_mb=round(peak_rss / (1024 * 1024), 2) if peak_rss else None,
            sandbox=self.name,
            detail="killed by watchdog" if forced_status else "",
        )
