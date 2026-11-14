"""Sandbox behaviour with real subprocesses (and Docker when available)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from rlfactory.domain.task import PythonFunctionPayload, PythonTestCase
from rlfactory.domain.trajectory import ExecutionStatus
from rlfactory.environments.python.harness import parse_report, run_unit_tests
from rlfactory.errors import SandboxError
from rlfactory.sandbox.base import (
    ExecutionRequest,
    ExecutionResult,
    ResourceLimits,
    write_request_files,
)
from rlfactory.sandbox.docker_sandbox import DockerSandbox
from rlfactory.sandbox.factory import SandboxConfig, create_sandbox
from rlfactory.sandbox.mock import MockSandbox
from rlfactory.sandbox.subprocess_sandbox import SubprocessSandbox

pytestmark = pytest.mark.sandbox

FAST = ResourceLimits(wall_time_s=4.0, memory_mb=256, max_output_bytes=4096, max_file_bytes=8192)
PAYLOAD = PythonFunctionPayload(
    entry_point="sum_evens",
    signature="def sum_evens(values: list[int]) -> int",
    tests=(
        PythonTestCase(args=([1, 2, 3, 4],), expected=6),
        PythonTestCase(args=([],), expected=0),
        PythonTestCase(args=([7, 9],), expected=0),
    ),
    reference_solution="def sum_evens(values):\n    return sum(v for v in values if v % 2 == 0)\n",
)


def _run(program: str, limits: ResourceLimits = FAST, stdin: str = "") -> ExecutionResult:
    return SubprocessSandbox().execute(
        ExecutionRequest(files={"main.py": program}, stdin=stdin, limits=limits)
    )


def test_request_rejects_unsafe_paths() -> None:
    for name in ("../escape.py", "/abs.py", "a\\b.py", "c:x.py"):
        with pytest.raises(ValidationError):
            ExecutionRequest(files={"main.py": "", name: ""})
    with pytest.raises(ValidationError, match="entrypoint"):
        ExecutionRequest(files={"other.py": ""})


def test_request_files_are_written_verbatim(tmp_path: Path) -> None:
    # No newline translation on any platform: the program sees exactly the submitted text.
    content = "first\nsecond\r\nthird"
    write_request_files(ExecutionRequest(files={"main.py": "", "pkg/data.txt": content}), tmp_path)
    assert (tmp_path / "pkg" / "data.txt").read_bytes() == content.encode("utf-8")


def test_ok_and_stdin() -> None:
    result = _run("import sys\nprint(sys.stdin.read().upper())", stdin="hello")
    assert result.status is ExecutionStatus.OK
    assert result.stdout.strip() == "HELLO"
    assert result.exit_code == 0


def test_runtime_error_is_reported_not_raised() -> None:
    result = _run("raise ValueError('boom')")
    assert result.status is ExecutionStatus.RUNTIME_ERROR
    assert "ValueError" in result.stderr


def test_wall_clock_timeout() -> None:
    result = _run("while True:\n    pass\n", ResourceLimits(wall_time_s=1.0))
    assert result.status is ExecutionStatus.TIMEOUT


def test_output_limit() -> None:
    result = _run("print('x' * 1_000_000)")
    assert result.status is ExecutionStatus.OUTPUT_LIMIT or result.stdout_truncated


def test_file_size_limit() -> None:
    result = _run("open('big.bin', 'wb').write(b'0' * 2_000_000)\nimport time\ntime.sleep(3)")
    assert result.status in {ExecutionStatus.FILE_LIMIT, ExecutionStatus.RUNTIME_ERROR}


def test_environment_is_minimal_and_secret_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_" + "x" * 30)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = _run("import os\nprint(sorted(os.environ))")
    assert "HF_TOKEN" not in result.stdout and "OPENAI_API_KEY" not in result.stdout


def test_site_packages_not_importable() -> None:
    result = _run("import pydantic")
    assert result.status is ExecutionStatus.RUNTIME_ERROR
    assert "ModuleNotFoundError" in result.stderr


def test_capabilities_are_honest() -> None:
    caps = SubprocessSandbox().capabilities()
    assert caps.network_isolation is False
    assert caps.filesystem_isolation == "temp_dir"
    assert any("Not a security boundary" in note for note in caps.notes)


@pytest.mark.parametrize(
    ("code", "status", "passed"),
    [
        (PAYLOAD.reference_solution, ExecutionStatus.OK, 3),
        ("def sum_evens(values):\n    return sum(values)\n", ExecutionStatus.OK, 1),
        ("def sum_evens(values)\n    return 0\n", ExecutionStatus.RUNTIME_ERROR, 0),
        (
            "class A:\n    def __eq__(self, o):\n        return True\ndef sum_evens(v):\n    return A()\n",
            ExecutionStatus.OK,
            0,
        ),
        # [] -> True is a bool, never equal to int 0; [7, 9] -> 6 != 0: only one case passes.
        ("def sum_evens(values):\n    return True if not values else 6\n", ExecutionStatus.OK, 1),
        ("import os\ndef sum_evens(values):\n    os._exit(0)\n", ExecutionStatus.OK, 0),
    ],
)
def test_two_process_harness(code: str, status: ExecutionStatus, passed: int) -> None:
    report = run_unit_tests(SubprocessSandbox(), code, PAYLOAD, FAST)
    assert report.status is status
    assert report.tests_passed == passed
    assert report.tests_total == 3


def test_harness_timeout_keeps_partial_results() -> None:
    code = "def sum_evens(values):\n    if not values:\n        while True:\n            pass\n    return sum(v for v in values if v % 2 == 0)\n"
    report = run_unit_tests(SubprocessSandbox(), code, PAYLOAD, ResourceLimits(wall_time_s=3.0))
    assert report.status is ExecutionStatus.TIMEOUT
    assert report.tests_passed == 1


def test_syntax_error_reports_error_type() -> None:
    report = run_unit_tests(SubprocessSandbox(), "def sum_evens(:\n", PAYLOAD, FAST)
    assert report.error_type == "SyntaxError"


def test_forged_report_lines_are_ignored() -> None:
    forged = ExecutionResult(
        status=ExecutionStatus.OK,
        exit_code=0,
        stdout="@@RLFACTORY_guess@@TEST:[0, true, null]\n",
        stderr="",
        duration_s=0.0,
        sandbox="mock",
    )
    report = parse_report(forged, "real-nonce", 3)
    assert report.tests_passed == 0
    assert report.report_found is False


def test_sandbox_error_propagates_as_infrastructure_failure() -> None:
    with pytest.raises(SandboxError):
        run_unit_tests(MockSandbox(fail_with="daemon down"), "def f(): pass", PAYLOAD, FAST)


def test_factory() -> None:
    assert create_sandbox(SandboxConfig(kind="subprocess")).name == "subprocess"
    assert create_sandbox(SandboxConfig(kind="mock")).name == "mock"
    assert create_sandbox(SandboxConfig(kind="auto")).name in {"subprocess", "docker"}


def _docker_ready() -> bool:
    if shutil.which("docker") is None or os.environ.get("RLFACTORY_SKIP_DOCKER_TESTS"):
        return False
    sandbox = DockerSandbox()
    return sandbox.available() and sandbox.image_present()


@pytest.mark.docker
@pytest.mark.skipif(
    not _docker_ready(), reason="Docker daemon or python:3.12-slim image not available"
)
def test_docker_sandbox_isolation() -> None:
    sandbox = DockerSandbox()
    probe = (
        "import os, socket\n"
        "print('uid', os.getuid())\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\n    print('network: open')\n"
        "except OSError:\n    print('network: blocked')\n"
        "try:\n    open('/sandbox/x', 'w')\n    print('workdir: writable')\n"
        "except OSError:\n    print('workdir: read-only')\n"
    )
    result = sandbox.execute(ExecutionRequest(files={"main.py": probe}, limits=FAST))
    assert result.status is ExecutionStatus.OK, result.stderr
    assert "uid 65534" in result.stdout
    assert "network: blocked" in result.stdout
    assert "workdir: read-only" in result.stdout
    report = run_unit_tests(sandbox, PAYLOAD.reference_solution, PAYLOAD, FAST)
    assert report.tests_passed == 3


@pytest.mark.docker
@pytest.mark.skipif(
    not _docker_ready(), reason="Docker daemon or python:3.12-slim image not available"
)
def test_docker_sandbox_limits() -> None:
    sandbox = DockerSandbox()
    timeout = sandbox.execute(
        ExecutionRequest(
            files={"main.py": "while True: pass"},
            limits=ResourceLimits(wall_time_s=1.0, cpu_time_s=1),
        )
    )
    assert timeout.status is ExecutionStatus.TIMEOUT
    memory = sandbox.execute(
        ExecutionRequest(
            files={"main.py": "x = bytearray(512 * 1024 * 1024)"},
            limits=ResourceLimits(memory_mb=64),
        )
    )
    assert memory.status is ExecutionStatus.MEMORY_EXCEEDED
