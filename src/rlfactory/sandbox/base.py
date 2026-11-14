"""Sandbox protocol and request/result records."""

from __future__ import annotations

import time
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rlfactory.domain.trajectory import ExecutionStatus


class ResourceLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    wall_time_s: float = Field(default=10.0, gt=0.0, le=600.0)
    cpu_time_s: int = Field(default=10, ge=1, le=600)
    memory_mb: int = Field(default=512, ge=32, le=16_384)
    max_output_bytes: int = Field(default=64_000, ge=1_024)
    max_file_bytes: int = Field(default=1_000_000, ge=1_024)
    max_processes: int = Field(default=32, ge=1)


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    files: dict[str, str]
    """Relative POSIX path -> UTF-8 text content."""
    entrypoint: str = "main.py"
    stdin: str = ""
    limits: ResourceLimits = ResourceLimits()

    @field_validator("files")
    @classmethod
    def _safe_paths(cls, files: dict[str, str]) -> dict[str, str]:
        for name in files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not name:
                raise ValueError(f"unsafe sandbox file path {name!r}")
        return files

    def model_post_init(self, __context: object) -> None:
        if self.entrypoint not in self.files:
            raise ValueError(f"entrypoint {self.entrypoint!r} is not among the request files")


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ExecutionStatus
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration_s: float = Field(ge=0.0)
    peak_memory_mb: float | None = None
    sandbox: str
    detail: str = ""


class SandboxCapabilities(BaseModel):
    """Honest description of what a sandbox implementation enforces."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    process_isolation: bool
    filesystem_isolation: Literal["none", "temp_dir", "container_read_only"]
    network_isolation: bool
    memory_limit: Literal["enforced", "best_effort", "none"]
    cpu_limit: Literal["enforced", "best_effort", "none"]
    notes: tuple[str, ...] = ()


class Sandbox(Protocol):
    name: str

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Run the request. Raises :class:`rlfactory.errors.SandboxError` only for sandbox
        (infrastructure) failures; failures of the program itself are reported in the result."""
        ...

    def capabilities(self) -> SandboxCapabilities: ...


def write_request_files(request: ExecutionRequest, workdir: Path) -> None:
    for name, content in request.files.items():
        target = workdir.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def read_capped(path: Path, limit: int) -> tuple[str, bool]:
    """Read at most ``limit`` bytes of a captured stream; report whether it was truncated."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            data = handle.read(limit)
    except OSError:
        return "", False
    return data.decode("utf-8", errors="replace"), size > limit


class Stopwatch:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
