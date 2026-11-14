"""Mock sandbox for tests: returns scripted results without executing anything."""

from __future__ import annotations

from collections.abc import Callable

from rlfactory.domain.trajectory import ExecutionStatus
from rlfactory.errors import SandboxError
from rlfactory.sandbox.base import ExecutionRequest, ExecutionResult, SandboxCapabilities

Handler = Callable[[ExecutionRequest], ExecutionResult]


class MockSandbox:
    name = "mock"

    def __init__(self, handler: Handler | None = None, *, fail_with: str | None = None) -> None:
        self.handler = handler
        self.fail_with = fail_with
        self.requests: list[ExecutionRequest] = []

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            name=self.name,
            process_isolation=False,
            filesystem_isolation="none",
            network_isolation=False,
            memory_limit="none",
            cpu_limit="none",
            notes=("Executes nothing; for tests only.",),
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        if self.fail_with is not None:
            raise SandboxError(self.fail_with)
        if self.handler is not None:
            return self.handler(request)
        return ExecutionResult(
            status=ExecutionStatus.OK,
            exit_code=0,
            stdout="",
            stderr="",
            duration_s=0.0,
            sandbox=self.name,
        )
