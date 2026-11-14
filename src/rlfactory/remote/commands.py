"""Command execution seam: real subprocess runner and a recording dry-run runner."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rlfactory.telemetry.logging import redact


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandRunner(Protocol):
    def run(self, args: Sequence[str], *, timeout: float | None = None) -> CommandResult: ...


class LocalCommandRunner:
    """Runs argument lists (never through a local shell). Commands are non-interactive, so stdin
    is /dev/null: ssh must never forward, or wait on, the user's terminal."""

    def run(self, args: Sequence[str], *, timeout: float | None = None) -> CommandResult:
        try:
            completed = subprocess.run(
                list(args),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(tuple(args), 124, "", f"timed out after {exc.timeout}s")
        except OSError as exc:
            return CommandResult(tuple(args), 127, "", str(exc))
        return CommandResult(
            tuple(args), completed.returncode, completed.stdout, redact(completed.stderr)
        )


Responder = Callable[[tuple[str, ...]], CommandResult | None]


@dataclass
class DryRunCommandRunner:
    """Records commands instead of running them; optional responder fakes outputs in tests."""

    responder: Responder | None = None
    commands: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,  # noqa: ARG002 - CommandRunner protocol
    ) -> CommandResult:
        command = tuple(args)
        self.commands.append(command)
        if self.responder is not None:
            response = self.responder(command)
            if response is not None:
                return response
        return CommandResult(command, 0, "", "")
