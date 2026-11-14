"""Rich progress reporting for rollouts and evaluations."""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from rlfactory.domain.trajectory import Trajectory
from rlfactory.rollouts.orchestrator import RolloutSummary


class RichRolloutProgress:
    def __init__(self, console: Console, label: str) -> None:
        self.console = console
        self.label = label
        self._progress: Progress | None = None
        self._task: TaskID | None = None
        self._done = 0
        self._passed = 0

    def start(self, total: int, skipped: int) -> None:
        if skipped:
            self.console.print(
                f"[dim]{self.label}: resuming, {skipped} trajectories already completed[/dim]"
            )
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("passed {task.fields[success]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )
        self._progress.start()
        self._task = self._progress.add_task(self.label, total=total, success="-")

    def advance(self, trajectory: Trajectory) -> None:
        self._done += 1
        self._passed += int(trajectory.passed)
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, advance=1, success=f"{self._passed / self._done:.0%}")

    def finish(self, summary: RolloutSummary) -> None:
        if self._progress is not None:
            self._progress.stop()
        rate = f"{summary.pass_rate:.1%}" if summary.pass_rate is not None else "n/a"
        self.console.print(
            f"{self.label}: {summary.completed} completed, {summary.failed} infra-failed, "
            f"pass rate {rate}, {summary.retries} retries, {summary.wall_time_s:.1f}s"
        )
