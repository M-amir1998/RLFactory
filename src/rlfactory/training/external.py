"""Shared helpers for backends whose framework runs only on a GPU host (verl, OpenRLHF)."""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rlfactory.datasets.records import RLPromptRecord
from rlfactory.domain.training import LaunchMode, PreparedTrainingRun, TrainingHandle, TrainingRun
from rlfactory.errors import BackendUnavailableError, ConfigError


def load_rl_records(run: TrainingRun) -> list[RLPromptRecord]:
    records_path = Path(run.dataset.path).parent / "records.jsonl"
    if not records_path.exists():
        raise ConfigError(
            f"{records_path} not found; build the dataset with `rlfactory dataset build-grpo`"
        )
    return [
        RLPromptRecord.model_validate_json(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def framework_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def remote_only_launch(prepared: PreparedTrainingRun, framework: str) -> TrainingHandle:
    """verl/OpenRLHF are never installed next to RLFactory's core; refuse local launches clearly."""
    if prepared.run.launch_mode is LaunchMode.REMOTE:
        raise BackendUnavailableError(
            f"{framework} runs are launched on the GPU host via `rlfactory train --remote`"
        )
    raise BackendUnavailableError(
        f"{framework} is not part of the local environment by design (conflicting pins). "
        f"The prepared run directory {prepared.workdir} contains launch.sh for a GPU host."
    )


def handle_for(
    prepared: PreparedTrainingRun, backend: str, remote_host: str | None = None
) -> TrainingHandle:
    return TrainingHandle(
        handle_id=f"{backend}-{prepared.run.run_id}",
        backend=backend,
        run_id=prepared.run.run_id,
        workdir=prepared.workdir,
        launch_mode=prepared.run.launch_mode,
        remote_host=remote_host,
        started_at=datetime.now(UTC),
    )


def write_jsonl(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
