"""Task suite serialisation (JSONL, one task per line)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from rlfactory.domain.task import Task
from rlfactory.errors import ConfigError


def save_tasks(tasks: Iterable[Task], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(task.model_dump_json() + "\n")
            count += 1
    tmp.replace(path)
    return count


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read task file {path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            tasks.append(Task.model_validate_json(line))
        except ValidationError as exc:
            raise ConfigError(f"{path}:{number}: invalid task record: {exc}") from exc
    return tasks
