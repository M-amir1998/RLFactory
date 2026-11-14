"""Task registry: discovers suite configs (``configs/tasks/*.yaml``) and materialises suites."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from rlfactory.config.loader import load_model
from rlfactory.domain.common import stable_hash
from rlfactory.domain.experiment import SuiteRef
from rlfactory.domain.task import Task, TaskDomain
from rlfactory.errors import ConfigError, RegistryError
from rlfactory.tasks.contamination import deduplicate
from rlfactory.tasks.generators.catalog import TemplateRegistry, default_template_registry
from rlfactory.tasks.io import load_tasks
from rlfactory.tasks.synthesis import GeneratedSource, generate_from_source


class FileSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["file"] = "file"
    path: str
    """JSONL of task records, relative to the suite config file."""


def _source_kind(value: Any) -> str:
    """Sources default to ``generated`` when ``kind`` is omitted."""
    if isinstance(value, dict):
        return str(value.get("kind", "generated"))
    return str(getattr(value, "kind", "generated"))


TaskSource = Annotated[
    Annotated[GeneratedSource, Tag("generated")] | Annotated[FileSource, Tag("file")],
    Discriminator(_source_kind),
]


class TaskSuiteConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str = "1"
    description: str = ""
    sources: tuple[TaskSource, ...] = Field(min_length=1)
    verifier_overrides: dict[TaskDomain, str] = Field(default_factory=dict)
    """Replay the same tasks under a different verifier (e.g. strict vs lenient numeric)."""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSuite:
    name: str
    version: str
    description: str
    tasks: tuple[Task, ...]

    def __len__(self) -> int:
        return len(self.tasks)

    def fingerprint(self) -> str:
        return stable_hash(sorted((t.task_id, t.verifier, t.fingerprint()) for t in self.tasks))

    def ref(self) -> SuiteRef:
        return SuiteRef(
            name=self.name,
            version=self.version,
            num_tasks=len(self.tasks),
            fingerprint=self.fingerprint(),
        )

    def by_domain(self) -> dict[TaskDomain, list[Task]]:
        grouped: dict[TaskDomain, list[Task]] = defaultdict(list)
        for task in self.tasks:
            grouped[task.domain].append(task)
        return dict(grouped)

    def get(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def with_tasks(self, tasks: list[Task], *, suffix: str) -> TaskSuite:
        return TaskSuite(f"{self.name}{suffix}", self.version, self.description, tuple(tasks))


def materialize_suite(
    config: TaskSuiteConfig, base_dir: Path, templates: TemplateRegistry | None = None
) -> TaskSuite:
    registry = templates or default_template_registry()
    tasks: list[Task] = []
    for source in config.sources:
        if isinstance(source, GeneratedSource):
            tasks.extend(generate_from_source(source, registry))
        else:
            tasks.extend(load_tasks((base_dir / source.path).resolve()))
    if config.verifier_overrides:
        tasks = [
            task.model_copy(update={"verifier": config.verifier_overrides[task.domain]})
            if task.domain in config.verifier_overrides
            else task
            for task in tasks
        ]
    unique, _ = deduplicate(tasks)
    return TaskSuite(config.name, config.version, config.description, tuple(unique))


class TaskRegistry:
    def __init__(self, config_dir: Path, templates: TemplateRegistry | None = None) -> None:
        self.suite_dir = config_dir / "tasks"
        self.templates = templates or default_template_registry()
        self._configs: dict[str, tuple[Path, TaskSuiteConfig]] | None = None
        self._suites: dict[str, TaskSuite] = {}

    def _discover(self) -> dict[str, tuple[Path, TaskSuiteConfig]]:
        if self._configs is None:
            found: dict[str, tuple[Path, TaskSuiteConfig]] = {}
            for path in sorted(self.suite_dir.glob("*.yaml")):
                config = load_model(path, TaskSuiteConfig)
                if config.name in found:
                    raise ConfigError(f"duplicate task suite name {config.name!r} in {path}")
                found[config.name] = (path, config)
            self._configs = found
        return self._configs

    def names(self) -> list[str]:
        return sorted(self._discover())

    def config(self, name: str) -> TaskSuiteConfig:
        try:
            return self._discover()[name][1]
        except KeyError as exc:
            raise RegistryError(
                f"unknown task suite {name!r}; known: {', '.join(self.names())}"
            ) from exc

    def load(self, name: str) -> TaskSuite:
        if name not in self._suites:
            path, config = self._discover().get(name, (None, None))
            if path is None or config is None:
                raise RegistryError(
                    f"unknown task suite {name!r}; known: {', '.join(self.names())}"
                )
            self._suites[name] = materialize_suite(config, path.parent, self.templates)
        return self._suites[name]
