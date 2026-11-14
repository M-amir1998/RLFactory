"""Train/eval contamination control.

A :class:`ContaminationFilter` is built from evaluation suites (and optional benchmark IDs or
held-out template families) and applied by dataset builders and the data flywheel. It excludes:

* evaluation task IDs, and any task *derived* from one by mutation (``parent_task_id``);
* prompts whose normalised text matches an evaluation prompt;
* benchmark IDs;
* whole template families when evaluating held-out templates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from rlfactory.domain.common import stable_hash
from rlfactory.domain.task import Task


def prompt_fingerprint(text: str) -> str:
    return stable_hash(" ".join(text.lower().split()), length=20)


@dataclass(frozen=True)
class ContaminationFilter:
    task_ids: frozenset[str] = field(default_factory=frozenset)
    prompt_hashes: frozenset[str] = field(default_factory=frozenset)
    families: frozenset[str] = field(default_factory=frozenset)
    benchmark_ids: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_tasks(
        cls,
        tasks: Iterable[Task],
        *,
        exclude_families: bool = False,
        benchmark_ids: Iterable[str] = (),
    ) -> ContaminationFilter:
        materialised = list(tasks)
        return cls(
            task_ids=frozenset(t.task_id for t in materialised),
            prompt_hashes=frozenset(prompt_fingerprint(t.prompt) for t in materialised),
            families=frozenset(t.family for t in materialised) if exclude_families else frozenset(),
            benchmark_ids=frozenset(benchmark_ids)
            | frozenset(t.metadata.benchmark_id for t in materialised if t.metadata.benchmark_id),
        )

    def merge(self, other: ContaminationFilter) -> ContaminationFilter:
        return ContaminationFilter(
            task_ids=self.task_ids | other.task_ids,
            prompt_hashes=self.prompt_hashes | other.prompt_hashes,
            families=self.families | other.families,
            benchmark_ids=self.benchmark_ids | other.benchmark_ids,
        )

    @property
    def empty(self) -> bool:
        return not (self.task_ids or self.prompt_hashes or self.families or self.benchmark_ids)

    def reason(self, task: Task) -> str | None:
        """Why ``task`` must be excluded from training data, or ``None`` if it is clean."""
        if task.task_id in self.task_ids:
            return "evaluation_task_id"
        if task.metadata.parent_task_id in self.task_ids:
            return "derived_from_evaluation_task"
        if task.metadata.benchmark_id and task.metadata.benchmark_id in self.benchmark_ids:
            return "benchmark_id"
        if task.family in self.families:
            return "held_out_family"
        if prompt_fingerprint(task.prompt) in self.prompt_hashes:
            return "evaluation_prompt_match"
        return None

    def reason_for_record(
        self, task_id: str, prompt_text: str, family: str | None = None
    ) -> str | None:
        """Same check for dataset records that only carry an id, prompt text and family."""
        if task_id in self.task_ids:
            return "evaluation_task_id"
        if family is not None and family in self.families:
            return "held_out_family"
        if prompt_fingerprint(prompt_text) in self.prompt_hashes:
            return "evaluation_prompt_match"
        return None

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "task_ids": sorted(self.task_ids),
            "prompt_hashes": sorted(self.prompt_hashes),
            "families": sorted(self.families),
            "benchmark_ids": sorted(self.benchmark_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> ContaminationFilter:
        return cls(
            task_ids=frozenset(data.get("task_ids", [])),
            prompt_hashes=frozenset(data.get("prompt_hashes", [])),
            families=frozenset(data.get("families", [])),
            benchmark_ids=frozenset(data.get("benchmark_ids", [])),
        )


def deduplicate(tasks: Iterable[Task]) -> tuple[list[Task], list[Task]]:
    """Drop exact duplicates by task id and by normalised prompt (first occurrence wins)."""
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    unique: list[Task] = []
    duplicates: list[Task] = []
    for task in tasks:
        fingerprint = prompt_fingerprint(task.prompt)
        if task.task_id in seen_ids or fingerprint in seen_prompts:
            duplicates.append(task)
            continue
        seen_ids.add(task.task_id)
        seen_prompts.add(fingerprint)
        unique.append(task)
    return unique, duplicates
