"""Synthetic task pipeline: generate -> validate -> difficulty -> deduplicate -> decontaminate.

No LLM is involved. An LLM-backed generator can be added later behind :class:`CandidateSource`;
its candidates would pass through exactly the same validation and filtering.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlfactory.domain.common import stable_seed
from rlfactory.domain.task import Task, TaskDomain
from rlfactory.tasks.contamination import ContaminationFilter, deduplicate
from rlfactory.tasks.generators.catalog import TemplateRegistry
from rlfactory.tasks.validation import TaskValidator


class GeneratedSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["generated"] = "generated"
    domain: TaskDomain
    templates: tuple[str, ...] = ()
    """Template names within the domain; empty means all templates of the domain."""
    count: int = Field(ge=1, le=100_000)
    difficulty: tuple[int, int] = (1, 5)
    seed: int = 0

    @model_validator(mode="after")
    def _difficulty_range(self) -> GeneratedSource:
        low, high = self.difficulty
        if not 1 <= low <= high <= 5:
            raise ValueError("difficulty must be a range within [1, 5]")
        return self


class CandidateSource(Protocol):
    """Anything that proposes candidate tasks (deterministic templates, or later an LLM)."""

    def candidates(self) -> Iterable[Task]: ...


def generate_from_source(source: GeneratedSource, templates: TemplateRegistry) -> list[Task]:
    selected = templates.by_domain(source.domain, source.templates)
    if not selected:
        return []
    tasks: list[Task] = []
    seen: set[str] = set()
    for index in range(source.count):
        template = selected[index % len(selected)]
        for attempt in range(20):
            rng = Random(stable_seed(source.seed, template.family, index, attempt))
            difficulty = rng.randint(*source.difficulty)
            task = template.make_task(template.sample(rng, difficulty))
            if task.task_id not in seen:
                seen.add(task.task_id)
                tasks.append(task)
                break
    return tasks


@dataclass
class SynthesisReport:
    candidates: int = 0
    invalid: dict[str, list[str]] = field(default_factory=dict)
    duplicates: int = 0
    contaminated: dict[str, str] = field(default_factory=dict)
    accepted: int = 0
    difficulty_histogram: dict[int, int] = field(default_factory=dict)
    by_family: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": self.candidates,
            "invalid": len(self.invalid),
            "invalid_reasons": self.invalid,
            "duplicates": self.duplicates,
            "contaminated": self.contaminated,
            "accepted": self.accepted,
            "difficulty_histogram": {
                str(k): v for k, v in sorted(self.difficulty_histogram.items())
            },
            "by_family": dict(sorted(self.by_family.items())),
        }


class SynthesisPipeline:
    def __init__(
        self,
        templates: TemplateRegistry,
        validator: TaskValidator | None,
        contamination: ContaminationFilter | None = None,
    ) -> None:
        self.templates = templates
        self.validator = validator
        self.contamination = contamination or ContaminationFilter()

    def run(self, sources: Sequence[GeneratedSource]) -> tuple[list[Task], SynthesisReport]:
        candidates = [
            task for source in sources for task in generate_from_source(source, self.templates)
        ]
        return self.filter(candidates)

    def filter(self, candidates: Sequence[Task]) -> tuple[list[Task], SynthesisReport]:
        """Validate, deduplicate and decontaminate candidates (also used for flywheel mutants)."""
        report = SynthesisReport(candidates=len(candidates))
        unique, duplicates = deduplicate(candidates)
        report.duplicates = len(duplicates)
        accepted: list[Task] = []
        for task in unique:
            reason = self.contamination.reason(task)
            if reason is not None:
                report.contaminated[task.task_id] = reason
                continue
            if self.validator is not None:
                outcome = self.validator.validate(task)
                if not outcome.valid:
                    report.invalid[task.task_id] = list(outcome.reasons)
                    continue
            accepted.append(task)
        report.accepted = len(accepted)
        report.difficulty_histogram = dict(
            Counter(t.metadata.difficulty.level for t in accepted if t.metadata.difficulty)
        )
        report.by_family = dict(Counter(t.family for t in accepted))
        return accepted, report
