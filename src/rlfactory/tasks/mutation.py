"""Deterministic task mutation: perturb template parameters and rebuild (ground truth recomputed)."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from rlfactory.domain.common import stable_seed
from rlfactory.domain.task import Task
from rlfactory.errors import RegistryError
from rlfactory.tasks.generators.base import MutationKind
from rlfactory.tasks.generators.catalog import TemplateRegistry


class TaskMutator:
    def __init__(self, templates: TemplateRegistry, *, seed: int = 0) -> None:
        self.templates = templates
        self.seed = seed

    def supported(self, task: Task) -> list[MutationKind]:
        try:
            template = self.templates.for_task(task)
        except RegistryError:
            return []
        return sorted(template.mutations, key=lambda kind: kind.value)

    def mutate(self, task: Task, kind: MutationKind, *, variant: int = 0) -> Task | None:
        """Apply one mutation. Returns ``None`` when it does not apply or changes nothing."""
        template = self.templates.for_task(task)
        if kind not in template.mutations:
            return None
        rng = Random(stable_seed(self.seed, task.task_id, kind.value, variant))
        params = template.params_of(task)
        try:
            mutated = template.mutate(params, kind, rng)
        except ValueError:
            return None
        if mutated is None or mutated == params:
            return None
        child = template.make_task(mutated, source="mutation", parent=task, mutation=kind)
        return None if child.task_id == task.task_id else child

    def mutate_many(
        self, task: Task, count: int, kinds: Sequence[MutationKind] | None = None
    ) -> list[Task]:
        """Up to ``count`` distinct mutants, cycling through the supported mutation kinds."""
        available = [k for k in (kinds or self.supported(task)) if k in self.supported(task)]
        if not available or count <= 0:
            return []
        mutants: dict[str, Task] = {}
        attempt = 0
        while len(mutants) < count and attempt < count * 4:
            kind = available[attempt % len(available)]
            child = self.mutate(task, kind, variant=attempt // len(available))
            if child is not None:
                mutants.setdefault(child.task_id, child)
            attempt += 1
        return list(mutants.values())
