"""Registry of all task templates, keyed by family (``<domain>.<template>``)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rlfactory.domain.task import Task, TaskDomain
from rlfactory.errors import RegistryError
from rlfactory.tasks.generators.arithmetic import (
    FractionSumTemplate,
    IntegerChainTemplate,
    PercentageTemplate,
    WordProblemTemplate,
)
from rlfactory.tasks.generators.base import TaskTemplate
from rlfactory.tasks.generators.python_tasks import python_templates
from rlfactory.tasks.generators.structured import (
    AggregateTemplate,
    ExtractionTemplate,
    FilterSortTemplate,
    LogicConstraintsTemplate,
)
from rlfactory.tasks.generators.tool_use import (
    CurrencyConversionTemplate,
    InventoryValueTemplate,
    OrderTotalTemplate,
)
from rlfactory.tasks.generators.transformation import transformation_templates

AnyTemplate = TaskTemplate[Any]


class TemplateRegistry:
    def __init__(self, templates: Iterable[AnyTemplate]) -> None:
        self._by_family: dict[str, AnyTemplate] = {}
        for template in templates:
            if template.family in self._by_family:
                raise RegistryError(f"duplicate template family {template.family!r}")
            self._by_family[template.family] = template

    def families(self) -> list[str]:
        return sorted(self._by_family)

    def get(self, family: str) -> AnyTemplate:
        try:
            return self._by_family[family]
        except KeyError as exc:
            raise RegistryError(f"unknown task family {family!r}") from exc

    def for_task(self, task: Task) -> AnyTemplate:
        if task.metadata.template is None:
            raise RegistryError(f"task {task.task_id} was not produced by a template")
        return self.get(task.metadata.family)

    def by_domain(self, domain: TaskDomain, names: Iterable[str] = ()) -> list[AnyTemplate]:
        wanted = set(names)
        selected = [
            t
            for family, t in sorted(self._by_family.items())
            if t.domain is domain and (not wanted or t.name in wanted)
        ]
        unknown = wanted - {t.name for t in selected}
        if unknown:
            raise RegistryError(f"unknown {domain.value} templates: {sorted(unknown)}")
        return selected


def default_template_registry() -> TemplateRegistry:
    templates: list[AnyTemplate] = [
        IntegerChainTemplate(),
        FractionSumTemplate(),
        WordProblemTemplate(),
        PercentageTemplate(),
        FilterSortTemplate(),
        AggregateTemplate(),
        LogicConstraintsTemplate(),
        ExtractionTemplate(),
        InventoryValueTemplate(),
        OrderTotalTemplate(),
        CurrencyConversionTemplate(),
        *python_templates(),
        *transformation_templates(),
    ]
    return TemplateRegistry(templates)
