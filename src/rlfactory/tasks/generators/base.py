"""Template abstraction: a task is a pure function of (template, typed parameters).

This makes generation, mutation and re-validation deterministic: mutating a task means perturbing
its parameters and rebuilding it, so ground truth is always recomputed rather than edited.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from random import Random
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from rlfactory.domain.task import (
    ArithmeticPayload,
    PythonFunctionPayload,
    StructuredOutputPayload,
    Task,
    TaskDomain,
    TaskMetadata,
    ToolUsePayload,
    TransformationPayload,
    make_task_id,
)
from rlfactory.tasks.difficulty import estimate_difficulty


class MutationKind(StrEnum):
    NUMBERS = "numbers"
    CONSTRAINTS = "constraints"
    VARIABLE_NAMES = "variable_names"
    INPUT_LENGTH = "input_length"
    FORMAT = "format"
    EDGE_CASE = "edge_case"


class TemplateParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


AnyPayload = (
    ArithmeticPayload
    | StructuredOutputPayload
    | PythonFunctionPayload
    | TransformationPayload
    | ToolUsePayload
)


@dataclass(frozen=True)
class BuiltTask:
    prompt: str
    payload: AnyPayload
    signals: dict[str, float] = field(default_factory=dict)


P = TypeVar("P", bound=TemplateParams)


class TaskTemplate(ABC, Generic[P]):
    name: str
    domain: TaskDomain
    environment: str
    verifier: str
    params_model: type[P]
    mutations: frozenset[MutationKind] = frozenset()
    version: str = "1"

    @property
    def family(self) -> str:
        return f"{self.domain.value}.{self.name}"

    @abstractmethod
    def sample(self, rng: Random, difficulty: int) -> P:
        """Draw parameters for a task of roughly the given difficulty (1-5)."""

    @abstractmethod
    def build(self, params: P) -> BuiltTask:
        """Render prompt and payload (including ground truth) from parameters."""

    def mutate(self, params: P, kind: MutationKind, rng: Random) -> P | None:  # noqa: ARG002
        """Return perturbed parameters, or ``None`` if ``kind`` does not apply."""
        return None

    def params_of(self, task: Task) -> P:
        return self.params_model.model_validate(task.metadata.params)

    def make_task(
        self,
        params: P,
        *,
        source: Literal["synthetic", "fixture", "external", "mutation"] = "synthetic",
        parent: Task | None = None,
        mutation: MutationKind | None = None,
    ) -> Task:
        built = self.build(params)
        params_json = params.model_dump(mode="json")
        return Task(
            task_id=make_task_id(self.domain, self.name, params_json),
            domain=self.domain,
            prompt=built.prompt,
            environment=self.environment,
            verifier=self.verifier,
            payload=built.payload,
            metadata=TaskMetadata(
                version=self.version,
                family=self.family,
                generator=self.domain.value,
                template=self.name,
                params=params_json,
                difficulty=estimate_difficulty(built.signals),
                source=source,
                parent_task_id=parent.task_id if parent is not None else None,
                mutation=mutation.value if mutation is not None else None,
            ),
        )
