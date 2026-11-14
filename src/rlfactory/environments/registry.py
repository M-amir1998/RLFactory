"""Environment registry: name -> versioned factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rlfactory.domain.common import ComponentRef
from rlfactory.domain.task import TaskDomain
from rlfactory.environments.arithmetic import ArithmeticEnvironment
from rlfactory.environments.base import Environment, EnvironmentContext
from rlfactory.environments.python import PythonEnvironment
from rlfactory.environments.structured import StructuredOutputEnvironment
from rlfactory.environments.tools.environment import ToolUseEnvironment
from rlfactory.environments.transformation import TransformationEnvironment
from rlfactory.errors import RegistryError

EnvironmentFactory = Callable[[EnvironmentContext], Environment]


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    version: str
    domain: TaskDomain
    description: str
    factory: EnvironmentFactory
    multi_step: bool = False
    requires_sandbox: bool = False

    @property
    def ref(self) -> ComponentRef:
        return ComponentRef(name=self.name, version=self.version)


class EnvironmentRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, EnvironmentSpec] = {}

    def register(self, spec: EnvironmentSpec) -> None:
        if spec.name in self._specs:
            raise RegistryError(f"environment {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def spec(self, name: str) -> EnvironmentSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._specs))
            raise RegistryError(f"unknown environment {name!r}; known: {known}") from exc

    def create(self, name: str, context: EnvironmentContext) -> Environment:
        return self.spec(name).factory(context)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[EnvironmentSpec]:
        return [self._specs[name] for name in self.names()]


def default_environment_registry() -> EnvironmentRegistry:
    registry = EnvironmentRegistry()
    registry.register(
        EnvironmentSpec(
            name=ArithmeticEnvironment.name,
            version=ArithmeticEnvironment.version,
            domain=TaskDomain.ARITHMETIC,
            description="Exact arithmetic; answer in <answer> tags.",
            factory=ArithmeticEnvironment,
        )
    )
    registry.register(
        EnvironmentSpec(
            name=StructuredOutputEnvironment.name,
            version=StructuredOutputEnvironment.version,
            domain=TaskDomain.STRUCTURED_OUTPUT,
            description="JSON answers validated against schemas, values and constraints.",
            factory=StructuredOutputEnvironment,
        )
    )
    registry.register(
        EnvironmentSpec(
            name=PythonEnvironment.name,
            version=PythonEnvironment.version,
            domain=TaskDomain.PYTHON,
            description="Generated Python functions run against hidden tests in a sandbox.",
            factory=PythonEnvironment,
            requires_sandbox=True,
        )
    )
    registry.register(
        EnvironmentSpec(
            name=TransformationEnvironment.name,
            version=TransformationEnvironment.version,
            domain=TaskDomain.TRANSFORMATION,
            description="Deterministic text transformations; exact output in <answer> tags.",
            factory=TransformationEnvironment,
        )
    )
    registry.register(
        EnvironmentSpec(
            name=ToolUseEnvironment.name,
            version=ToolUseEnvironment.version,
            domain=TaskDomain.TOOL_USE,
            description="Multi-step deterministic tool use (calculator, kv_lookup, json_transform).",
            factory=ToolUseEnvironment,
            multi_step=True,
        )
    )
    return registry
