"""Environment protocol and the single-turn base implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

from rlfactory.domain.common import ComponentRef
from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import ModelAction, Observation, StepResult
from rlfactory.environments.tools.registry import ToolRegistry, default_tool_registry
from rlfactory.sandbox.base import ResourceLimits, Sandbox


class Environment(Protocol):
    """Gym-style episode API.

    ``reset`` starts an episode and returns the first observation (the prompt), ``step`` applies
    one model action, ``close`` releases resources. Instances are single-episode-at-a-time and are
    not shared between concurrent rollouts.
    """

    name: str
    version: str

    def reset(self, task: Task) -> Observation: ...

    def step(self, action: ModelAction) -> StepResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EnvironmentContext:
    """Shared, thread-safe dependencies injected into environments."""

    sandbox: Sandbox
    tools: ToolRegistry = field(default_factory=default_tool_registry)
    python_limits: ResourceLimits = field(default_factory=ResourceLimits)


class EpisodeStateError(RuntimeError):
    """``step`` called before ``reset`` or after the episode finished."""


class SingleTurnEnvironment(ABC):
    """Base for environments where one action ends the episode."""

    name: str
    version: str
    domain: TaskDomain

    def __init__(self, context: EnvironmentContext) -> None:
        self.context = context
        self._task: Task | None = None
        self._done = False

    @property
    def ref(self) -> ComponentRef:
        return ComponentRef(name=self.name, version=self.version)

    def reset(self, task: Task) -> Observation:
        if task.domain is not self.domain:
            raise ValueError(f"{self.name} environment cannot run {task.domain} tasks")
        self._task = task
        self._done = False
        return Observation(content=self.render_prompt(task))

    def render_prompt(self, task: Task) -> str:
        return task.prompt

    def step(self, action: ModelAction) -> StepResult:
        task = self._require_active()
        observation = self.evaluate(task, action)
        self._done = True
        return StepResult(observation=observation, done=True)

    def _require_active(self) -> Task:
        if self._task is None:
            raise EpisodeStateError("call reset() before step()")
        if self._done:
            raise EpisodeStateError("episode already finished; call reset() for a new one")
        return self._task

    @abstractmethod
    def evaluate(self, task: Task, action: ModelAction) -> Observation:
        """Interpret ``action`` for ``task`` (parse and, where needed, execute)."""

    def close(self) -> None:
        self._task = None
        self._done = True
