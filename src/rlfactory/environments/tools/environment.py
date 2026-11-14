"""Multi-step tool-use environment: observation -> tool action -> observation -> ... -> answer."""

from __future__ import annotations

import json

from rlfactory.domain.common import ComponentRef
from rlfactory.domain.task import Task, TaskDomain, ToolCall, ToolUsePayload
from rlfactory.domain.trajectory import (
    ModelAction,
    Observation,
    Role,
    StepResult,
    ToolCallRecord,
)
from rlfactory.environments.base import EnvironmentContext, EpisodeStateError
from rlfactory.environments.parsing import ANSWER_TAG, extract_tag, parse_tool_call
from rlfactory.environments.tools.registry import ToolState, ToolUsageError
from rlfactory.errors import RegistryError


class ToolUseEnvironment:
    """Each turn the model either calls exactly one tool or gives a final ``<answer>``.

    If a turn contains both, the tool call wins (the answer is ignored). Invalid calls produce an
    error observation rather than ending the episode; exceeding ``max_steps`` truncates it.
    """

    name = "tool_use"
    version = "1.0.0"

    def __init__(self, context: EnvironmentContext) -> None:
        self.tools = context.tools
        self._payload: ToolUsePayload | None = None
        self._state = ToolState()
        self._steps = 0
        self._done = True

    @property
    def ref(self) -> ComponentRef:
        return ComponentRef(name=self.name, version=self.version)

    def reset(self, task: Task) -> Observation:
        if task.domain is not TaskDomain.TOOL_USE or not isinstance(task.payload, ToolUsePayload):
            raise ValueError("tool_use environment requires a tool-use task")
        unknown = [name for name in task.payload.tools if name not in self.tools]
        if unknown:
            raise RegistryError(f"task requires unregistered tools: {unknown}")
        self._payload = task.payload
        self._state = ToolState(kv_store=task.payload.kv_store, document=task.payload.document)
        self._steps = 0
        self._done = False
        content = (
            f"{task.prompt}\n\nAvailable tools:\n{self.tools.describe(task.payload.tools)}\n\n"
            f"You have at most {task.payload.max_steps} turns."
        )
        return Observation(content=content)

    def _execute(self, call: ToolCall | None, error: str | None) -> ToolCallRecord:
        if self._payload is None:
            raise EpisodeStateError("call reset() before step()")
        if call is None:
            return ToolCallRecord(call=None, error=error or "malformed tool call")
        if call.name not in self._payload.tools:
            return ToolCallRecord(call=call, error=f"tool {call.name!r} is not available")
        try:
            result = self.tools.get(call.name)(call.arguments, self._state)
        except ToolUsageError as exc:
            return ToolCallRecord(call=call, error=str(exc))
        return ToolCallRecord(call=call, result=result)

    def step(self, action: ModelAction) -> StepResult:
        if self._payload is None:
            raise EpisodeStateError("call reset() before step()")
        if self._done:
            raise EpisodeStateError("episode already finished; call reset() for a new one")
        self._steps += 1
        call, call_error = parse_tool_call(action.text)
        answer = extract_tag(action.text, ANSWER_TAG)
        if call is None and call_error is None and answer is not None:
            self._done = True
            return StepResult(
                observation=Observation(content="Final answer received.", parsed_answer=answer),
                done=True,
            )
        if call is not None or call_error is not None:
            record = self._execute(call, call_error)
            content = (
                f"Error: {record.error}"
                if record.error is not None
                else json.dumps(record.result, sort_keys=True)
            )
            observation = Observation(content=content, role=Role.TOOL, tool=record)
        else:
            observation = Observation(
                content="Reply with one <tool_call> or a final <answer>.",
                parse_error="no_tool_call_or_answer",
            )
        if self._steps >= self._payload.max_steps:
            self._done = True
            return StepResult(observation=observation, done=True, truncated=True)
        return StepResult(observation=observation, done=False)

    def close(self) -> None:
        self._payload = None
        self._done = True
