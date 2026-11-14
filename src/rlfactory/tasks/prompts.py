"""Prompt rendering shared by rollouts, evaluation and every dataset builder.

Training prompts and rollout/evaluation prompts come from the same builder, so the model is trained
on exactly the format it is evaluated on. The template has a name and version recorded in lineage.
"""

from __future__ import annotations

from rlfactory.domain.common import ComponentRef
from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import Message, Observation, Role

DEFAULT_SYSTEM_PROMPT = "You are a precise assistant that solves verifiable tasks."

FORMAT_INSTRUCTIONS: dict[TaskDomain, str] = {
    TaskDomain.ARITHMETIC: (
        "Give the final answer inside <answer></answer> tags as an integer, a terminating "
        "decimal or a reduced fraction."
    ),
    TaskDomain.STRUCTURED_OUTPUT: "Give only the JSON value inside <answer></answer> tags.",
    TaskDomain.PYTHON: (
        "Reply with the complete function in a single ```python code block. Do not include "
        "tests or example usage."
    ),
    TaskDomain.TRANSFORMATION: (
        "Give only the transformed text inside <answer></answer> tags, one line per line of input."
    ),
    TaskDomain.TOOL_USE: (
        'Call one tool per turn with <tool_call>{"name": "...", "arguments": {...}}</tool_call>. '
        "After the last tool result, give the final answer inside <answer></answer> tags."
    ),
}


class PromptBuilder:
    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        *,
        name: str = "rlfactory-chat",
        version: str = "1",
    ) -> None:
        self.system_prompt = system_prompt
        self.template = ComponentRef(name=name, version=version)

    def system_message(self, task: Task) -> Message:
        return Message(
            role=Role.SYSTEM,
            content=f"{self.system_prompt}\n\n{FORMAT_INSTRUCTIONS[task.domain]}",
        )

    def initial_messages(self, task: Task, observation: Observation) -> tuple[Message, ...]:
        """System message plus the environment's first observation as the user turn."""
        return (self.system_message(task), Message(role=Role.USER, content=observation.content))
