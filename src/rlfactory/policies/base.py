"""Policy protocol and request/response records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field

from rlfactory.domain.common import DomainModel
from rlfactory.domain.model import GenerationConfig
from rlfactory.domain.trajectory import Message, Role, TokenUsage


class GenerationRequest(DomainModel):
    request_id: str
    messages: tuple[Message, ...] = Field(min_length=1)
    config: GenerationConfig
    seed: int
    task_id: str
    """Only simulation policies (mock/scripted) may use this; real policies must ignore it."""
    sample_index: int = 0
    turn: int = 0
    attempt: int = 1


class GenerationResult(DomainModel):
    text: str
    finish_reason: str | None = None
    usage: TokenUsage = TokenUsage()
    latency_s: float = 0.0
    model: str | None = None


class Policy(Protocol):
    policy_id: str
    model: str | None
    checkpoint: str | None
    simulated: bool

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Raise :class:`rlfactory.errors.PolicyTimeoutError` / :class:`PolicyError` on failure."""
        ...

    async def aclose(self) -> None: ...


TOOL_RESULT_PREFIX = "Tool result:\n"


def to_chat_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    """Convert to plain chat dicts accepted by every OpenAI-compatible server and chat template.

    RLFactory tool observations are plain text, not OpenAI function-call results (which require a
    ``tool_call_id``), so they are sent as user turns with an explicit prefix. Dataset adapters use
    the same conversion so training data matches what the policy saw during rollouts.
    """
    converted: list[dict[str, str]] = []
    for message in messages:
        if message.role is Role.TOOL:
            converted.append({"role": "user", "content": TOOL_RESULT_PREFIX + message.content})
        else:
            converted.append({"role": message.role.value, "content": message.content})
    return converted


def approximate_tokens(text: str) -> int:
    """Whitespace token count, used only when a backend reports no usage."""
    return len(text.split())
