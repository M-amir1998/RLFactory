"""Scripted policy: returns pre-written responses (per task and turn) or a callable's output."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from rlfactory.domain.trajectory import TokenUsage
from rlfactory.errors import PolicyError
from rlfactory.policies.base import GenerationRequest, GenerationResult, approximate_tokens

Responder = Callable[[GenerationRequest], str]


class ScriptedPolicy:
    simulated = True

    def __init__(
        self,
        responses: Mapping[str, Sequence[str]] | Responder,
        *,
        policy_id: str = "scripted-policy",
        default: str | None = None,
        failures: Mapping[tuple[str, int], Exception] | None = None,
    ) -> None:
        """``responses[task_id][turn]`` is returned for that turn (the last entry repeats).

        ``failures[(task_id, attempt)]`` raises the given exception for that attempt, which lets
        tests script infrastructure faults and retries.
        """
        self.policy_id = policy_id
        self.model: str | None = "rlfactory/scripted-policy"
        self.checkpoint: str | None = None
        self._responses = responses
        self._default = default
        self._failures = dict(failures or {})
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        failure = self._failures.get((request.task_id, request.attempt))
        if failure is not None:
            raise failure
        if callable(self._responses):
            text = self._responses(request)
        else:
            turns = self._responses.get(request.task_id)
            if not turns:
                if self._default is None:
                    raise PolicyError(
                        f"no scripted response for {request.task_id}", retryable=False
                    )
                text = self._default
            else:
                text = turns[min(request.turn, len(turns) - 1)]
        return GenerationResult(
            text=text,
            finish_reason="stop",
            usage=TokenUsage(completion_tokens=approximate_tokens(text), estimated=True),
            model=self.model,
        )

    async def aclose(self) -> None:
        return None
