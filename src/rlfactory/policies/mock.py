"""Deterministic mock policy for CPU-only tests and pipeline simulation.

Behaviour is a pure function of ``(seed, task_id, sample_index, turn, attempt)``:

* the episode outcome (correct / wrong / malformed, verbose or not) depends only on
  ``(seed, task_id, sample_index)``, so retries reproduce the same answer;
* simulated infrastructure faults (timeouts, server errors) also depend on ``attempt``, so a retry
  can succeed, exactly like a flaky server.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from random import Random

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.domain.common import stable_seed
from rlfactory.domain.task import Task, TaskDomain
from rlfactory.domain.trajectory import TokenUsage
from rlfactory.errors import PolicyError, PolicyTimeoutError
from rlfactory.policies.base import GenerationRequest, GenerationResult, approximate_tokens
from rlfactory.policies.oracle import ResponseMode, ResponseOracle


class MockPolicyProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    correct_probability: float = Field(default=0.6, ge=0.0, le=1.0)
    domain_correct_probability: dict[TaskDomain, float] = Field(default_factory=dict)
    malformed_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    verbose_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    timeout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    error_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_s: float = Field(default=0.0, ge=0.0)
    latency_jitter_s: float = Field(default=0.0, ge=0.0)

    def p_correct(self, domain: TaskDomain) -> float:
        return self.domain_correct_probability.get(domain, self.correct_probability)


class MockPolicy:
    simulated = True

    def __init__(
        self,
        tasks: Iterable[Task],
        profile: MockPolicyProfile | None = None,
        *,
        policy_id: str = "mock-policy",
        seed: int = 0,
        checkpoint: str | None = None,
    ) -> None:
        self.policy_id = policy_id
        self.model: str | None = "rlfactory/mock-policy"
        self.checkpoint = checkpoint
        self.profile = profile or MockPolicyProfile()
        self.seed = seed
        self._tasks = {task.task_id: task for task in tasks}
        self._oracle = ResponseOracle()

    def add_tasks(self, tasks: Iterable[Task]) -> None:
        self._tasks.update({task.task_id: task for task in tasks})

    def episode_mode(self, task: Task, sample_index: int) -> tuple[ResponseMode, bool]:
        # Common random numbers: the outcome draw depends on (seed, task, sample) only, never on the
        # policy id, so a profile with higher p_correct is correct on a superset of samples and
        # simulated comparisons are not dominated by independent sampling noise.
        rng = Random(stable_seed(self.seed, task.task_id, sample_index, "episode"))
        roll = rng.random()
        verbose = rng.random() < self.profile.verbose_probability
        malformed = self.profile.malformed_probability
        if roll < malformed:
            return ResponseMode.MALFORMED, verbose
        if roll < malformed + (1 - malformed) * self.profile.p_correct(task.domain):
            return ResponseMode.CORRECT, verbose
        return ResponseMode.WRONG, verbose

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        infra = Random(
            stable_seed(
                self.seed,
                request.task_id,
                request.sample_index,
                request.turn,
                request.attempt,
                "infra",
            )
        )
        delay = self.profile.latency_s + infra.uniform(0.0, self.profile.latency_jitter_s)
        if delay > 0:
            await asyncio.sleep(delay)
        if infra.random() < self.profile.timeout_probability:
            raise PolicyTimeoutError("simulated policy timeout")
        if infra.random() < self.profile.error_probability:
            raise PolicyError("simulated server error (HTTP 503)", retryable=True)
        task = self._tasks.get(request.task_id)
        if task is None:
            raise PolicyError(
                f"mock policy has no oracle entry for {request.task_id}", retryable=False
            )
        mode, verbose = self.episode_mode(task, request.sample_index)
        style = Random(
            stable_seed(self.seed, request.task_id, request.sample_index, request.turn, "style")
        )
        text = self._oracle.respond(task, mode, verbose=verbose, turn=request.turn, rng=style)
        prompt_tokens = sum(approximate_tokens(m.content) for m in request.messages)
        return GenerationResult(
            text=text,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=approximate_tokens(text),
                estimated=True,
            ),
            latency_s=time.perf_counter() - started,
            model=self.model,
        )

    async def aclose(self) -> None:
        return None
