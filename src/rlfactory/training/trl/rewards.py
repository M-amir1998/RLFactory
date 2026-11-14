"""GRPO reward functions that delegate to RLFactory verifiers and rewards.

Contract (TRL 1.18, read from ``grpo_trainer.py``): functions are called as
``f(prompts=..., completions=..., completion_ids=..., **dataset_columns, trainer_state=..., ...)``
and return ``list[float | None]``; ``None`` drops the function for that sample. Each RLFactory
reward component becomes its own function with a unique ``__name__``, so TRL logs
``rewards/rlfactory_correctness/mean`` etc. and component values stay visible during training.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from rlfactory.domain.task import Task
from rlfactory.domain.training import RewardFunctionSpec
from rlfactory.training.scoring import CompletionScorer, UnsupportedTaskError

RewardFn = Callable[..., list[float | None]]


@lru_cache(maxsize=8192)
def _parse_task(task_json: str) -> Task:
    return Task.model_validate_json(task_json)


def completion_text(completion: Any) -> str:
    """Conversational completions are message lists; standard ones are strings."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return str(last.get("content", "")) if isinstance(last, dict) else str(last)
    return str(completion)


def _make(component: str, scorer: CompletionScorer) -> RewardFn:
    def reward(
        prompts: Sequence[Any] | None = None,
        completions: Sequence[Any] = (),
        completion_ids: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> list[float | None]:
        tasks = kwargs.get("rlfactory_task")
        if tasks is None:
            raise ValueError("GRPO dataset must include the 'rlfactory_task' column")
        rewards = kwargs.get("rlfactory_reward") or ["binary"] * len(completions)
        values: list[float | None] = []
        for completion, task_json, reward_name in zip(completions, tasks, rewards, strict=True):
            try:
                _, result = scorer.score(
                    _parse_task(task_json), completion_text(completion), reward_name
                )
            except UnsupportedTaskError:
                values.append(None)
                continue
            values.append(
                result.total
                if component == "total"
                else float(getattr(result.components, component))
            )
        return values

    reward.__name__ = f"rlfactory_{component}"
    reward.__qualname__ = reward.__name__
    return reward


def build_reward_functions(
    specs: Sequence[RewardFunctionSpec], scorer: CompletionScorer | None = None
) -> list[RewardFn]:
    if not specs:
        raise ValueError("GRPO needs at least one reward function spec")
    names = [spec.component for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate reward components: {names}")
    shared = scorer or CompletionScorer()
    return [_make(spec.component, shared) for spec in specs]
