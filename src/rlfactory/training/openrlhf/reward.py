"""OpenRLHF reward function backed by RLFactory verifiers.

OpenRLHF 0.11 loads the module attribute ``reward_func(queries, prompts, labels)`` from the file
passed via ``--reward.remote_url``. ``queries`` are prompt + response strings, ``labels`` come from
``--data.label_key`` (the serialised RLFactory task). The documented return value is a dict with
``rewards`` and ``scores`` tensors plus ``extra_logs``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from rlfactory.domain.task import Task
from rlfactory.training.scoring import CompletionScorer


@lru_cache(maxsize=1)
def _scorer() -> CompletionScorer:
    return CompletionScorer()


def response_from_query(query: str, prompt: str) -> str:
    return query[len(prompt) :] if query.startswith(prompt) else query


def score_batch(
    queries: Sequence[str], prompts: Sequence[str], labels: Sequence[str]
) -> tuple[list[float], list[float]]:
    rewards: list[float] = []
    verified: list[float] = []
    for query, prompt, label in zip(queries, prompts, labels, strict=True):
        verification, reward = _scorer().score(
            Task.model_validate_json(label), response_from_query(query, prompt)
        )
        rewards.append(reward.total)
        verified.append(1.0 if verification.passed else 0.0)
    return rewards, verified


def reward_func(
    queries: Sequence[str], prompts: Sequence[str], labels: Sequence[str], **_: Any
) -> dict[str, Any]:
    import torch

    rewards, verified = score_batch(queries, prompts, labels)
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "scores": torch.tensor(verified, dtype=torch.float32),
        "extra_logs": {"rlfactory_verified": torch.tensor(verified, dtype=torch.float32)},
    }
