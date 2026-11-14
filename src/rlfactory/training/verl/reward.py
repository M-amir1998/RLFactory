"""verl custom reward function backed by RLFactory verifiers.

verl 0.9 calls ``compute_score(data_source, solution_str, ground_truth, extra_info=None)`` with
keyword arguments (``reward.custom_reward_function.path`` / ``.name``). ``ground_truth`` holds the
serialised RLFactory task; a dict return value must contain ``"score"`` (other keys are logged).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from rlfactory.domain.task import Task
from rlfactory.training.scoring import CompletionScorer


@lru_cache(maxsize=1)
def _scorer() -> CompletionScorer:
    return CompletionScorer()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    task = Task.model_validate_json(ground_truth)
    reward_name = str((extra_info or {}).get("reward", "binary"))
    verification, reward = _scorer().score(task, solution_str, reward_name)
    components = {f"reward_{k}": v for k, v in reward.components.as_dict().items()}
    return {"score": reward.total, "verified": 1.0 if verification.passed else 0.0, **components}
