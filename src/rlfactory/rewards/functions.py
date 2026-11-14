"""Reward functions.

Verification answers *whether* the task was solved; a reward function decides *what number* the
learner sees. Every function returns a :class:`RewardBreakdown` so components stay visible.

Bounds (documented and exposed through :meth:`bounds`):

=====================  ===============================  ====================
Function               Component                        Range
=====================  ===============================  ====================
BinaryReward           correctness                      {failure, success}
WeightedVerifierReward correctness                      [0, weight]
FormatReward           format                           [-penalty, weight]
LengthPenalty          length                           [-coefficient, 0]
ToolCostPenalty        tool_cost                        [-max_penalty, 0]
CompositeReward        sum of children, then clipped    [clip_min, clip_max]
=====================  ===============================  ====================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Protocol

from typing_extensions import override

from rlfactory.domain.common import ComponentRef
from rlfactory.domain.reward import RewardBreakdown, RewardComponent, RewardResult
from rlfactory.domain.task import Task, ToolUsePayload
from rlfactory.domain.trajectory import Trajectory
from rlfactory.domain.verification import VerificationResult
from rlfactory.errors import ConfigError

Bounds = Mapping[RewardComponent, tuple[float, float]]


class RewardFunction(Protocol):
    name: str
    version: str

    def compute(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardResult: ...

    def bounds(self) -> Bounds: ...


class BaseReward(ABC):
    implementation_version = "1.0.0"

    def __init__(self, name: str, config_fingerprint: str = "default") -> None:
        self.name = name
        self.version = f"{self.implementation_version}+{config_fingerprint}"

    @property
    def ref(self) -> ComponentRef:
        return ComponentRef(name=self.name, version=self.version)

    @abstractmethod
    def breakdown(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardBreakdown: ...

    @abstractmethod
    def bounds(self) -> Bounds: ...

    def compute(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardResult:
        parts = self.breakdown(task, trajectory, verification)
        return RewardResult(total=parts.total(), components=parts, reward_fn=self.ref)


def format_ok(trajectory: Trajectory) -> bool:
    """The environment could parse a final answer (or code block) from the last action."""
    observation = trajectory.final_observation
    return (
        observation is not None
        and observation.parse_error is None
        and observation.parsed_answer is not None
    )


def completion_length(trajectory: Trajectory) -> int:
    """Completion tokens if reported, otherwise a chars/4 approximation."""
    tokens = trajectory.completion_tokens
    if tokens > 0:
        return tokens
    return sum(len(action.text) for action in trajectory.actions) // 4


class BinaryReward(BaseReward):
    def __init__(
        self,
        name: str = "binary",
        *,
        success: float = 1.0,
        failure: float = 0.0,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        self.success = success
        self.failure = failure

    @override
    def breakdown(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardBreakdown:
        return RewardBreakdown(correctness=self.success if verification.passed else self.failure)

    @override
    def bounds(self) -> Bounds:
        low, high = sorted((self.success, self.failure))
        return {RewardComponent.CORRECTNESS: (low, high)}


class WeightedVerifierReward(BaseReward):
    """``weight * verification.score``: keeps partial credit (e.g. fraction of tests passed)."""

    def __init__(
        self,
        name: str = "weighted_verifier",
        *,
        weight: float = 1.0,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        if weight <= 0:
            raise ConfigError("weight must be positive")
        self.weight = weight

    @override
    def breakdown(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardBreakdown:
        return RewardBreakdown(correctness=self.weight * verification.score)

    @override
    def bounds(self) -> Bounds:
        return {RewardComponent.CORRECTNESS: (0.0, self.weight)}


class FormatReward(BaseReward):
    def __init__(
        self,
        name: str = "format",
        *,
        weight: float = 0.1,
        penalty: float = 0.0,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        if weight < 0 or penalty < 0:
            raise ConfigError("format weight and penalty must be non-negative")
        self.weight = weight
        self.penalty = penalty

    @override
    def breakdown(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardBreakdown:
        return RewardBreakdown(format=self.weight if format_ok(trajectory) else -self.penalty)

    @override
    def bounds(self) -> Bounds:
        return {RewardComponent.FORMAT: (-self.penalty, self.weight)}


class LengthPenalty(BaseReward):
    """Soft over-length penalty (DAPO-style): zero below ``max_tokens - buffer_tokens``, linear
    inside the buffer, ``-coefficient`` beyond ``max_tokens``. It never rewards brevity itself,
    which avoids pushing policies towards pathologically short answers."""

    def __init__(
        self,
        name: str = "length_penalty",
        *,
        max_tokens: int = 1024,
        buffer_tokens: int = 256,
        coefficient: float = 0.5,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        if not 0 < buffer_tokens <= max_tokens:
            raise ConfigError("need 0 < buffer_tokens <= max_tokens")
        if coefficient < 0:
            raise ConfigError("coefficient must be non-negative")
        self.max_tokens = max_tokens
        self.buffer_tokens = buffer_tokens
        self.coefficient = coefficient

    def penalty(self, length: int) -> float:
        threshold = self.max_tokens - self.buffer_tokens
        if length <= threshold:
            return 0.0
        if length <= self.max_tokens:
            return -self.coefficient * (length - threshold) / self.buffer_tokens
        return -self.coefficient

    @override
    def breakdown(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardBreakdown:
        return RewardBreakdown(length=self.penalty(completion_length(trajectory)))

    @override
    def bounds(self) -> Bounds:
        return {RewardComponent.LENGTH: (-self.coefficient, 0.0)}


class ToolCostPenalty(BaseReward):
    """Penalise tool calls beyond a known-good plan (plus slack) and failed calls."""

    def __init__(
        self,
        name: str = "tool_cost",
        *,
        cost_per_call: float = 0.02,
        error_cost: float = 0.05,
        slack_calls: int = 1,
        max_penalty: float = 0.2,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        if min(cost_per_call, error_cost, max_penalty) < 0 or slack_calls < 0:
            raise ConfigError("tool cost parameters must be non-negative")
        self.cost_per_call = cost_per_call
        self.error_cost = error_cost
        self.slack_calls = slack_calls
        self.max_penalty = max_penalty

    @override
    def breakdown(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardBreakdown:
        records = trajectory.tool_records()
        if not records:
            return RewardBreakdown()
        payload = task.payload
        allowance = (
            len(payload.reference_plan) if isinstance(payload, ToolUsePayload) else 0
        ) + self.slack_calls
        errors = sum(1 for record in records if not record.succeeded)
        raw = self.cost_per_call * max(0, len(records) - allowance) + self.error_cost * errors
        return RewardBreakdown(tool_cost=-min(self.max_penalty, raw))

    @override
    def bounds(self) -> Bounds:
        return {RewardComponent.TOOL_COST: (-self.max_penalty, 0.0)}


def _sum_bounds(children: Sequence[RewardFunction]) -> dict[RewardComponent, tuple[float, float]]:
    totals: dict[RewardComponent, tuple[float, float]] = {}
    for child in children:
        for component, (low, high) in child.bounds().items():
            current = totals.get(component, (0.0, 0.0))
            totals[component] = (current[0] + low, current[1] + high)
    return totals


class CompositeReward(BaseReward):
    """Sum of child breakdowns, optionally clipped.

    Guard against over-shaping: the maximum positive non-correctness bonus may not exceed
    ``max_bonus_ratio`` x the maximum correctness reward unless ``allow_overshaping`` is set
    explicitly (only for ablations that study reward hacking).
    """

    def __init__(
        self,
        name: str,
        children: Sequence[RewardFunction],
        *,
        clip_min: float | None = None,
        clip_max: float | None = None,
        max_bonus_ratio: float = 0.5,
        allow_overshaping: bool = False,
        config_fingerprint: str = "default",
    ) -> None:
        super().__init__(name, config_fingerprint)
        if not children:
            raise ConfigError(f"composite reward {name!r} needs children")
        if clip_min is not None and clip_max is not None and clip_min >= clip_max:
            raise ConfigError("clip_min must be < clip_max")
        self.children = tuple(children)
        self.clip_min = clip_min
        self.clip_max = clip_max
        totals = _sum_bounds(self.children)
        correctness_high = totals.get(RewardComponent.CORRECTNESS, (0.0, 0.0))[1]
        bonus_high = sum(
            max(0.0, high)
            for component, (_, high) in totals.items()
            if component is not RewardComponent.CORRECTNESS
        )
        if not allow_overshaping:
            if correctness_high <= 0:
                raise ConfigError(f"composite reward {name!r} has no positive correctness signal")
            if bonus_high > max_bonus_ratio * correctness_high:
                raise ConfigError(
                    f"composite reward {name!r} is over-shaped: non-correctness bonus up to "
                    f"{bonus_high:g} vs correctness up to {correctness_high:g} "
                    f"(limit ratio {max_bonus_ratio:g}); set allow_overshaping for ablations"
                )
        self.overshaped = bonus_high > max_bonus_ratio * max(correctness_high, 0.0)

    @override
    def breakdown(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardBreakdown:
        total = RewardBreakdown()
        for child in self.children:
            total = total.plus(child.compute(task, trajectory, verification).components)
        return total

    @override
    def compute(
        self, task: Task, trajectory: Trajectory, verification: VerificationResult
    ) -> RewardResult:
        children = {
            child.name: child.compute(task, trajectory, verification) for child in self.children
        }
        parts = RewardBreakdown()
        for result in children.values():
            parts = parts.plus(result.components)
        raw = parts.total()
        clipped = raw
        if self.clip_min is not None:
            clipped = max(self.clip_min, clipped)
        if self.clip_max is not None:
            clipped = min(self.clip_max, clipped)
        return RewardResult(
            total=clipped,
            components=parts,
            reward_fn=self.ref,
            clipped=clipped != raw,
            details={name: result.total for name, result in children.items()},
        )

    @override
    def bounds(self) -> Bounds:
        return _sum_bounds(self.children)
