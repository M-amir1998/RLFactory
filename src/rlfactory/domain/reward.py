"""Reward records. Rewards are decomposed into named components for debugging and analysis."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, JsonValue, model_validator

from rlfactory.domain.common import ComponentRef, DomainModel


class RewardComponent(StrEnum):
    CORRECTNESS = "correctness"
    FORMAT = "format"
    LENGTH = "length"
    TOOL_COST = "tool_cost"
    SAFETY = "safety"


class RewardBreakdown(DomainModel):
    """Additive reward components.

    Conventions: ``correctness`` and ``format`` are non-negative bonuses; ``length``, ``tool_cost``
    and ``safety`` are non-positive penalties. The scalar reward is the (optionally clipped) sum.
    """

    correctness: float = 0.0
    format: float = 0.0
    length: float = Field(default=0.0, le=0.0)
    tool_cost: float = Field(default=0.0, le=0.0)
    safety: float = Field(default=0.0, le=0.0)

    def total(self) -> float:
        return self.correctness + self.format + self.length + self.tool_cost + self.safety

    def as_dict(self) -> dict[str, float]:
        return {component.value: getattr(self, component.value) for component in RewardComponent}

    def plus(self, other: RewardBreakdown) -> RewardBreakdown:
        return RewardBreakdown(
            correctness=self.correctness + other.correctness,
            format=self.format + other.format,
            length=self.length + other.length,
            tool_cost=self.tool_cost + other.tool_cost,
            safety=self.safety + other.safety,
        )


class RewardResult(DomainModel):
    total: float
    components: RewardBreakdown
    reward_fn: ComponentRef
    clipped: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _finite(self) -> RewardResult:
        if self.total != self.total or self.total in (float("inf"), float("-inf")):
            raise ValueError("reward total must be finite")
        return self
