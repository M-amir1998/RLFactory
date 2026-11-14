"""Config-driven reward registry (named reward specs, versioned by config hash)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from rlfactory.domain.common import stable_hash
from rlfactory.errors import ConfigError, RegistryError
from rlfactory.rewards.functions import (
    BinaryReward,
    CompositeReward,
    FormatReward,
    LengthPenalty,
    RewardFunction,
    ToolCostPenalty,
    WeightedVerifierReward,
)


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BinaryRewardSpec(_Spec):
    type: Literal["binary"] = "binary"
    success: float = 1.0
    failure: float = 0.0


class WeightedVerifierRewardSpec(_Spec):
    type: Literal["weighted_verifier"] = "weighted_verifier"
    weight: float = Field(default=1.0, gt=0.0)


class FormatRewardSpec(_Spec):
    type: Literal["format"] = "format"
    weight: float = Field(default=0.1, ge=0.0)
    penalty: float = Field(default=0.0, ge=0.0)


class LengthPenaltySpec(_Spec):
    type: Literal["length_penalty"] = "length_penalty"
    max_tokens: int = Field(default=1024, ge=1)
    buffer_tokens: int = Field(default=256, ge=1)
    coefficient: float = Field(default=0.5, ge=0.0)


class ToolCostPenaltySpec(_Spec):
    type: Literal["tool_cost"] = "tool_cost"
    cost_per_call: float = Field(default=0.02, ge=0.0)
    error_cost: float = Field(default=0.05, ge=0.0)
    slack_calls: int = Field(default=1, ge=0)
    max_penalty: float = Field(default=0.2, ge=0.0)


class CompositeRewardSpec(_Spec):
    type: Literal["composite"] = "composite"
    components: tuple[str, ...] = Field(min_length=1)
    clip_min: float | None = None
    clip_max: float | None = None
    max_bonus_ratio: float = Field(default=0.5, ge=0.0)
    allow_overshaping: bool = False


RewardSpec = Annotated[
    BinaryRewardSpec
    | WeightedVerifierRewardSpec
    | FormatRewardSpec
    | LengthPenaltySpec
    | ToolCostPenaltySpec
    | CompositeRewardSpec,
    Field(discriminator="type"),
]


class RewardConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    rewards: dict[str, RewardSpec] = Field(default_factory=dict)


DEFAULT_REWARD_SPECS: dict[str, RewardSpec] = {
    "binary": BinaryRewardSpec(),
    "weighted_verifier": WeightedVerifierRewardSpec(),
    "format_bonus": FormatRewardSpec(weight=0.1),
    "length_penalty": LengthPenaltySpec(),
    "tool_cost": ToolCostPenaltySpec(),
    "correctness_format": CompositeRewardSpec(components=("binary", "format_bonus")),
    "rlvr_default": CompositeRewardSpec(
        components=("binary", "format_bonus", "length_penalty", "tool_cost"),
        clip_min=-1.0,
        clip_max=1.1,
    ),
    # Deliberately mis-specified reward for the reward-shaping ablation. Never use for training
    # outside that study; the reward-hacking analyser is expected to flag its trajectories.
    "overshaped_format": FormatRewardSpec(weight=0.6),
    "weak_correctness": WeightedVerifierRewardSpec(weight=0.4),
    "overshaped": CompositeRewardSpec(
        components=("weak_correctness", "overshaped_format"), allow_overshaping=True
    ),
}


class RewardRegistry:
    def __init__(self, specs: Mapping[str, RewardSpec] | None = None) -> None:
        self._specs: dict[str, RewardSpec] = dict(DEFAULT_REWARD_SPECS)
        if specs:
            self._specs.update(specs)
        self._instances: dict[str, RewardFunction] = {}

    def names(self) -> list[str]:
        return sorted(self._specs)

    def register(self, name: str, reward: RewardFunction) -> None:
        self._instances[name] = reward

    def get(self, name: str) -> RewardFunction:
        if name not in self._instances:
            self._instances[name] = self._build(name, ())
        return self._instances[name]

    def _build(self, name: str, stack: tuple[str, ...]) -> RewardFunction:
        if name in stack:
            raise ConfigError(f"reward reference cycle: {' -> '.join((*stack, name))}")
        if name in self._instances:
            return self._instances[name]
        try:
            spec = self._specs[name]
        except KeyError as exc:
            raise RegistryError(
                f"unknown reward {name!r}; known: {', '.join(self.names())}"
            ) from exc
        fp = stable_hash(spec.model_dump(mode="json"), length=8)
        if isinstance(spec, BinaryRewardSpec):
            return BinaryReward(
                name, success=spec.success, failure=spec.failure, config_fingerprint=fp
            )
        if isinstance(spec, WeightedVerifierRewardSpec):
            return WeightedVerifierReward(name, weight=spec.weight, config_fingerprint=fp)
        if isinstance(spec, FormatRewardSpec):
            return FormatReward(
                name, weight=spec.weight, penalty=spec.penalty, config_fingerprint=fp
            )
        if isinstance(spec, LengthPenaltySpec):
            return LengthPenalty(
                name,
                max_tokens=spec.max_tokens,
                buffer_tokens=spec.buffer_tokens,
                coefficient=spec.coefficient,
                config_fingerprint=fp,
            )
        if isinstance(spec, ToolCostPenaltySpec):
            return ToolCostPenalty(
                name,
                cost_per_call=spec.cost_per_call,
                error_cost=spec.error_cost,
                slack_calls=spec.slack_calls,
                max_penalty=spec.max_penalty,
                config_fingerprint=fp,
            )
        children = [self._build(child, (*stack, name)) for child in spec.components]
        return CompositeReward(
            name,
            children,
            clip_min=spec.clip_min,
            clip_max=spec.clip_max,
            max_bonus_ratio=spec.max_bonus_ratio,
            allow_overshaping=spec.allow_overshaping,
            config_fingerprint=fp,
        )
